#!/usr/bin/env python3
"""
Coverage maps, simple version.

Three inputs, because the question is whether the APIs already tested cover
the APIs you might test and the APIs you will finally be judged on:

  --input    Formulation_example_input_file.xlsx   formulations run so far
  --apis     API_information_for_selection.xlsx    the candidate list
  --holdout  FPN wet SDD stability data.xlsx       the final test set

Six figures (no 2_pca.png: the single-panel PCA was the
structural half of 6_pca_comparison.png):
  1_similarity.png      ECFP4 Tanimoto matrix over every API, clustered
  3_umap.png            UMAP on ECFP4 at three n_neighbors settings
  4_design_grid.png     API x polymer, loading on x, outcome by colour
  5_reach.png           similarity to the nearest *tested* API, two panels
  6_pca_comparison.png  structural vs physchem side by side
  7_measured.png        measured thermal / ADMET strips + ionisation class

Features are grouped by provenance and kind, so the groups are disjoint:

  structural        graph descriptors          -> PCA
  physchem          computed properties and    -> PCA
                    charge distribution
  thermal_measured  Tm, Tg, Tm/Tg              -> strip plot
  admet_measured    S+logP, S+Peff, ionisation -> strip plot

Fingerprints stay out of the PCAs: a sparse 2048-bit vector over a few dozen
molecules gives a projection dominated by bit sparsity, and Euclidean
distance on bits is the wrong metric. They do their work in figures 1 and 3,
which use Jaccard.

Held-out APIs enter by structure only -- no labels are read from that file,
and the scaler, PCA and UMAP are fitted WITHOUT them and then used to project
them in. So the axes are not defined by the test set.

That keeps the map honest, but it cannot stop the real leak, which is acting
on what you see. If this map leads you to add training APIs, change features
or retune a threshold, the held-out set is no longer held out. Read it; do not
optimise against it.

Usage:
    python coverage_simple.py \\
        --input   "../data/raw/Formulation_example_input_file.xlsx" \\
        --apis    "../data/raw/API_information_for_selection.xlsx" \\
        --holdout "../data/raw/FPN wet SDD stability data.xlsx"

--apis and --holdout are both optional; with neither, it maps the tested set.

Needs rdkit, scikit-learn, umap-learn, scipy, pandas, matplotlib. Does not
need the TeNNet-SAC stack.
"""
import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

# Typographic minus, held in a constant: a backslash escape is not allowed
# inside an f-string expression before Python 3.12.
MINUS = "\u2212"

STABLE, UNSTABLE = "#2a78d6", "#d03b3b"   # blue/red: CVD-safe, unlike green/red
ACCENT, MUTED, GRID, INK = "#eb6834", "#52514e", "#e4e3de", "#0b0b0b"

# role -> (marker, edge colour, label). Shape as well as fill, so the three
# groups stay distinguishable in greyscale and for colour-blind readers.
ROLES = {
    "tested":   ("o", "white", "tested (training)"),
    "candidate": ("o", MUTED, "candidate, untested"),
    "holdout":  ("^", INK, "held-out test set"),
}

RENAME = {"API": "api", "SMILES": "smiles", "Polymer": "polymer_name",
          "API_loading": "api_loading_percent", "truth": "target"}

# Graph descriptors computed from SMILES. PEOE_VSA1-14 are included because
# they are reduced below into charge-distribution fractions.
DESCRIPTORS = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "RingCount", "FractionCSP3", "HeavyAtomCount", "BertzCT", "LabuteASA",
    "Chi0n", "Chi1n", "Chi2n", "Kappa1", "Kappa2", "Kappa3",
] + [f"PEOE_VSA{i}" for i in range(1, 15)]

# RDKit bins PEOE surface area by Gasteiger charge at these edges.
# Low bins are electron-rich surface, high bins electron-poor. Note this runs
# OPPOSITE to a COSMO sigma profile, where positive sigma is electron-rich.
PEOE_NEGATIVE = [f"PEOE_VSA{i}" for i in range(1, 7)]    # charge < -0.05
PEOE_NEUTRAL = ["PEOE_VSA7", "PEOE_VSA8"]                # -0.05 .. +0.05
PEOE_POSITIVE = [f"PEOE_VSA{i}" for i in range(9, 15)]   # charge > +0.05

# Measured / ADMET columns read from the input files. Spellings differ between
# files, so each is matched on a normalised name: "S+logP" -> "slogp".
MEASURED = {
    "tm_deg_c": ["apitmdegreesc", "apitmdegc", "tmdegreesc", "tmdegc",
                 "meltingpoint", "tm"],
    "tg_deg_c": ["apitgdegreesc", "apitgdegc", "tgdegreesc", "tgdegc",
                 "glasstransition", "tg"],
    "logp": ["splogp", "slogp", "apilogp", "logp"],
    "permeability": ["speff", "sppeff", "apipermeability", "permeability",
                     "peff"],
    "acidic_pka_min": ["sacidicpkamin", "sacidicpka", "spacidicpkamin",
                       "spacidicpka", "acidicpkamin", "acidicpkalowest",
                       "acidicpka"],
    "basic_pka_max": ["sbasicpkamax", "sbasicpka", "spbasicpkamax",
                      "spbasicpka", "basicpkamax", "basicpkahighest",
                      "basicpka"],
}

# Fallback for columns whose exact spelling is not listed above: match on
# required tokens, then prefer the variant that matches the aggregation we
# want. ADMET exports vary ("S+Acidic_pKa", "S+Acidic_pKa_min",
# "Acidic_pKa_lowest"), so exact lists alone keep going stale.
MEASURED_TOKENS = {
    "acidic_pka_min": (("acidic", "pka"), ("min", "lowest")),
    "basic_pka_max": (("basic", "pka"), ("max", "highest")),
    "permeability": (("peff",), ()),
    "tm_deg_c": (("tm",), ("deg", "melt")),
    "tg_deg_c": (("tg",), ("deg", "glass")),
}

IONISATION = ["acid", "weak_base", "strong_base", "zwitterion"]

# Four groups, partitioned by provenance and kind rather than by provenance
# alone, so the sets are disjoint and each answers its own question.
#
# structural        graph-derived: connectivity, shape, rings, complexity
# physchem          computed physicochemical, incl. charge distribution
# thermal_measured  measured Tm / Tg
# admet_measured    ADMET Predictor outputs and the ionisation class
#
# The two computed groups get PCA panels; the two measured groups are shown
# as strip plots, since three or four correlated features give a PCA nothing
# to do. tm_div_tg_k is deliberately absent from any PCA -- it is a function
# of Tm and Tg, so including all three double-counts -- but is kept for the
# strips, where it is the quantity formulators actually read.
# Pruned against the collinearity report rather than by taste:
#   HeavyAtomCount ~ Chi1n 0.97, ~ Chi2n 0.93   -> keep HeavyAtomCount only
#   Kappa2 ~ Kappa3 0.92                        -> keep Kappa2
#   MolWt ~ LabuteASA 0.99                      -> keep MolWt
# peoe_neutral_frac is computed in descriptor_frame but deliberately left
# out of this group: the three fractions sum to one, so any two determine
# the third. Keeping all three would spend a PCA dimension on an exact
# linear dependency that pairwise correlation does not necessarily flag.
FEATURE_SETS = {
    "structural": ["HeavyAtomCount", "BertzCT", "Kappa2", "RingCount",
                   "NumAromaticRings", "NumAliphaticRings", "FractionCSP3"],
    "physchem": ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
                 "NumRotatableBonds", "peoe_negative_frac",
                 "peoe_positive_frac"],
}

STRIP_SETS = {
    "thermal_measured": ["tm_deg_c", "tg_deg_c", "tm_div_tg_k"],
    "admet_measured": ["logp", "permeability"],
}

# Each PCA group keeps exactly one size descriptor (HeavyAtomCount,
# MolWt), so PC1 is not "how big is it" three times over. The per-group
# collinearity report keeps this auditable as features are added.
COMPARE_DEFAULT = ("structural", "physchem")

SET_TITLES = {
    "structural": ("Structural", "graph descriptors · label-free"),
    "physchem": ("Physicochemical", "computed properties + charge "
                 "distribution · label-free"),
}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                     "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


# --- loading ---------------------------------------------------------------
def canonical_parent(smiles):
    """
    Canonical SMILES of the largest fragment.

    Counterions must go before any descriptor is computed: with a salt in the
    SMILES, MaxPartialCharge becomes the counterion's formal charge (+1 for
    Li, +2 for Zn) and MolWt counts mass the API does not have.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol)


def ionisation_flags(acidic_pka_min, basic_pka_max):
    """
    acid / weak_base / strong_base / zwitterion from pKa.

    Transcribed from molecule_instance.compute_acid_base_zwitterion in
    ads_molecule_classes.py (thresholds 6.5 and 2, attributed there to Mike
    Morgen), rather than imported -- importing that module pulls in torch and
    tennetsac, which this script otherwise does not need.
    """
    acid_threshold, base_lower, base_upper = 6.5, 2.0, 6.5
    acid = weak_base = strong_base = zwitterion = 0

    has_acid = acidic_pka_min is not None and not pd.isna(acidic_pka_min)
    has_base = basic_pka_max is not None and not pd.isna(basic_pka_max)

    if has_acid and acidic_pka_min < acid_threshold:
        acid = 1
    if has_base:
        if basic_pka_max > base_upper:
            strong_base = 1
        elif basic_pka_max > base_lower:
            weak_base = 1
        if basic_pka_max > base_lower and has_acid:
            if (acidic_pka_min < acid_threshold
                    and acidic_pka_min < basic_pka_max):
                zwitterion = 1
    return {"acid": acid, "weak_base": weak_base,
            "strong_base": strong_base, "zwitterion": zwitterion}


def _parse_record(text):
    """
    Parse one cell holding a record. Accepts JSON and Python-literal dicts,
    and unwraps a single {"input_data": {...}} envelope.
    """
    if not isinstance(text, str) or "{" not in text:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        try:
            obj = ast.literal_eval(text)
        except Exception:
            return None
    if isinstance(obj, dict) and len(obj) == 1 and "input_data" in obj:
        obj = obj["input_data"]
    return obj if isinstance(obj, dict) else None


def _expand_record_columns(df):
    """
    Expand any column whose cells hold a record dict into real columns.

    The held-out workbook keeps each row as JSON under `input_data`, so the
    fields the rest of this script looks for do not exist as columns until
    this runs. Expanded fields win on a name collision, since they are the
    detailed ones.
    """
    for col in list(df.columns):
        series = df[col]
        # Not `dtype == object`: pandas 3 gives text columns a dedicated
        # string dtype, so test by exclusion instead.
        if (pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_bool_dtype(series)
                or pd.api.types.is_datetime64_any_dtype(series)):
            continue
        parsed = series.map(_parse_record)
        if parsed.notna().sum() < max(1, len(series) * 0.5):
            continue
        flat = pd.json_normalize(
            [r if isinstance(r, dict) else {} for r in parsed])
        flat.index = df.index
        if flat.empty:
            continue
        print(f"note: expanded '{col}' into {flat.shape[1]} column(s)")
        df = df.drop(columns=[col]).drop(
            columns=[c for c in flat.columns if c in df.columns])
        df = pd.concat([df, flat], axis=1)
    return df


def _read_raw(path, sheet, header):
    path = Path(path)
    df = (pd.read_csv(path, header=header) if path.suffix.lower() == ".csv"
          else pd.read_excel(path, sheet_name=sheet, header=header))
    return _expand_record_columns(df)


def _has_columns(df, required):
    """True when every required token group resolves to some column."""
    for options, contains in required:
        try:
            _find_column(df, options, contains)
        except KeyError:
            return False
    return True


def _read(path, sheet=0, header=None, required=(), max_scan=6):
    """
    Read a sheet, finding the header row when it is not the first.

    Title rows and merged cells above the real header are common in these
    workbooks, and the three input files do not agree on how many there are,
    so the row is detected per file rather than set globally. header= forces
    a row and skips detection.
    """
    if header is not None:
        return _read_raw(path, sheet, header)

    for candidate in range(max_scan):
        try:
            df = _read_raw(path, sheet, candidate)
        except Exception:
            continue
        if not required or _has_columns(df, required):
            if candidate:
                print(f"note: {Path(path).name} -- header found on row "
                      f"{candidate + 1}")
            return df

    # Nothing matched; return row 0 so the caller raises the informative error.
    return _read_raw(path, sheet, 0)


def _find_column(df, options, contains=None):
    """
    Locate a column by exact name or substring, ignoring case.

    Column names are coerced with str() because a spreadsheet header can come
    back as a float or a NaN when the header row is not where pandas assumed.
    """
    lower = {str(c).lower().strip(): c for c in df.columns
             if len(str(c)) < 80 and "{" not in str(c)}
    for opt in options:
        if opt in lower:
            return lower[opt]
    if contains:
        for low, orig in lower.items():
            if contains in low:
                return orig
    raise KeyError(
        f"No column matching {options} in {list(df.columns)}. "
        "If those names look like numbers or 'Unnamed: N', the header row is "
        "not the first row -- rerun with --header 1 (or 2)."
    )


def _normalise(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


PKA_COLUMNS = {"acidic_pka_min": "min", "basic_pka_max": "max"}


def parse_pka(value, mode):
    """
    One pKa number from a cell that may hold several.

    ADMET Predictor emits zero, one or many pKa values per molecule, written
    into a single cell as "3.5; 9.8". pd.to_numeric turns that into NaN,
    which silently zeroes every ionisation flag. Mirrors the intent of
    molecule_instance.create_list_from_str: lowest acidic, highest basic.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        parts = re.split(r"[;,|/]", str(value))
    nums = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            nums.append(float(part))
        except ValueError:
            found = re.findall(r"-?\d+\.?\d*", part)
            nums.extend(float(f) for f in found)
    if not nums:
        return np.nan
    return min(nums) if mode == "min" else max(nums)


def extract_measured(df):
    """
    Pull Tm, Tg, logP and permeability out of whatever they are called here.

    Returns a frame with only the columns that were found, so a missing
    property degrades the affected feature set rather than crashing.
    """
    lookup = {_normalise(c): c for c in df.columns}

    def resolve(target, options):
        for opt in options:
            if opt in lookup:
                return lookup[opt]
        required, preferred = MEASURED_TOKENS.get(target, ((), ()))
        if not required:
            return None
        hits = [orig for low, orig in lookup.items()
                if all(tok in low for tok in required)]
        if not hits:
            return None
        for tok in preferred:
            for orig in hits:
                if tok in _normalise(orig):
                    return orig
        return hits[0]

    found, report = {}, {}
    for target, options in MEASURED.items():
        col = resolve(target, options)
        if col is not None:
                raw = df[col]
                if target in PKA_COLUMNS:
                    values = raw.map(lambda v: parse_pka(v, PKA_COLUMNS[target]))
                else:
                    values = pd.to_numeric(raw, errors="coerce")
                found[target] = values
                report[target] = (col, int(values.notna().sum()),
                                  len(values), raw.dropna().iloc[0]
                                  if raw.notna().any() else None)
    extract_measured.last_report = report
    return pd.DataFrame(found, index=df.index)


def report_measured(path, report):
    """Say which measured columns were found, and whether they parsed."""
    name = Path(path).name
    for target in MEASURED:
        if target not in report:
            print(f"  {name}: '{target}' column NOT FOUND")
            continue
        col, ok, total, sample = report[target]
        if ok == 0:
            print(f"  {name}: '{target}' found as '{col}' but 0/{total} "
                  f"parsed; first raw value: {sample!r}")
        elif ok < total:
            print(f"  {name}: '{target}' from '{col}' -- {ok}/{total} parsed")


FORMULATION_REQUIRED = [(["api", "api_name"], "api"),
                        (["smiles"], "smiles"),
                        (["polymer", "polymer_name"], "polymer")]
API_LIST_REQUIRED = [(["api", "api_name", "name", "compound", "molecule"],
                      "api"),
                     (["smiles"], "smiles")]


def load_formulations(path, sheet=0, header=None):
    """The formulation table: one row per API x polymer x loading."""
    raw = _read(path, sheet, header, FORMULATION_REQUIRED)
    df = raw.rename(columns={k: v for k, v in RENAME.items()})
    df["smiles"] = [canonical_parent(s) for s in df["smiles"]]
    df = df.dropna(subset=["smiles"])
    measured = extract_measured(raw)
    report_measured(path, extract_measured.last_report)
    for col, values in measured.items():
        df[col] = values.values
    return df


def load_api_list(path, sheet=0, header=None):
    """
    A table of APIs: name plus SMILES. Column names vary between the
    selection file and the held-out file, so both are located by pattern.
    Only the structure is read -- never an outcome column.
    """
    df = _read(path, sheet, header, API_LIST_REQUIRED)
    name_col = _find_column(df, ["api", "api_name", "name", "compound",
                                 "molecule"], contains="api")
    smi_col = _find_column(df, ["smiles"], contains="smiles")
    out = pd.DataFrame({
        "api": df[name_col].astype(str).str.strip(),
        "smiles": [canonical_parent(s) for s in df[smi_col]],
    })
    measured = extract_measured(df)
    report_measured(path, extract_measured.last_report)
    for col, values in measured.items():
        out[col] = values.values
    out = out.dropna(subset=["smiles"])
    return out.drop_duplicates("api").set_index("api").sort_index()


# --- overlap ---------------------------------------------------------------
def inchikeys(smiles):
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        out.append(Chem.MolToInchiKey(mol) if mol else None)
    return pd.Series(out, index=smiles.index if hasattr(smiles, "index") else None)


def _keyed(smiles_by_name):
    """name -> (InChIKey, skeleton) for one input group."""
    out = {}
    for name, smi in smiles_by_name.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        key = Chem.MolToInchiKey(mol)
        out[name] = (key, key[:14])
    return out


def _describe(a, b, index):
    """Names on both sides of a shared key, as 'x / y' when they differ."""
    pairs = []
    for name_a, (key_a, skel_a) in a.items():
        for name_b, (key_b, skel_b) in b.items():
            if (key_a, skel_a)[index] == (key_b, skel_b)[index]:
                pairs.append(name_a if name_a == name_b
                             else f"{name_a} / {name_b}")
    return sorted(set(pairs))


def report_overlap(formulations, candidates, holdout):
    """
    Compare the RAW input groups by structure, before any de-duplication.

    This has to run on the files as loaded: if an API were dropped from the
    held-out group by name precedence first, the contamination it represents
    would never be reported. Names are not comparable across files -- synonyms,
    salt forms and stereochemistry variants differ -- so matching is on
    InChIKey, with the first 14 characters (the skeleton block) catching salt,
    stereo and tautomer variants of the same scaffold.
    """
    if holdout is None or not len(holdout):
        return
    tested = _keyed(formulations.groupby("api")["smiles"].first().to_dict())
    held = _keyed(holdout["smiles"].to_dict())
    cand = _keyed(candidates["smiles"].to_dict()) if candidates is not None else {}

    print("\nstructure overlap check (InChIKey, not name)")

    exact = _describe(tested, held, 0)
    skel = [p for p in _describe(tested, held, 1) if p not in exact]
    if exact:
        print(f"  !! {len(exact)} structure(s) in BOTH training and held-out: "
              f"{', '.join(exact)}")
        print("     the final test is contaminated for these -- drop them from "
              "training or report that subset separately")
    else:
        print("  ok: no structure appears in both training and held-out")
    if skel:
        print(f"  ?  {len(skel)} same-skeleton pair(s) (salt / stereo / "
              f"tautomer variants): {', '.join(skel)}")
        print("     check whether these should count as the same compound")

    if cand:
        both = _describe(cand, held, 0)
        if both:
            print(f"  i  {len(both)} structure(s) in both the candidate list "
                  f"and the held-out set: {', '.join(both)}")
            print("     fine as long as they are never added to training")


# --- assembly --------------------------------------------------------------
def build_api_table(formulations, candidates=None, holdout=None):
    """
    One row per API with a role. Precedence is tested > candidate > holdout:
    an API with formulations is training data whatever else it appears in,
    which is exactly the case the overlap check needs to flag.
    """
    measured = [c for c in MEASURED if c in formulations.columns]
    frames = []
    tested = formulations.groupby("api").agg(
        smiles=("smiles", "first"),
        **{c: (c, "first") for c in measured})
    tested["role"] = "tested"
    frames.append(tested)
    for df, role in ((candidates, "candidate"), (holdout, "holdout")):
        if df is not None and len(df):
            d = df.copy()
            d["role"] = role
            frames.append(d)

    apis = pd.concat(frames)
    apis = apis[~apis.index.duplicated(keep="first")].sort_index()
    if "target" in formulations:
        apis["unstable_frac"] = formulations.groupby("api")["target"].mean()
    return apis


def fingerprints(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def tanimoto(fps):
    n = len(fps)
    m = np.eye(n)
    for i in range(n - 1):
        row = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        m[i, i + 1:] = m[i + 1:, i] = row
    return m


def descriptor_frame(apis):
    """RDKit descriptors, charge fractions, measured properties, ionisation."""
    rows = [{d: Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s)).get(d)
             for d in DESCRIPTORS} for s in apis["smiles"]]
    out = pd.DataFrame(rows, index=apis.index)

    # Charge distribution as three fractions of surface area. Preferred over
    # Max/MinPartialCharge, which are two order statistics of the same
    # quantity and go NaN on elements Gasteiger has no parameters for.
    peoe_total = out[[c for c in out.columns
                      if c.startswith("PEOE_VSA")]].sum(axis=1)
    safe = peoe_total.replace(0, np.nan)
    out["peoe_negative_frac"] = out[PEOE_NEGATIVE].sum(axis=1) / safe
    out["peoe_neutral_frac"] = out[PEOE_NEUTRAL].sum(axis=1) / safe
    out["peoe_positive_frac"] = out[PEOE_POSITIVE].sum(axis=1) / safe

    for col in MEASURED:
        if col in apis.columns:
            out[col] = pd.to_numeric(apis[col], errors="coerce")
    if {"tm_deg_c", "tg_deg_c"} <= set(out.columns):
        out["tm_div_tg_k"] = ((out["tm_deg_c"] + 273.15)
                              / (out["tg_deg_c"] + 273.15))

    if {"acidic_pka_min", "basic_pka_max"} <= set(out.columns):
        flags = [ionisation_flags(a, b) for a, b in
                 zip(out["acidic_pka_min"], out["basic_pka_max"])]
        for key in IONISATION:
            out[key] = [f[key] for f in flags]
    return out


def report_collinearity(X, name, threshold=0.9):
    """
    Name the near-duplicate pairs inside a feature group.

    Several descriptors in each group track molecular size, so a PC1 can be
    "how big is it" three times over. Reporting the pairs makes that visible
    instead of leaving it to be inferred from the loadings.
    """
    if X.shape[1] < 2 or len(X) < 4:
        return
    corr = X.corr(method="spearman").abs()
    pairs = [(corr.columns[i], corr.columns[j], corr.iat[i, j])
             for i in range(len(corr)) for j in range(i + 1, len(corr))
             if corr.iat[i, j] >= threshold]
    if pairs:
        print(f"  {name}: {len(pairs)} near-duplicate pair(s) "
              f"(|rho| >= {threshold})")
        for a, b, r in sorted(pairs, key=lambda t: -t[2])[:5]:
            print(f"      {a} ~ {b}  rho={r:.2f}")


def feature_frame(apis, which, verbose=False):
    """
    One feature set, with unavailable columns dropped and reported.

    Rows with any remaining gap are dropped rather than imputed: filling a
    missing Tg with a median would place that API in the middle of the map and
    make it look better covered than it is.
    """
    full = descriptor_frame(apis)
    wanted = FEATURE_SETS[which]
    have = [c for c in wanted if c in full.columns and full[c].notna().any()]
    missing = [c for c in wanted if c not in have]
    X = full[have]
    incomplete = X.index[X.isna().any(axis=1)]
    if verbose:
        if missing:
            print(f"  {which}: {len(missing)} feature(s) unavailable "
                  f"({', '.join(missing)})")
        if len(incomplete):
            print(f"  {which}: dropping {len(incomplete)} API(s) with gaps "
                  f"({', '.join(incomplete[:6])}"
                  f"{', ...' if len(incomplete) > 6 else ''})")
    X = X.dropna()
    if verbose:
        report_collinearity(X, which)
    return X


# --- drawing helpers -------------------------------------------------------
def _face(apis):
    """
    Tested APIs take an outcome colour; everything else is hollow.

    Returned as RGBA tuples rather than a mix of names and tuples, which
    older matplotlib will not accept as a single colour list.
    """
    to_rgba = plt.matplotlib.colors.to_rgba
    if "unstable_frac" not in apis:
        return pd.Series([to_rgba(STABLE if r == "tested" else "white")
                          for r in apis["role"]], index=apis.index)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "d", [STABLE, "#f0efec", UNSTABLE])
    return pd.Series(
        [tuple(cmap(v)) if r == "tested" and pd.notna(v) else to_rgba("white")
         for r, v in zip(apis["role"], apis.get("unstable_frac", apis["role"]))],
        index=apis.index)


def _scatter(ax, xy, apis, size=100):
    face = _face(apis)
    for role, (marker, edge, _) in ROLES.items():
        sel = (apis["role"] == role).values
        if not sel.any():
            continue
        ax.scatter(xy[sel, 0], xy[sel, 1], marker=marker,
                   c=list(face[sel]), edgecolor=edge,
                   linewidth=1.3 if role == "tested" else 1.5,
                   s=size * (1.15 if role == "holdout" else 1.0), zorder=3)


def _legend(ax, apis):
    handles = [
        plt.Line2D([], [], marker=m, ls="none", markersize=8,
                   markerfacecolor=(STABLE if r == "tested" else "white"),
                   markeredgecolor=e, label=lab)
        for r, (m, e, lab) in ROLES.items() if (apis["role"] == r).any()
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)


def _annotate(ax, xy, names, fontsize=7.5):
    """
    Greedy anti-overlap labelling: offset above by default, flip below or
    further out when the slot is taken. A white halo keeps a label readable
    where it still lands on a point.
    """
    xy = np.asarray(xy, dtype=float)
    if not len(xy):
        return
    span_x = (xy[:, 0].max() - xy[:, 0].min()) or 1.0
    span_y = (xy[:, 1].max() - xy[:, 1].min()) or 1.0
    taken = []
    order = np.argsort(-xy[:, 1])
    for i in order:
        x, y = xy[i]
        dy, va = 10, "bottom"
        for cand_dy, cand_va in ((10, "bottom"), (-10, "top"),
                                 (20, "bottom"), (-20, "top")):
            key = (x / span_x, (y + cand_dy * span_y / 380) / span_y)
            if not any(abs(key[0] - t[0]) < 0.085
                       and abs(key[1] - t[1]) < 0.032 for t in taken):
                taken.append(key)
                dy, va = cand_dy, cand_va
                break
        ax.annotate(list(names)[i], (x, y), xytext=(0, dy),
                    textcoords="offset points", fontsize=fontsize,
                    color=MUTED, ha="center", va=va,
                    path_effects=[pe.withStroke(linewidth=2.6,
                                                foreground="white")])


def _axis_drivers(pca, columns, n=3):
    """The features that define each of the first two axes, with direction."""
    lines = []
    for axis in (0, 1):
        load = pca.components_[axis]
        top = np.argsort(np.abs(load))[::-1][:n]
        parts = ["{} ({})".format(columns[k], "+" if load[k] >= 0 else MINUS)
                 for k in top]
        lines.append(f"PC{axis + 1} driven by: " + ", ".join(parts))
    return lines


# --- 1. similarity ---------------------------------------------------------
def fig_similarity(apis, sim, out):
    names = list(apis.index)
    order = leaves_list(linkage(squareform(1 - sim, checks=False), "average"))
    m = sim[np.ix_(order, order)]
    labels = [names[i] for i in order]
    roles = [apis["role"].iloc[i] for i in order]

    fig, ax = plt.subplots(figsize=(0.40 * len(names) + 3.6,
                                    0.40 * len(names) + 3))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(length=0)
    ax.grid(False)

    for i, role in enumerate(roles):
        marker, edge, _ = ROLES[role]
        ax.plot(-1.6, i, marker=marker, markersize=5, clip_on=False,
                markerfacecolor=STABLE if role == "tested" else "white",
                markeredgecolor=edge if role != "tested" else STABLE)

    off = m.copy()
    np.fill_diagonal(off, np.nan)
    ax.set_title("API structural similarity (ECFP4 Tanimoto)", pad=22)
    ax.text(0, 1.01, "clustered order · left marker: filled circle tested, "
            "hollow circle candidate, triangle held-out · median "
            f"nearest-neighbour {np.median(np.nanmax(off, axis=1)):.2f}",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    fig.savefig(out / "1_similarity.png")
    plt.close(fig)


# --- 2. PCA ----------------------------------------------------------------
def _pca_panel(ax, apis, X, label_all=True):
    """
    One PCA panel. Scaler and PCA are fitted WITHOUT the held-out set, which
    is then projected in, so the axes are not defined by the test structures.

    Loading arrows are deliberately omitted: on a panel this dense they sat
    on top of the points. The features that define each axis are listed above
    the plot instead.

    Returns the per-API PC1/PC2 scores and the driver lines.
    """
    sub = apis.loc[X.index]
    fit_mask = (sub["role"] != "holdout").values
    if fit_mask.sum() < 3:
        fit_mask = np.ones(len(sub), bool)

    scaler = StandardScaler().fit(X.values[fit_mask])
    Xs = scaler.transform(X.values)
    pca = PCA().fit(Xs[fit_mask])
    scores = pca.transform(Xs)
    var = pca.explained_variance_ratio_

    _scatter(ax, scores[:, :2], sub, size=90 if label_all else 70)
    if label_all:
        _annotate(ax, scores[:, :2], X.index)
    else:
        held = (sub["role"] == "holdout").values
        if held.any():
            _annotate(ax, scores[held, :2], X.index[held], fontsize=6.5)

    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.margins(0.16)
    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    frame = pd.DataFrame(scores[:, :2], index=X.index, columns=["PC1", "PC2"])
    return frame, _axis_drivers(pca, list(X.columns))


def _titled_notes(ax, title, lines, start=1.012, step=0.030):
    """
    Title and note lines stacked above the axes, bottom-up, so nothing is
    written over the data and no title padding has to be guessed.
    """
    for i, line in enumerate(lines):
        ax.text(0.0, start + i * step, line, transform=ax.transAxes,
                fontsize=7.5, color=MUTED, ha="left", va="bottom")
    ax.text(0.0, start + len(lines) * step + 0.012, title,
            transform=ax.transAxes, fontsize=11, fontweight="semibold",
            color=INK, ha="left", va="bottom")


def fig_pca_comparison(apis, out, sets=COMPARE_DEFAULT):
    """
    The same APIs under two (or three) feature sets, side by side.

    What matters is where they disagree. An API that looks well covered
    structurally but sits alone in the SME panel is a thermal-property gap
    that the fingerprints hide -- and for physical stability that is the gap
    that counts.
    """
    frames = {}
    print("feature sets:")
    for which in sets:
        X = feature_frame(apis, which, verbose=True)
        if X.shape[1] >= 3 and len(X) >= 5:
            frames[which] = X
        else:
            print(f"  {which}: skipped, too few usable features or APIs")
    if len(frames) < 2:
        print("  skipped 6_pca_comparison.png (need at least two feature sets)")
        return

    fig, axes = plt.subplots(1, len(frames), figsize=(6.6 * len(frames), 7.0))
    rows, feature_lines = [], []
    for ax, (which, X) in zip(np.atleast_1d(axes), frames.items()):
        scores, drivers = _pca_panel(ax, apis, X, label_all=False)
        name, note = SET_TITLES[which]
        _titled_notes(ax, f"{name}  ·  {X.shape[1]} features",
                      drivers[::-1] + [f"{note} · {len(X)} APIs"])
        feature_lines.append(f"{name}: " + ", ".join(X.columns))

        # Scale-free: how far each API sits from the training centroid.
        sub = apis.loc[X.index]
        tested = (sub["role"] == "tested").values
        if tested.any():
            z = (scores.values - scores.values[tested].mean(0))
            z /= (scores.values[tested].std(0) + 1e-9)
            d = np.linalg.norm(z, axis=1)
            rows.append(pd.Series(d, index=X.index, name=which))

    _legend(np.atleast_1d(axes)[0], apis)
    fig.suptitle(f"Coverage under {len(frames)} feature sets", x=0.0,
                 ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    note = ("\n".join(feature_lines) + "\n\n"
            "Only held-out APIs are labelled, to keep the panels readable; "
            "per-API distances are in pca_distances.csv. Structural uses "
            "connectivity alone; SME adds measured thermal and ADMET "
            "properties, so disagreement between the panels marks an API "
            "whose property profile its structure does not predict.")
    fig.text(0, -0.03, note, fontsize=7.5, color=MUTED, va="top")
    fig.savefig(out / "6_pca_comparison.png")
    plt.close(fig)

    if rows:
        dist = pd.concat(rows, axis=1)
        dist["role"] = apis["role"].reindex(dist.index)
        dist.round(3).to_csv(out / "pca_distances.csv")
        print("\ndistance from the training centroid "
              "(standard deviations, per feature set)")
        print(dist.groupby("role")[list(frames)].median().round(2)
              .to_string())
        spread = (dist[list(frames)].max(axis=1)
                  - dist[list(frames)].min(axis=1)).sort_values(ascending=False)
        top = spread.head(5)
        print("\nAPIs whose apparent coverage depends most on the feature set:")
        for name, gap in top.items():
            print(f"  {name:<24} range {gap:.2f} sd  ({dist.loc[name, 'role']})")


# --- 3. UMAP ---------------------------------------------------------------
def fig_umap(apis, fps, out, settings=(3, 5, 8)):
    import umap

    bits = np.array([list(fp) for fp in fps], dtype=np.uint8)
    fit_mask = (apis["role"] != "holdout").values
    n_fit = int(fit_mask.sum())
    settings = [s for s in settings if s < n_fit] or [max(2, n_fit - 1)]

    fig, axes = plt.subplots(1, len(settings), figsize=(4.8 * len(settings), 5.0))
    projected = True

    for ax, nn in zip(np.atleast_1d(axes), settings):
        reducer = umap.UMAP(n_neighbors=nn, min_dist=0.1, metric="jaccard",
                            random_state=0)
        try:
            base = reducer.fit_transform(bits[fit_mask])
            emb = np.empty((len(bits), 2))
            emb[fit_mask] = base
            if (~fit_mask).any():
                emb[~fit_mask] = reducer.transform(bits[~fit_mask])
        except Exception:
            # Some metric/version combinations cannot project new points.
            # Fall back to fitting on everything and say so.
            projected = False
            emb = umap.UMAP(n_neighbors=nn, min_dist=0.1, metric="jaccard",
                            random_state=0).fit_transform(bits)

        _scatter(ax, emb, apis, size=74)
        _annotate(ax, emb, apis.index, fontsize=6.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.margins(0.16)
        ax.set_title(f"n_neighbors = {nn}")

    _legend(np.atleast_1d(axes)[0], apis)
    fig.suptitle("ECFP4 structural diversity (UMAP)", x=0.0, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    note = ("held-out points projected into an embedding fitted without them"
            if projected else
            "WARNING: projection unavailable, embedding fitted on all points")
    fig.text(0, -0.02, f"{note}. Three settings shown because at this sample "
             "size the layout is unstable: trust a grouping only if it "
             "survives all three and shows up in the similarity matrix. "
             "Axis units and between-cluster distances are not meaningful.",
             fontsize=7.5, color=MUTED)
    fig.savefig(out / "3_umap.png")
    plt.close(fig)
    if not projected:
        print("  note: UMAP could not project held-out points; fitted on all")


# --- 4. design grid --------------------------------------------------------
def fig_design_grid(df, out):
    polymers = sorted(df["polymer_name"].unique())
    names = sorted(df["api"].unique())
    ypos = {a: i for i, a in enumerate(names)}
    has_target = "target" in df

    loads = df["api_loading_percent"]
    pad = max(1.0, 0.07 * (loads.max() - loads.min()))
    xlim = (loads.min() - pad, loads.max() + pad)

    fig, axes = plt.subplots(
        1, len(polymers), sharey=True,
        figsize=(2.9 * len(polymers) + 2, 0.4 * len(names) + 3.0))

    for ax, poly in zip(np.atleast_1d(axes), polymers):
        sub = df[df["polymer_name"] == poly]
        for api in names:
            if api not in set(sub["api"]):
                ax.axhspan(ypos[api] - 0.5, ypos[api] + 0.5, color=GRID,
                           alpha=0.55, lw=0, zorder=0)

        for api, g in sub.groupby("api"):
            y, x = ypos[api], g["api_loading_percent"].values
            if len(x) > 1:
                ax.plot([x.min(), x.max()], [y, y], color=MUTED, lw=0.8,
                        alpha=0.45, zorder=2)
            if has_target:
                t = g["target"].values
                # Dots for both classes: red = positive (target 1,
                # crystalline), blue = negative (target 0). Colour alone
                # carries the outcome here, which is acceptable only because
                # blue/red clears the colour-vision separation floor
                # (deutan dE 23.8); green/red would not.
                for val, marker, color in ((0, "o", STABLE), (1, "o", UNSTABLE)):
                    sel = t == val
                    if sel.any():
                        ax.scatter(x[sel], np.full(sel.sum(), y), marker=marker,
                                   color=color, s=64, edgecolor="white",
                                   linewidth=1.3, zorder=4)
            else:
                ax.scatter(x, np.full(len(x), y), s=58, color=STABLE,
                           edgecolor="white", linewidth=1.3, zorder=4)

        ax.set_xlim(*xlim)
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("API loading (wt %)")
        ax.set_title(f"{poly}  ·  {sub['api'].nunique()}/{len(names)} APIs")

    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(names)))
    ax0.set_yticklabels(names, fontsize=8)
    if has_target:
        handles = [plt.Line2D([], [], marker="o", ls="none", color=c, label=l)
                   for c, l in ((STABLE, "stable / amorphous (0)"),
                                (UNSTABLE, "unstable / crystalline (1)"))]
        handles.append(plt.Line2D([], [], marker="s", ls="none", color=GRID,
                                  label="not tested"))
        fig.legend(handles=handles, loc="lower left", ncol=3,
                   bbox_to_anchor=(0, 0.905), fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.87))
    # Title placed absolutely rather than via suptitle, which tight_layout
    # was dropping onto the legend row once the labels got longer.
    fig.text(0.0, 0.995, "Formulation design grid (training APIs only)",
             fontsize=12, fontweight="semibold", color=INK, ha="left",
             va="top")
    fig.savefig(out / "4_design_grid.png")
    plt.close(fig)


# --- 5. reach --------------------------------------------------------------
def fig_reach(apis, sim, out):
    """
    Similarity to the nearest *training* API, for candidates and for the
    held-out set. For the held-out panel this is the headline diagnostic:
    computed before the final test, it says whether that test is a fair
    extrapolation challenge or an impossible one.
    """
    names = np.array(apis.index)
    tested = (apis["role"] == "tested").values
    if not tested.any():
        return

    panels = [(r, (apis["role"] == r).values) for r in ("candidate", "holdout")]
    panels = [(r, m) for r, m in panels if m.any()]
    if not panels:
        print("  skipped 5_reach.png (no candidate or held-out APIs)")
        return

    heights = [int(m.sum()) for _, m in panels]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(9.5, 0.33 * sum(heights) + 2.2 * len(panels)),
        gridspec_kw={"height_ratios": heights, "hspace": 0.32}, squeeze=False)

    titles = {"candidate": "Candidate APIs vs the training set",
              "holdout": "Held-out test APIs vs the training set"}

    for ax, (role, mask) in zip(axes[:, 0], panels):
        sub = sim[np.ix_(mask, tested)]
        best = sub.max(axis=1)
        nearest = names[tested][sub.argmax(axis=1)]
        order = np.argsort(best)
        y = np.arange(order.size)

        color = STABLE if role == "candidate" else INK
        ax.barh(y, best[order], color=color, height=0.62, zorder=3)
        for i, k in enumerate(order):
            ax.annotate(f"{best[k]:.2f}  (closest: {nearest[k]})",
                        (best[k], i), xytext=(6, 0),
                        textcoords="offset points", va="center",
                        fontsize=7.6, color=MUTED)

        ax.set_yticks(y)
        ax.set_yticklabels(names[mask][order], fontsize=8)
        ax.set_ylim(-0.6, order.size - 0.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Tanimoto similarity to nearest training API")
        ax.grid(axis="y", visible=False)
        ax.set_title(titles[role], pad=22)
        ax.text(0, 1.01, f"n = {order.size} · median {np.median(best):.2f} · "
                f"max {best.max():.2f}", transform=ax.transAxes,
                fontsize=8, color=MUTED)

    fig.text(0, -0.015, "Low bars are the APIs the training set says least "
             "about. For the held-out panel, a low median means the final "
             "result should be reported as an extrapolation benchmark rather "
             "than a performance estimate.", fontsize=7.5, color=MUTED)
    fig.savefig(out / "5_reach.png")
    plt.close(fig)


STRIP_LABELS = {
    "tm_deg_c": "Tm (°C)", "tg_deg_c": "Tg (°C)",
    "tm_div_tg_k": "Tm / Tg (K)", "logp": "S+logP",
    "permeability": "S+Peff",
}


def fig_measured(apis, out):
    """
    Measured thermal and ADMET properties as strip plots, plus the
    ionisation class.

    A strip rather than a PCA: three or four correlated measured properties
    give a projection nothing to do, and each value is readable on its own.
    """
    full = descriptor_frame(apis)
    rows = [(group, col) for group, cols in STRIP_SETS.items()
            for col in cols if col in full.columns and full[col].notna().any()]
    have_ion = all(k in full.columns for k in IONISATION)
    if not rows:
        print("  skipped 7_measured.png (no measured properties found)")
        return

    n = len(rows) + (1 if have_ion else 0)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, 1.25 * n + 1.8),
                             squeeze=False)
    rng = np.random.default_rng(0)
    role_y = {"tested": 0.22, "candidate": 0.0, "holdout": -0.22}

    for ax, (group, col) in zip(axes[: len(rows), 0], rows):
        vals = full[col].astype(float)
        for role, (marker, edge, label) in ROLES.items():
            sel = (apis["role"] == role).values
            if not sel.any():
                continue
            y = role_y[role] + rng.uniform(-0.06, 0.06, int(sel.sum()))
            ax.scatter(vals[sel], y, marker=marker,
                       c=list(_face(apis)[sel]), edgecolor=edge,
                       linewidth=1.3, s=58, zorder=3)
        ax.set_yticks(list(role_y.values()))
        ax.set_yticklabels(["tested", "cand.", "held-out"], fontsize=7)
        ax.set_ylim(-0.42, 0.42)
        ax.grid(axis="y", visible=False)
        finite = vals.dropna()
        ax.text(1.005, 0.5,
                f"{STRIP_LABELS.get(col, col)}\nmedian {finite.median():,.1f}",
                transform=ax.transAxes, fontsize=7.6, va="center",
                color=MUTED)
        ax.text(0.0, 1.03, f"{group} · {len(finite)} of {len(apis)} APIs",
                transform=ax.transAxes, fontsize=7, color=MUTED)

    if have_ion:
        ax = axes[len(rows), 0]
        order = list(ROLES)
        width = 0.8 / len(IONISATION)
        for k, flag in enumerate(IONISATION):
            counts = [int(full.loc[(apis["role"] == r).values, flag].sum())
                      for r in order]
            xs = np.arange(len(order)) + (k - 1.5) * width
            ax.bar(xs, counts, width=width * 0.9, zorder=3,
                   color=plt.matplotlib.colors.to_rgba(STABLE,
                                                       0.45 + 0.18 * k),
                   label=flag.replace("_", " "))
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([ROLES[r][2] for r in order], fontsize=7.6)
        ax.set_ylabel("APIs", fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=4, fontsize=7.5, loc="upper right")
        ax.text(0.0, 1.03, "admet_measured · ionisation class from pKa "
                "(thresholds 6.5 / 2)", transform=ax.transAxes, fontsize=7,
                color=MUTED)

    fig.suptitle("Measured properties by role", x=0.0, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    fig.text(0, -0.03, "Marker shape and row both carry the role; fill "
             "colour carries the outcome for tested APIs. A property whose "
             "held-out points fall outside the tested spread is an "
             "extrapolation the structural maps will not show.",
             fontsize=7.5, color=MUTED, va="top")
    fig.savefig(out / "7_measured.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="formulation table")
    p.add_argument("--apis", default=None, help="candidate API list")
    p.add_argument("--holdout", default=None, help="held-out test API list")
    p.add_argument("--outdir", default="coverage_out")
    p.add_argument("--sheet", default=0)
    p.add_argument("--header", type=int, default=None,
                   help="0-based header row, forced for every file. Omit to "
                        "detect it per file (title rows are handled).")
    p.add_argument("--no-compare", dest="compare", action="store_false",
                   help="skip the feature-set comparison figure")
    a = p.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sheet = int(a.sheet) if str(a.sheet).isdigit() else a.sheet

    df = load_formulations(a.input, sheet, a.header)
    candidates = load_api_list(a.apis, sheet, a.header) if a.apis else None
    holdout = load_api_list(a.holdout, sheet, a.header) if a.holdout else None

    # Check the raw inputs before de-duplication collapses the evidence.
    report_overlap(df, candidates, holdout)

    apis = build_api_table(df, candidates, holdout)
    fps = fingerprints(apis["smiles"])
    sim = tanimoto(fps)

    counts = apis["role"].value_counts()
    n_poly = df["polymer_name"].nunique()
    n_tested = int(counts.get("tested", 0))
    cells = df.groupby(["api", "polymer_name"]).ngroups
    print(f"\n{len(apis)} APIs total · "
          + " · ".join(f"{int(counts.get(r, 0))} {r}" for r in ROLES))
    print(f"{n_poly} polymers · {len(df)} formulations · design grid "
          f"{cells}/{n_tested * n_poly} cells filled\n")

    fig_similarity(apis, sim, out)
    fig_umap(apis, fps, out)
    fig_design_grid(df, out)
    fig_reach(apis, sim, out)
    if a.compare:
        fig_pca_comparison(apis, out, COMPARE_DEFAULT)
    fig_measured(apis, out)
    print(f"\nwrote figures to {out}")


if __name__ == "__main__":
    main()
