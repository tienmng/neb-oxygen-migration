#!/usr/bin/env python3
"""
Coverage maps, simple version.

Three inputs, because the question is whether the APIs already tested cover
the APIs you might test and the APIs you will finally be judged on:

  --input    Formulation_example_input_file.xlsx   formulations run so far
  --apis     API_information_for_selection.xlsx    the candidate list
  --holdout  FPN wet SDD stability data.xlsx       the final test set

Five figures:
  1_similarity.png   ECFP4 Tanimoto matrix over every API, clustered
  2_pca.png          PCA of physicochemical descriptors
  3_umap.png         UMAP on ECFP4 at three n_neighbors settings
  4_design_grid.png  API x polymer, loading on x, outcome by marker
  5_reach.png        similarity to the nearest *tested* API, two panels

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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

DESCRIPTORS = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
               "NumRotatableBonds", "NumAromaticRings", "FractionCSP3",
               "HeavyAtomCount", "BertzCT", "Chi0", "Kappa2"]

# Measured / ADMET columns read from the input files. Spellings differ between
# the formulation file and the selection file, so each is matched on a
# normalised name (lower case, alphanumerics only): "S+logP" -> "slogp".
MEASURED = {
    "tm_deg_c": ["apitmdegreesc", "apitmdegc", "tmdegreesc", "tmdegc",
                 "meltingpoint", "tm"],
    "tg_deg_c": ["apitgdegreesc", "apitgdegc", "tgdegreesc", "tgdegc",
                 "glasstransition", "tg"],
    "logp": ["splogp", "slogp", "apilogp", "logp"],
    "permeability": ["speff", "sppeff", "apipermeability", "permeability",
                     "peff"],
}

# Three feature sets, deliberately differing in how much they know.
#
# structural  generic RDKit descriptors. No labels anywhere in its
#             construction, so this is the coverage map that can be defended.
# sme         the api_atts list from wet_stability_model_dev_v0.py, minus the
#             three ionisation flags, which are derived in
#             molecule_instance.compute_acid_base_zwitterion from pKa columns
#             whose thresholds are not reproduced here. Domain-chosen but
#             still label-free.
# boruta      the API-level members of hand_currated_boruta_from_drop_one_atts.
#             Selected against the training labels, so a map drawn in these
#             axes is label-informed: read it as "where the test set sits in
#             the space the model keys on", never as evidence the test is fair.
#
# The four formulation-level members of the Boruta list (form_api_loading,
# kinetic_tg_mix_theoretical_c, thermo_excess_gibbs_energy,
# thermo_ln_gamma_api) are undefined for an API with no formulation, so they
# cannot appear here. Reaching them needs a fixed reference formulation and
# the TeNNet-SAC engine.
FEATURE_SETS = {
    "structural": ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
                   "NumRotatableBonds", "NumAromaticRings", "FractionCSP3",
                   "HeavyAtomCount", "BertzCT"],
    "sme": ["tm_deg_c", "tg_deg_c", "tm_div_tg_k", "logp", "permeability",
            "MolWt", "NumRotatableBonds", "NumHAcceptors", "NumHDonors",
            "Chi0", "Kappa2"],
    "boruta": ["Chi0", "Kappa2", "MolWt", "NumHAcceptors", "tm_div_tg_k",
               "permeability"],
}

# Boruta-6 is a strict subset of the SME-11, so it adds no information --
# only a reweighting. It is off by default and available via --include-boruta
# for the separate, post-hoc question of where the test set sits in the space
# the model actually keys on.
COMPARE_DEFAULT = ("structural", "sme")

SET_TITLES = {
    "structural": ("Structural", "generic RDKit descriptors · label-free"),
    "sme": ("SME physicochemical", "api_atts minus ionisation flags · "
            "label-free"),
    "boruta": ("Boruta-selected", "API-level subset · LABEL-INFORMED"),
}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                     "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


# --- loading ---------------------------------------------------------------
def _read(path, sheet=0):
    path = Path(path)
    return (pd.read_csv(path) if path.suffix.lower() == ".csv"
            else pd.read_excel(path, sheet_name=sheet))


def _find_column(df, options, contains=None):
    """Locate a column by exact name or substring, ignoring case."""
    lower = {c.lower().strip(): c for c in df.columns}
    for opt in options:
        if opt in lower:
            return lower[opt]
    if contains:
        for low, orig in lower.items():
            if contains in low:
                return orig
    raise KeyError(f"No column matching {options} in {list(df.columns)}")


def _normalise(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def extract_measured(df):
    """
    Pull Tm, Tg, logP and permeability out of whatever they are called here.

    Returns a frame with only the columns that were found, so a missing
    property degrades the affected feature set rather than crashing.
    """
    lookup = {_normalise(c): c for c in df.columns}
    found = {}
    for target, options in MEASURED.items():
        for opt in options:
            if opt in lookup:
                found[target] = pd.to_numeric(df[lookup[opt]], errors="coerce")
                break
    return pd.DataFrame(found, index=df.index)


def load_formulations(path, sheet=0):
    """The formulation table: one row per API x polymer x loading."""
    raw = _read(path, sheet)
    df = raw.rename(columns={k: v for k, v in RENAME.items()})
    df["smiles"] = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in df["smiles"]]
    for col, values in extract_measured(raw).items():
        df[col] = values.values
    return df


def load_api_list(path, sheet=0):
    """
    A table of APIs: name plus SMILES. Column names vary between the
    selection file and the held-out file, so both are located by pattern.
    Only the structure is read -- never an outcome column.
    """
    df = _read(path, sheet)
    name_col = _find_column(df, ["api", "api_name", "name", "compound",
                                 "molecule"], contains="api")
    smi_col = _find_column(df, ["smiles"], contains="smiles")
    out = pd.DataFrame({
        "api": df[name_col].astype(str).str.strip(),
        "smiles": [Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(str(s)))
                   else None for s in df[smi_col]],
    })
    for col, values in extract_measured(df).items():
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
    """RDKit descriptors plus whatever measured properties came from the files."""
    rows = [{d: Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s)).get(d)
             for d in DESCRIPTORS} for s in apis["smiles"]]
    out = pd.DataFrame(rows, index=apis.index)
    for col in MEASURED:
        if col in apis.columns:
            out[col] = pd.to_numeric(apis[col], errors="coerce")
    if {"tm_deg_c", "tg_deg_c"} <= set(out.columns):
        out["tm_div_tg_k"] = ((out["tm_deg_c"] + 273.15)
                              / (out["tg_deg_c"] + 273.15))
    return out


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
    return X.dropna()


# --- drawing helpers -------------------------------------------------------
def _face(apis):
    """Tested APIs take an outcome colour; everything else is hollow."""
    if "unstable_frac" not in apis:
        return pd.Series(["#2a78d6" if r == "tested" else "white"
                          for r in apis["role"]], index=apis.index)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "d", [STABLE, "#f0efec", UNSTABLE])
    return pd.Series(
        [cmap(v) if r == "tested" and pd.notna(v) else "white"
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
    for name, (x, y) in zip(names, xy):
        ax.annotate(name, (x, y), xytext=(0, 9), textcoords="offset points",
                    fontsize=fontsize, color=MUTED, ha="center")


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
    ax.set_xticks(range(len(names)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), labels, fontsize=7)
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
def _pca_panel(ax, apis, X, label_all=True, arrow_labels=True):
    """
    One PCA panel. Scaler and PCA are fitted WITHOUT the held-out set, which
    is then projected in, so the axes are not defined by the test structures.

    Returns the per-API PC1/PC2 scores.
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

    if arrow_labels:
        scale = np.abs(scores[:, :2]).max() * 0.72
        for k, col in enumerate(X.columns):
            dx, dy = pca.components_[:2, k] * scale
            ax.annotate("", (dx, dy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.1))
            ax.annotate(col, (dx, dy), fontsize=7, color=ACCENT,
                        fontweight="semibold",
                        xytext=(6 * np.sign(dx or 1), 6 * np.sign(dy or 1)),
                        textcoords="offset points",
                        ha="left" if dx >= 0 else "right",
                        va="bottom" if dy >= 0 else "top")

    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.margins(0.16)
    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    return pd.DataFrame(scores[:, :2], index=X.index, columns=["PC1", "PC2"])


def fig_pca(apis, out, which="structural"):
    X = feature_frame(apis, which, verbose=True)
    fig, ax = plt.subplots(figsize=(10, 7.8))
    _pca_panel(ax, apis, X)
    name, note = SET_TITLES[which]
    ax.set_title(f"{name} property space (PCA)", pad=22)
    ax.text(0, 1.01, f"{X.shape[1]} scaled features · {note} · "
            f"{len(X)} of {len(apis)} APIs · axes fitted without the "
            "held-out set", transform=ax.transAxes, fontsize=8, color=MUTED)
    _legend(ax, apis.loc[X.index])
    fig.savefig(out / "2_pca.png")
    plt.close(fig)


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

    fig, axes = plt.subplots(1, len(frames), figsize=(6.2 * len(frames), 6.6))
    rows = []
    for ax, (which, X) in zip(np.atleast_1d(axes), frames.items()):
        scores = _pca_panel(ax, apis, X, label_all=False, arrow_labels=True)
        name, note = SET_TITLES[which]
        ax.set_title(f"{name}  ·  {X.shape[1]} features", pad=22)
        ax.text(0, 1.01, f"{note} · {len(X)} APIs", transform=ax.transAxes,
                fontsize=7.5, color=MUTED)

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
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    note = ("Only held-out APIs are labelled, to keep the panels readable; "
            "per-API distances are in pca_distances.csv. Structural uses "
            "connectivity alone; SME adds measured thermal and ADMET "
            "properties, so disagreement between the panels marks an API "
            "whose property profile its structure does not predict.")
    if "boruta" in frames:
        note += (" The Boruta panel is label-informed -- its axes were chosen "
                 "using training outcomes, so it cannot be used to argue the "
                 "held-out test is fair.")
    fig.text(0, -0.02, note, fontsize=7.5, color=MUTED)
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
                for val, marker, color in ((0, "o", STABLE), (1, "X", UNSTABLE)):
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

    np.atleast_1d(axes)[0].set_yticks(range(len(names)), names, fontsize=8)
    if has_target:
        handles = [plt.Line2D([], [], marker=m, ls="none", color=c, label=l)
                   for m, c, l in (("o", STABLE, "stable"),
                                   ("X", UNSTABLE, "unstable"))]
        handles.append(plt.Line2D([], [], marker="s", ls="none", color=GRID,
                                  label="not tested"))
        fig.legend(handles=handles, loc="lower left", ncol=3,
                   bbox_to_anchor=(0, 0.93), fontsize=8)

    fig.suptitle("Formulation design grid (training APIs only)", x=0.0,
                 ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
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

        ax.set_yticks(y, names[mask][order], fontsize=8)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="formulation table")
    p.add_argument("--apis", default=None, help="candidate API list")
    p.add_argument("--holdout", default=None, help="held-out test API list")
    p.add_argument("--outdir", default="coverage_out")
    p.add_argument("--sheet", default=0)
    p.add_argument("--features", default="structural",
                   choices=list(FEATURE_SETS),
                   help="feature set for the main PCA (default: structural)")
    p.add_argument("--no-compare", dest="compare", action="store_false",
                   help="skip the feature-set comparison figure")
    p.add_argument("--include-boruta", action="store_true",
                   help="add the label-informed Boruta panel to the "
                        "comparison (a strict subset of the SME set)")
    a = p.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sheet = int(a.sheet) if str(a.sheet).isdigit() else a.sheet

    df = load_formulations(a.input, sheet)
    candidates = load_api_list(a.apis, sheet) if a.apis else None
    holdout = load_api_list(a.holdout, sheet) if a.holdout else None

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
    fig_pca(apis, out, which=a.features)
    fig_umap(apis, fps, out)
    fig_design_grid(df, out)
    fig_reach(apis, sim, out)
    if a.compare:
        sets = COMPARE_DEFAULT + (("boruta",) if a.include_boruta else ())
        fig_pca_comparison(apis, out, sets)
    print(f"\nwrote figures to {out}")


if __name__ == "__main__":
    main()
