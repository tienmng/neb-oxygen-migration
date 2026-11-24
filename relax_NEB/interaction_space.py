#!/usr/bin/env python3
"""
Tier 4 -- interaction / thermodynamic space.

Tiers 1-3 map molecules. This maps *formulations*: where an API-polymer-loading
combination sits in the space of quantities that actually drive physical
stability -- activity coefficient, surface-charge complementarity, and the
Fox mixture Tg relative to storage temperature.

Why it matters for new chemistry: a polymer entering the model as an identity
is out of domain the moment it is new. Entering through interaction
descriptors, a chemically novel polymer can still land inside the training
range. That is the whole argument for this tier.

Outputs
-------
  interaction_features.csv      one row per API x polymer x loading
  t4_interaction_space.png      ln(gamma) vs storage margin, coloured by outcome
  t4_complementarity.png        H-bond donor/acceptor matching vs ln(gamma)
  t4_sigma_profiles.png         the profiles themselves, for sanity checking
  t4_polymer_properties.png     z-scored polymer property heatmap

Running it
----------
Needs torch + tennetsac, so it runs in your modelling environment, not in a
bare rdkit install. Profiles are cached to sigma_cache.npz, keyed by canonical
SMILES, so reruns are cheap.

    python interaction_space.py --validate          # check sign convention FIRST
    python interaction_space.py \
        --input   "../data/raw/Formulation_example_input_file.xlsx" \
        --apis    "../data/raw/API_information_for_selection.xlsx" \
        --holdout "../data/raw/FPN wet SDD stability data.xlsx" \
        --model-dir ../model

--engine mock swaps the sigma-profile generator for a Gasteiger-charge
surrogate. That exists to exercise the code paths without torch; its numbers
are NOT physically meaningful and every figure it produces is watermarked.

Conventions matched to your code
--------------------------------
Mixture Tg uses the FOX equation on weight fractions, and the polymer Tg is
taken at relative humidity from Tg_vs_RH_slope/intercept, exactly as
formulation_instance.compute_kinetics does. Activity coefficients come from
your activity_coeff_computation_engine, including its HPMCAS-M averaging, so
the numbers here line up with thermo_* in the model's feature table.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage_simple import (  # noqa: E402
    STABLE, UNSTABLE, MUTED, GRID, INK, ROLES,
    load_formulations, load_api_list, build_api_table, canonical_parent,
    _face, _annotate,
)

SIGMA_HB = 0.0084          # e/A^2, standard COSMO-SAC H-bond cutoff
AEFF = 5.8447              # A^2, segment area
DEFAULT_RH = 0.25          # matches data_processing.rh_for_poly_tg_calc
DEFAULT_T_STORAGE_C = 25.0

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                     "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


# ===========================================================================
# Sigma profiles
# ===========================================================================
def sigma_grid(n_bins):
    """Bin centres for a symmetric COSMO-SAC sigma profile."""
    half = {41: 0.020, 51: 0.025, 61: 0.030}.get(n_bins)
    if half is None:
        raise ValueError(f"Unrecognised sigma profile length: {n_bins}")
    return np.linspace(-half, half, n_bins)


def as_profile(prf):
    """Torch tensor or array -> 1-D float array of A^2 per bin."""
    if hasattr(prf, "detach"):
        prf = prf.squeeze().detach().cpu().numpy()
    return np.asarray(prf, dtype=float).ravel()


def profile_descriptors(prf, threshold=SIGMA_HB):
    """
    Surface-area-weighted descriptors from one sigma profile.

    Positive sigma is electron-rich surface (H-bond acceptor) under the usual
    COSMO-SAC convention. VERIFY with --validate before trusting the labels:
    if acetone comes out donor-dominant the convention is inverted and the
    acceptor/donor names must be swapped.
    """
    p = as_profile(prf)
    s = sigma_grid(p.size)
    total = p.sum()
    if total <= 0:
        return {k: np.nan for k in
                ("acceptor_area", "donor_area", "nonpolar_area",
                 "acceptor_frac", "donor_frac", "nonpolar_frac",
                 "polar_frac", "hb_asymmetry", "sigma_mean", "sigma_var",
                 "total_area")}
    acc = p[s >= threshold].sum()
    don = p[s <= -threshold].sum()
    npl = p[np.abs(s) < threshold].sum()
    mean = float((p * s).sum() / total)
    return {
        "acceptor_area": float(acc), "donor_area": float(don),
        "nonpolar_area": float(npl),
        "acceptor_frac": float(acc / total), "donor_frac": float(don / total),
        "nonpolar_frac": float(npl / total),
        "polar_frac": float((acc + don) / total),
        "hb_asymmetry": float((acc - don) / total),
        "sigma_mean": mean,
        "sigma_var": float((p * (s - mean) ** 2).sum() / total),
        "total_area": float(total),
    }


def pair_descriptors(prf_api, prf_poly, threshold=SIGMA_HB):
    """
    API-polymer H-bond complementarity.

    Defined for any polymer whose SMILES can be written, which is the property
    the new-polymer question needs: these coordinates exist for a chemistry
    the model has never seen.
    """
    a = profile_descriptors(prf_api, threshold)
    b = profile_descriptors(prf_poly, threshold)
    pa, pb = as_profile(prf_api), as_profile(prf_poly)
    if pa.size != pb.size:
        raise ValueError("Profiles are on different grids.")
    # Overlap integral: API surface at sigma against polymer surface at -sigma.
    # The grid is symmetric, so reversing maps sigma -> -sigma.
    overlap = float((pa * pb[::-1]).sum() / (pa.sum() * pb.sum()))
    return {
        "hb_match": float(a["donor_frac"] * b["acceptor_frac"]
                          + a["acceptor_frac"] * b["donor_frac"]),
        "profile_overlap": overlap,
        "donor_ratio": float(a["donor_frac"] / max(b["donor_frac"], 1e-9)),
        "acceptor_ratio": float(a["acceptor_frac"] / max(b["acceptor_frac"], 1e-9)),
    }


# ===========================================================================
# Engines
# ===========================================================================
class MockEngine:
    """
    Gasteiger-charge surrogate for a sigma profile. TEST SCAFFOLD ONLY.

    Spreads each atom's partial charge over an approximate surface area into
    the COSMO-SAC grid. The shape is plausible enough to exercise every code
    path, and the sign is deliberately set so positive sigma is electron-rich,
    matching the COSMO convention -- but these are not COSMO profiles and no
    number from them means anything physically.
    """
    is_mock = True

    def __init__(self, polymer_smiles, n_bins=51):
        self.polymer_smiles = polymer_smiles
        self.n_bins = n_bins

    def sigma_profile(self, smiles):
        from rdkit.Chem import rdPartialCharges
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        rdPartialCharges.ComputeGasteigerCharges(mol)
        grid = sigma_grid(self.n_bins)
        prf = np.zeros(self.n_bins)
        for atom in mol.GetAtoms():
            q = atom.GetDoubleProp("_GasteigerCharge")
            if not np.isfinite(q):
                continue
            # Electron-rich atom (negative q) -> positive screening density.
            sigma = np.clip(-q * 0.05, grid[0], grid[-1])
            area = 8.0 + 4.0 * atom.GetDegree()
            prf += area * np.exp(-0.5 * ((grid - sigma) / 0.0035) ** 2)
        total_area = float(Descriptors.LabuteASA(mol))
        prf = prf / prf.sum() * total_area
        volume = total_area * 0.6
        return prf, total_area, volume

    def ln_gamma(self, api_smiles, polymer_name, x_api):
        """Crude Flory-Huggins-like stand-in, purely to populate the axes."""
        pa, area_a, _ = self.sigma_profile(api_smiles)
        pb, area_b, _ = self.sigma_profile(self._poly_smiles(polymer_name))
        d = pair_descriptors(pa, pb)
        chi = 2.5 * (1.0 - d["profile_overlap"] * 40.0)
        x_poly = 1.0 - x_api
        return chi * x_poly ** 2, chi * x_api ** 2

    def _poly_smiles(self, name):
        value = self.polymer_smiles[name]
        return value["1"] if isinstance(value, dict) else value


class TennetEngine:
    """Thin wrapper over the project's activity_coeff_computation_engine."""
    is_mock = False

    def __init__(self, model_dir, polymer_smiles, temperature_c=25.0):
        sys.path.insert(0, str(Path(model_dir).resolve()))
        from ads_molecule_classes import activity_coeff_computation_engine
        self.engine = activity_coeff_computation_engine(
            polymer_smiles, temperature_of_interest_deg_c=temperature_c)
        self.engine.load_tennet_sac_models()
        self.polymer_smiles = polymer_smiles

    def sigma_profile(self, smiles):
        prf, area, volume = self.engine.sigma_profile_wrapper(smiles)
        return as_profile(prf), float(area), float(volume)

    def ln_gamma(self, api_smiles, polymer_name, x_api):
        return self.engine.calculate_activity_coeffs({
            "mole_fractions_list": [x_api],
            "polymer_name": polymer_name,
            "polymer_smiles": self.polymer_smiles[polymer_name],
            "canonical_api_smiles": api_smiles,
        })


class ProfileCache:
    """Disk-backed sigma-profile cache keyed by canonical SMILES."""

    def __init__(self, engine, path="sigma_cache.npz"):
        self.engine = engine
        self.path = Path(path)
        self.store = {}
        if self.path.exists() and not getattr(engine, "is_mock", False):
            with np.load(self.path, allow_pickle=True) as z:
                self.store = {k: z[k] for k in z.files}
            print(f"loaded {len(self.store)} cached profiles from {self.path}")

    def get(self, smiles):
        if smiles not in self.store:
            prf, area, volume = self.engine.sigma_profile(smiles)
            self.store[smiles] = np.concatenate([[area, volume], prf])
        row = self.store[smiles]
        return row[2:], float(row[0]), float(row[1])

    def save(self):
        if not getattr(self.engine, "is_mock", False) and self.store:
            np.savez_compressed(self.path, **self.store)
            print(f"cached {len(self.store)} profiles to {self.path}")


# ===========================================================================
# Thermodynamics / kinetics
# ===========================================================================
def polymer_tg_c(props, rh=DEFAULT_RH):
    """Polymer Tg at relative humidity, as compute_kinetics does it."""
    return props["Tg_vs_RH_slope"] * rh + props["Tg_vs_RH_intercept"]


def fox_tg_mix_k(api_tg_c, poly_tg_c_value, w_api):
    """Fox equation on weight fractions: 1/Tg_mix = w1/Tg1 + w2/Tg2."""
    api_k = api_tg_c + 273.15
    poly_k = poly_tg_c_value + 273.15
    if api_k <= 0 or poly_k <= 0:
        return np.nan
    return 1.0 / (w_api / api_k + (1.0 - w_api) / poly_k)


def mole_fraction(w_percent, m_api, m_poly):
    """Weight percent -> API mole fraction for a binary mixture."""
    w = w_percent / 100.0
    n_api = w / m_api
    n_poly = (1.0 - w) / m_poly
    return n_api / (n_api + n_poly) if (n_api + n_poly) else np.nan


# ===========================================================================
# Table
# ===========================================================================
def build_interaction_table(apis, formulations, polymer_props, polymer_smiles,
                            cache, engine, reference_loadings=(10, 25, 40),
                            rh=DEFAULT_RH, t_storage_c=DEFAULT_T_STORAGE_C):
    """
    One row per API x polymer x loading.

    Tested APIs contribute their real formulations with outcomes. Untested
    APIs are placed at reference loadings in every polymer, so every candidate
    gets coordinates in the same space -- that is what makes this a coverage
    map rather than a description of the training set.
    """
    tested = formulations.copy()
    tested["source"] = "observed"
    rows = []

    for api, row in apis.iterrows():
        smi = row["smiles"]
        prf_api, area_api, _ = cache.get(smi)
        d_api = profile_descriptors(prf_api)
        mw_api = Descriptors.MolWt(Chem.MolFromSmiles(smi))
        tg_api = row.get("tg_deg_c", np.nan)

        if row["role"] == "tested":
            sub = tested[tested["api"] == api]
            plan = [(r["polymer_name"], r["api_loading_percent"],
                     r.get("target", np.nan), "observed")
                    for _, r in sub.iterrows()]
        else:
            plan = [(p, L, np.nan, "reference")
                    for p in polymer_props for L in reference_loadings]

        for poly, loading, target, source in plan:
            if poly not in polymer_props:
                continue
            props = polymer_props[poly]
            prf_poly, area_poly, _ = cache.get(
                _first_smiles(polymer_smiles[poly]))
            d_poly = profile_descriptors(prf_poly)
            d_pair = pair_descriptors(prf_api, prf_poly)

            x_api = mole_fraction(loading, mw_api, props["mw"])
            try:
                ln_g_api, ln_g_poly = engine.ln_gamma(smi, poly, x_api)
            except Exception as exc:                     # noqa: BLE001
                print(f"  ln_gamma failed for {api} / {poly} @ {loading}%: {exc}")
                ln_g_api = ln_g_poly = np.nan

            tg_poly_c = polymer_tg_c(props, rh)
            tg_mix_k = fox_tg_mix_k(tg_api, tg_poly_c, loading / 100.0)

            rec = {
                "api": api, "polymer": poly, "api_loading_percent": loading,
                "role": row["role"], "source": source, "target": target,
                "x_api": x_api,
                "ln_gamma_api": ln_g_api, "ln_gamma_poly": ln_g_poly,
                "delta_ln_gamma": ln_g_api - ln_g_poly,
                "tg_mix_k": tg_mix_k,
                "tg_mix_c": tg_mix_k - 273.15,
                "storage_delta_t": tg_mix_k - (t_storage_c + 273.15),
            }
            rec.update({f"api_sp_{k}": v for k, v in d_api.items()})
            rec.update({f"poly_sp_{k}": v for k, v in d_poly.items()})
            rec.update({f"pair_{k}": v for k, v in d_pair.items()})
            rows.append(rec)

    return pd.DataFrame(rows)


def _first_smiles(value):
    """HPMCAS-M maps to several variants; take the first for descriptors."""
    return value["1"] if isinstance(value, dict) else value


# ===========================================================================
# Figures
# ===========================================================================
def _watermark(fig, engine):
    if getattr(engine, "is_mock", False):
        fig.text(0.5, 0.5, "MOCK ENGINE\nnot physically meaningful",
                 fontsize=34, color="#d03b3b", alpha=0.16, ha="center",
                 va="center", rotation=28, fontweight="bold", zorder=10)


def fig_interaction_space(table, out, engine):
    """
    The tier-4 map: interaction strength against kinetic margin.

    If the classes separate here, the model has a learnable physical signal.
    If they overlap, that is where the data runs out -- and it is a more
    honest statement of the limit than any fingerprint map gives.
    """
    fig, ax = plt.subplots(figsize=(10, 7.4))
    obs = table[table["source"] == "observed"].dropna(
        subset=["ln_gamma_api", "storage_delta_t"])
    ref = table[table["source"] == "reference"].dropna(
        subset=["ln_gamma_api", "storage_delta_t"])

    for val, color, label in ((0, STABLE, "stable / amorphous (0)"),
                              (1, UNSTABLE, "unstable / crystalline (1)")):
        sel = obs[obs["target"] == val]
        if len(sel):
            ax.scatter(sel["ln_gamma_api"], sel["storage_delta_t"],
                       c=color, s=68, edgecolor="white", linewidth=1.2,
                       zorder=4, label=label)
    for role, marker in (("candidate", "o"), ("holdout", "^")):
        sel = ref[ref["role"] == role]
        if len(sel):
            ax.scatter(sel["ln_gamma_api"], sel["storage_delta_t"],
                       marker=marker, facecolor="none",
                       edgecolor=MUTED if role == "candidate" else INK,
                       s=52, linewidth=1.2, zorder=3,
                       label=f"{role} (reference loadings)")

    ax.axhline(0, color=UNSTABLE, linewidth=1.0, linestyle="--", alpha=0.6)
    ax.annotate("Tg(mix) = storage T", (0.995, 0), xycoords=("axes fraction",
                "data"), ha="right", va="bottom", fontsize=7.5,
                color=UNSTABLE)
    ax.set_xlabel("ln γ (API) — interaction strength")
    ax.set_ylabel("Tg(mix) − storage T  (K) — kinetic margin")
    ax.set_title("Interaction space", pad=22)
    ax.text(0, 1.01, "filled = observed outcome · hollow = untested, placed "
            "at reference loadings · Fox Tg(mix), polymer Tg at "
            f"{DEFAULT_RH:.0%} RH", transform=ax.transAxes, fontsize=8,
            color=MUTED)
    ax.legend(loc="best", fontsize=8)
    _watermark(fig, engine)
    fig.savefig(out / "t4_interaction_space.png")
    plt.close(fig)


def fig_complementarity(table, out, engine):
    """H-bond complementarity against interaction strength."""
    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    obs = table[table["source"] == "observed"].dropna(
        subset=["pair_hb_match", "ln_gamma_api"])
    ref = table[table["source"] == "reference"].dropna(
        subset=["pair_hb_match", "ln_gamma_api"])

    for val, color, label in ((0, STABLE, "stable"), (1, UNSTABLE, "unstable")):
        sel = obs[obs["target"] == val]
        if len(sel):
            ax.scatter(sel["pair_hb_match"], sel["ln_gamma_api"], c=color,
                       s=68, edgecolor="white", linewidth=1.2, zorder=4,
                       label=label)
    if len(ref):
        ax.scatter(ref["pair_hb_match"], ref["ln_gamma_api"], marker="^",
                   facecolor="none", edgecolor=INK, s=46, linewidth=1.1,
                   zorder=3, label="untested")

    ax.set_xlabel("H-bond match  (API donor × polymer acceptor + reverse)")
    ax.set_ylabel("ln γ (API)")
    ax.set_title("Complementarity vs interaction strength", pad=22)
    ax.text(0, 1.01, "both axes are defined for any polymer whose SMILES can "
            "be written, including chemistries the model has never seen",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.legend(loc="best", fontsize=8)
    _watermark(fig, engine)
    fig.savefig(out / "t4_complementarity.png")
    plt.close(fig)


def fig_sigma_profiles(cache, apis, polymer_smiles, out, engine, n_apis=6):
    """The profiles themselves -- the first place a convention error shows."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    grid = None

    for name, smi in list(polymer_smiles.items()):
        prf, _, _ = cache.get(_first_smiles(smi))
        grid = sigma_grid(prf.size)
        axes[0].plot(grid, prf, linewidth=1.8, label=name)
    axes[0].set_title("Polymers", pad=14)

    shown = apis.head(n_apis)
    for name, row in shown.iterrows():
        prf, _, _ = cache.get(row["smiles"])
        grid = sigma_grid(prf.size)
        axes[1].plot(grid, prf, linewidth=1.5, label=name)
    axes[1].set_title(f"APIs (first {len(shown)})", pad=14)

    for ax in axes:
        if grid is not None:
            ax.axvspan(-SIGMA_HB, SIGMA_HB, color=GRID, alpha=0.6, lw=0,
                       zorder=0)
            ax.axvline(SIGMA_HB, color=MUTED, lw=0.8, alpha=0.6)
            ax.axvline(-SIGMA_HB, color=MUTED, lw=0.8, alpha=0.6)
        ax.set_xlabel("σ  (e/Å²)")
        ax.set_ylabel("area  (Å² per bin)")
        ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("Sigma profiles", x=0.0, ha="left", fontsize=12,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.text(0, -0.03, "Shaded band is |σ| < 0.0084, the non-polar region. "
             "Right of it is electron-rich surface (H-bond acceptor) under "
             "the usual COSMO-SAC convention; left is donor. Run --validate "
             "to confirm that convention before reading these as donor and "
             "acceptor.", fontsize=7.5, color=MUTED, va="top")
    _watermark(fig, engine)
    fig.savefig(out / "t4_sigma_profiles.png")
    plt.close(fig)


def fig_polymer_properties(polymer_props, out):
    """
    Z-scored polymer property heatmap.

    Deliberately not a PCA: with four polymers a projection is rank-deficient
    and its loadings are decided by which of four points sits where. The
    heatmap shows every value instead.
    """
    df = pd.DataFrame(polymer_props).T
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    z = (df - df.mean()) / df.std(ddof=0).replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(1.15 * len(df.columns) + 4.0,
                                    0.6 * len(df) + 3.0))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([c.replace("_", " ") for c in df.columns],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=9)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)

    for i in range(len(df)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            zz = z.iloc[i, j]
            ax.text(j, i, f"{val:,.4g}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(zz) > 1.2 else INK)

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("z-score across the four polymers", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title("Polymer properties", pad=22)
    ax.text(0, 1.01, "cells show the raw value; colour is the z-score · "
            "no PCA: four points cannot support one",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.savefig(out / "t4_polymer_properties.png")
    plt.close(fig)


# ===========================================================================
# Validation
# ===========================================================================
REFERENCES = {
    "water": "O", "methanol": "CO", "acetone": "CC(=O)C",
    "dmso": "CS(=O)(=O)C", "acetic acid": "CC(=O)O",
    "n-hexane": "CCCCCC", "benzene": "c1ccccc1",
    "N-methylacetamide": "CNC(=O)C",
}


def validate(cache):
    """
    Confirm the sign convention before any descriptor is trusted.

    Acetone and DMSO are the discriminating cases: both accept hydrogen bonds
    and neither donates. If either comes out donor-dominant, the convention is
    inverted and the acceptor/donor masks in profile_descriptors must swap.
    """
    print(f"{'molecule':<20}{'acceptor':>10}{'donor':>10}{'nonpolar':>10}"
          f"{'HBA':>6}{'HBD':>6}")
    rows = {}
    for name, smi in REFERENCES.items():
        prf, area, _ = cache.get(canonical_parent(smi))
        d = profile_descriptors(prf)
        mol = Chem.MolFromSmiles(smi)
        rows[name] = d
        print(f"{name:<20}{d['acceptor_frac']:>10.3f}{d['donor_frac']:>10.3f}"
              f"{d['nonpolar_frac']:>10.3f}"
              f"{Descriptors.NumHAcceptors(mol):>6}"
              f"{Descriptors.NumHDonors(mol):>6}")
        if abs(d["total_area"] - area) > max(1.0, 0.02 * area):
            print(f"  ! profile sums to {d['total_area']:.1f} but area is "
                  f"{area:.1f} -- grid or units may be wrong")

    print()
    verdicts = []
    for name in ("acetone", "dmso"):
        if name in rows:
            d = rows[name]
            verdicts.append(d["acceptor_frac"] > d["donor_frac"])
    if verdicts and all(verdicts):
        print("convention OK: acceptor-only molecules are acceptor-dominant")
    elif verdicts:
        print("!! INVERTED: acetone/DMSO came out donor-dominant. Swap the "
              "acceptor and donor masks in profile_descriptors.")
    for name in ("n-hexane", "benzene"):
        if name in rows and rows[name]["nonpolar_frac"] < 0.8:
            print(f"!! {name} is only {rows[name]['nonpolar_frac']:.0%} "
                  "non-polar -- check the grid inference")

    print("\nthreshold sensitivity (acetone):")
    prf, _, _ = cache.get(canonical_parent(REFERENCES["acetone"]))
    for thr in (0.0070, 0.0084, 0.0100, 0.0120):
        d = profile_descriptors(prf, thr)
        print(f"  {thr:.4f}  acceptor={d['acceptor_frac']:.3f}  "
              f"donor={d['donor_frac']:.3f}  "
              f"nonpolar={d['nonpolar_frac']:.3f}")


# ===========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--input", help="formulation table")
    p.add_argument("--apis", default=None, help="candidate API list")
    p.add_argument("--holdout", default=None, help="held-out API list")
    p.add_argument("--outdir", default="interaction_out")
    p.add_argument("--model-dir", default="../model",
                   help="directory holding ads_molecule_classes.py")
    p.add_argument("--polymer-props", default="polymer_properties.json")
    p.add_argument("--polymer-smiles", default=None,
                   help="JSON of polymer SMILES; defaults to the map inside "
                        "data_processing_class_physical_stability.py")
    p.add_argument("--engine", choices=("real", "mock"), default="real")
    p.add_argument("--reference-loadings", default="10,25,40")
    p.add_argument("--rh", type=float, default=DEFAULT_RH)
    p.add_argument("--t-storage", type=float, default=DEFAULT_T_STORAGE_C)
    p.add_argument("--cache", default="sigma_cache.npz")
    p.add_argument("--validate", action="store_true",
                   help="check the sigma sign convention and exit")
    a = p.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    polymer_props = json.loads(Path(a.polymer_props).read_text())
    if a.polymer_smiles:
        polymer_smiles = json.loads(Path(a.polymer_smiles).read_text())
    else:
        polymer_smiles = _load_polymer_smiles_from_source(a.model_dir)

    bad = [n for n, v in polymer_smiles.items()
           if Chem.MolFromSmiles(_first_smiles(v)) is None]
    if bad:
        print(f"ERROR: polymer SMILES do not parse: {', '.join(bad)}")
        print("The copy in data_processing_class_physical_stability.py is "
              "truncated. Supply intact strings via --polymer-smiles.")
        return 1

    if a.engine == "mock":
        print("WARNING: mock engine -- figures are watermarked and the "
              "numbers are not physically meaningful.\n")
        engine = MockEngine(polymer_smiles)
    else:
        engine = TennetEngine(a.model_dir, polymer_smiles, a.t_storage)
    cache = ProfileCache(engine, a.cache)

    if a.validate:
        validate(cache)
        cache.save()
        return 0

    if not a.input:
        p.error("--input is required unless --validate is given")

    df = load_formulations(a.input)
    candidates = load_api_list(a.apis) if a.apis else None
    holdout = load_api_list(a.holdout) if a.holdout else None
    apis = build_api_table(df, candidates, holdout)

    loadings = tuple(float(x) for x in a.reference_loadings.split(","))
    print(f"\n{len(apis)} APIs x {len(polymer_props)} polymers · "
          f"reference loadings {loadings}")

    table = build_interaction_table(apis, df, polymer_props, polymer_smiles,
                                    cache, engine, loadings, a.rh, a.t_storage)
    cache.save()
    table.to_csv(out / "interaction_features.csv", index=False)
    print(f"interaction_features.csv: {len(table)} rows, "
          f"{table.shape[1]} columns")

    obs = table[table["source"] == "observed"].dropna(subset=["target"])
    if len(obs):
        print("\nseparation in interaction space (observed only):")
        for col in ("ln_gamma_api", "storage_delta_t", "pair_hb_match",
                    "pair_profile_overlap"):
            if col in obs:
                g = obs.groupby("target")[col].median()
                if len(g) == 2:
                    print(f"  {col:<22} stable {g.get(0, np.nan):>9.3f}   "
                          f"unstable {g.get(1, np.nan):>9.3f}")

    fig_interaction_space(table, out, engine)
    fig_complementarity(table, out, engine)
    fig_sigma_profiles(cache, apis, polymer_smiles, out, engine)
    fig_polymer_properties(polymer_props, out)
    print(f"\nwrote figures to {out}")
    return 0


def _load_polymer_smiles_from_source(model_dir):
    """Lift POLYMER_SMILES_MAP out of the source without importing torch."""
    import ast
    import re
    path = Path(model_dir) / "data_processing_class_physical_stability.py"
    src = path.read_text()
    start = src.index("self.POLYMER_SMILES_MAP = {")
    block = src[start + len("self.POLYMER_SMILES_MAP = "):]
    depth = 0
    for i, ch in enumerate(block):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(re.sub(r"#.*", "", block[:i + 1]))
    raise ValueError(f"Could not parse POLYMER_SMILES_MAP from {path}")


if __name__ == "__main__":
    sys.exit(main())
