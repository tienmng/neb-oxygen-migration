#!/usr/bin/env python3
"""
Tier 5 -- landscape roughness.

Tiers 1-4 place molecules and formulations in various spaces. This asks the
prior question: in each of those spaces, does being close actually predict
having the same outcome?

If close pairs reliably share an outcome the space is smooth and distance in
it is worth trusting. If disagreement stays near the base rate even for
near-identical pairs, that space is rough and a coverage claim made in it is
false comfort. Tier 6 needs one distance metric; this is how you choose it
rather than assume it.

Outputs
-------
  t5_roughness.png       disagreement vs distance, one panel per space
  t5_cliffs.png          the conditional activity cliffs, ranked
  roughness_summary.csv  per-space smoothness statistics
  cliffs.csv             same polymer, similar loading, similar API, opposite outcome

Two things this is careful about
--------------------------------
Pairs are not independent: with 7 APIs and ~37 formulations, each API appears
in many pairs. Confidence intervals come from a bootstrap over APIs, not over
pairs, which is the difference between an honest interval and a flattering one.

Within-API pairs are reported separately as a baseline. Two formulations of
the same molecule differ only by polymer and loading, so their disagreement
rate is the irreducible formulation effect. If a chemical space's nearest
pairs disagree as often as that baseline, knowing the chemistry added nothing.

Usage
-----
    python roughness.py \\
        --input "../data/raw/Formulation_example_input_file.xlsx" \\
        --interaction interaction_out/interaction_features.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage_simple import (  # noqa: E402
    STABLE, UNSTABLE, MUTED, GRID, INK, FEATURE_SETS,
    load_formulations, load_api_list, build_api_table,
    fingerprints, tanimoto, feature_frame,
)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                     "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})

SPACE_NOTES = {
    "ecfp4": "Tanimoto on ECFP4 · API-level",
    "structural": "scaled graph descriptors · API-level",
    "physchem": "scaled computed properties · API-level",
    "interaction": "scaled thermo + kinetic + complementarity · formulation-level",
}


# ===========================================================================
# Distances
# ===========================================================================
def api_level_distances(apis, formulations):
    """
    Pairwise distance between formulations, induced by their APIs.

    Two formulations of the same API are at distance zero in every API-level
    space; they are kept and flagged, not discarded, because their
    disagreement rate is the baseline the chemical spaces must beat.
    """
    spaces = {}
    order = list(apis.index)
    pos = {a: i for i, a in enumerate(order)}

    sim = tanimoto(fingerprints(apis["smiles"]))
    spaces["ecfp4"] = 1.0 - sim

    for name in FEATURE_SETS:
        X = feature_frame(apis, name)
        if X.shape[1] < 2 or len(X) < 4:
            continue
        Xs = StandardScaler().fit_transform(X.values)
        full = np.full((len(order), len(order)), np.nan)
        idx = [pos[a] for a in X.index]
        d = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=-1)
        for a, ia in enumerate(idx):
            full[ia, idx] = d[a]
        spaces[name] = full

    return spaces, pos


def interaction_distances(table, formulations):
    """
    Formulation-level distance from the tier-4 feature table.

    Only observed rows are used -- reference rows have no outcome, so they
    cannot contribute a pair.
    """
    obs = table[(table.get("source") == "observed")
                & table["target"].notna()].copy()
    if obs.empty:
        return None, None
    cols = [c for c in obs.columns
            if c.startswith(("ln_gamma", "delta_ln", "g_excess", "tg_mix",
                             "storage_delta", "process_delta", "pair_"))]
    cols = [c for c in cols if obs[c].notna().all() and obs[c].std() > 0]
    if len(cols) < 2:
        return None, None
    Xs = StandardScaler().fit_transform(obs[cols].values)
    d = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=-1)
    key = list(zip(obs["api"], obs["polymer"],
                   obs["api_loading_percent"].round(3)))
    return d, key


# ===========================================================================
# Pair table
# ===========================================================================
def build_pairs(formulations, spaces, pos, inter_d=None, inter_key=None):
    """One row per pair of labelled formulations, with distance in each space."""
    df = formulations.dropna(subset=["target"]).reset_index(drop=True)
    n = len(df)
    if n < 4:
        raise ValueError(f"Only {n} labelled formulations; too few for pairs.")

    inter_lookup = {}
    if inter_d is not None:
        for i, k in enumerate(inter_key):
            inter_lookup[k] = i

    rows = []
    for i in range(n - 1):
        ri = df.iloc[i]
        for j in range(i + 1, n):
            rj = df.iloc[j]
            rec = {
                "api_i": ri["api"], "api_j": rj["api"],
                "polymer_i": ri["polymer_name"], "polymer_j": rj["polymer_name"],
                "loading_i": ri["api_loading_percent"],
                "loading_j": rj["api_loading_percent"],
                "same_api": ri["api"] == rj["api"],
                "same_polymer": ri["polymer_name"] == rj["polymer_name"],
                "d_loading": abs(ri["api_loading_percent"]
                                 - rj["api_loading_percent"]),
                "disagree": int(ri["target"] != rj["target"]),
            }
            for name, mat in spaces.items():
                a, b = pos.get(ri["api"]), pos.get(rj["api"])
                rec[f"dist_{name}"] = (mat[a, b] if a is not None
                                       and b is not None else np.nan)
            if inter_d is not None:
                ki = inter_lookup.get((ri["api"], ri["polymer_name"],
                                       round(ri["api_loading_percent"], 3)))
                kj = inter_lookup.get((rj["api"], rj["polymer_name"],
                                       round(rj["api_loading_percent"], 3)))
                rec["dist_interaction"] = (inter_d[ki, kj]
                                           if ki is not None and kj is not None
                                           else np.nan)
            rows.append(rec)
    return pd.DataFrame(rows)


# ===========================================================================
# Statistics
# ===========================================================================
def wilson(k, n, z=1.96):
    """Wilson interval; behaves sensibly at k=0 and k=n, unlike the normal one."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def grouped_bootstrap_rho(pairs, col, n_boot=400, seed=0):
    """
    Spearman(distance, disagreement) with APIs resampled, not pairs.

    Resampling pairs would treat each of several hundred pairs as independent
    evidence when they are generated by a handful of molecules.
    """
    from scipy.stats import spearmanr
    sub = pairs[(~pairs["same_api"]) & pairs[col].notna()]
    if len(sub) < 10:
        return np.nan, (np.nan, np.nan)
    point = spearmanr(sub[col], sub["disagree"]).statistic

    apis = sorted(set(sub["api_i"]) | set(sub["api_j"]))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        keep = set(rng.choice(apis, size=len(apis), replace=True))
        s = sub[sub["api_i"].isin(keep) & sub["api_j"].isin(keep)]
        if len(s) > 10 and s["disagree"].nunique() > 1 and s[col].nunique() > 1:
            draws.append(spearmanr(s[col], s["disagree"]).statistic)
    if not draws:
        return point, (np.nan, np.nan)
    return point, (float(np.nanpercentile(draws, 2.5)),
                   float(np.nanpercentile(draws, 97.5)))


def summarise(pairs, spaces_present):
    """Per-space smoothness statistics, plus the within-API baseline."""
    within = pairs[pairs["same_api"]]
    baseline = within["disagree"].mean() if len(within) else np.nan
    overall = pairs[~pairs["same_api"]]["disagree"].mean()

    rows = []
    for name in spaces_present:
        col = f"dist_{name}"
        sub = pairs[(~pairs["same_api"]) & pairs[col].notna()]
        if len(sub) < 10:
            continue
        q = sub[col].quantile(0.10)
        near = sub[sub[col] <= q]
        rho, (lo, hi) = grouped_bootstrap_rho(pairs, col)
        k, n = int(near["disagree"].sum()), len(near)
        wlo, whi = wilson(k, n)
        rows.append({
            "space": name,
            "pairs": len(sub),
            "nearest_decile_n": n,
            "nearest_decile_disagree": k / n if n else np.nan,
            "nearest_decile_lo": wlo,
            "nearest_decile_hi": whi,
            "all_pairs_disagree": sub["disagree"].mean(),
            "within_api_baseline": baseline,
            "spearman_rho": rho,
            "rho_lo": lo, "rho_hi": hi,
        })
    return pd.DataFrame(rows), baseline, overall


def find_cliffs(pairs, sim_threshold=0.5, loading_tol=10.0):
    """
    Conditional activity cliffs.

    A cliff only means something with the context held fixed: same polymer,
    similar loading, similar API, opposite outcome. Unconditioned "cliffs"
    mostly reflect formulation differences, not chemistry.
    """
    if "dist_ecfp4" not in pairs:
        return pd.DataFrame()
    c = pairs[(~pairs["same_api"])
              & pairs["same_polymer"]
              & (pairs["d_loading"] <= loading_tol)
              & (pairs["dist_ecfp4"] <= 1.0 - sim_threshold)
              & (pairs["disagree"] == 1)].copy()
    if c.empty:
        return c
    c["tanimoto"] = 1.0 - c["dist_ecfp4"]
    cols = ["api_i", "api_j", "polymer_i", "loading_i", "loading_j",
            "d_loading", "tanimoto"]
    return c[cols].sort_values("tanimoto", ascending=False).reset_index(drop=True)


# ===========================================================================
# Figures
# ===========================================================================
def fig_roughness(pairs, summary, baseline, overall, out, n_bins=8):
    spaces = list(summary["space"])
    if not spaces:
        print("  skipped t5_roughness.png (no usable spaces)")
        return
    fig, axes = plt.subplots(1, len(spaces), sharey=True,
                             figsize=(4.3 * len(spaces) + 1.2, 5.0),
                             squeeze=False)

    for ax, name in zip(axes[0], spaces):
        col = f"dist_{name}"
        sub = pairs[(~pairs["same_api"]) & pairs[col].notna()].copy()
        # Percentile x-axis so panels with different units are comparable.
        sub["pct"] = sub[col].rank(pct=True) * 100
        edges = np.linspace(0, 100, n_bins + 1)
        xs, ys, los, his = [], [], [], []
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            b = sub[(sub["pct"] > lo_e) & (sub["pct"] <= hi_e)]
            if len(b) < 3:
                continue
            k, n = int(b["disagree"].sum()), len(b)
            wlo, whi = wilson(k, n)
            xs.append((lo_e + hi_e) / 2)
            ys.append(k / n)
            los.append(wlo)
            his.append(whi)

        ax.fill_between(xs, los, his, color=STABLE, alpha=0.15, lw=0)
        ax.plot(xs, ys, color=STABLE, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.1, zorder=3)
        ax.axhline(overall, color=MUTED, ls="--", lw=1.0)
        ax.annotate("all pairs", (100, overall), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=7,
                    color=MUTED)
        if np.isfinite(baseline):
            ax.axhline(baseline, color=UNSTABLE, ls=":", lw=1.2)
            ax.annotate("same-API baseline", (100, baseline), xytext=(-2, 3),
                        textcoords="offset points", ha="right", fontsize=7,
                        color=UNSTABLE)

        row = summary[summary["space"] == name].iloc[0]
        ax.set_xlabel("distance percentile  (0 = most similar)")
        ax.set_title(name, pad=22)
        ax.text(0, 1.01, f"{SPACE_NOTES.get(name, '')} · ρ = "
                f"{row['spearman_rho']:.2f} "
                f"[{row['rho_lo']:.2f}, {row['rho_hi']:.2f}]",
                transform=ax.transAxes, fontsize=7.5, color=MUTED)
        ax.set_ylim(-0.03, 1.03)

    axes[0][0].set_ylabel("fraction of pairs with opposite outcome")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.text(0.0, 0.995, "Landscape roughness", fontsize=12,
             fontweight="semibold", color=INK, ha="left", va="top")
    fig.text(0.0, 0.95, "A smooth space slopes up from left to right. A flat "
             "line means distance carries no information about outcome.",
             fontsize=8, color=MUTED, ha="left", va="top")
    fig.text(0, -0.03, "Bands are Wilson intervals on each bin. ρ is "
             "Spearman(distance, disagreement) with its 95% interval from a "
             "bootstrap over APIs, not over pairs — pairs from 7 molecules "
             "are not 600 independent observations.",
             fontsize=7.5, color=MUTED, va="top")
    fig.savefig(out / "t5_roughness.png")
    plt.close(fig)


def fig_cliffs(cliffs, out, limit=15):
    if cliffs.empty:
        print("  no conditional cliffs found; skipping t5_cliffs.png")
        return
    c = cliffs.head(limit).iloc[::-1]
    labels = [f"{r.api_i} vs {r.api_j}\n{r.polymer_i} · "
              f"{r.loading_i:g}% / {r.loading_j:g}%" for r in c.itertuples()]
    y = np.arange(len(c))

    fig, ax = plt.subplots(figsize=(9.5, 0.52 * len(c) + 2.8))
    ax.barh(y, c["tanimoto"], color=UNSTABLE, height=0.6, zorder=3)
    for i, v in enumerate(c["tanimoto"]):
        ax.annotate(f"{v:.2f}", (v, i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7.6,
                    color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Tanimoto similarity between the two APIs")
    ax.grid(axis="y", visible=False)
    ax.set_title("Conditional activity cliffs", pad=22)
    ax.text(0, 1.01, "same polymer · similar loading · similar API · "
            "opposite outcome", transform=ax.transAxes, fontsize=8,
            color=MUTED)
    fig.text(0, -0.02, "Each of these is either real chemistry or a bad row. "
             "At this dataset size a single mislabelled formulation can move "
             "a model, so they are worth checking by hand before anything "
             "is fitted.", fontsize=7.5, color=MUTED, va="top")
    fig.savefig(out / "t5_cliffs.png")
    plt.close(fig)


# ===========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="formulation table")
    p.add_argument("--apis", default=None)
    p.add_argument("--holdout", default=None)
    p.add_argument("--interaction", default=None,
                   help="interaction_features.csv from tier 4")
    p.add_argument("--outdir", default="roughness_out")
    p.add_argument("--sim-threshold", type=float, default=0.5,
                   help="minimum Tanimoto for a pair to count as a cliff")
    p.add_argument("--loading-tol", type=float, default=10.0,
                   help="maximum loading difference (wt %%) for a cliff")
    a = p.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_formulations(a.input)
    if "target" not in df.columns:
        print("ERROR: no target column; tier 5 needs outcomes.")
        return 1
    candidates = load_api_list(a.apis) if a.apis else None
    holdout = load_api_list(a.holdout) if a.holdout else None
    apis = build_api_table(df, candidates, holdout)

    spaces, pos = api_level_distances(apis, df)
    inter_d = inter_key = None
    if a.interaction and Path(a.interaction).exists():
        table = pd.read_csv(a.interaction)
        inter_d, inter_key = interaction_distances(table, df)
        if inter_d is None:
            print("note: interaction table had too few usable columns")

    pairs = build_pairs(df, spaces, pos, inter_d, inter_key)
    present = [s for s in list(spaces) + (["interaction"] if inter_d is not None
                                          else [])
               if f"dist_{s}" in pairs and pairs[f"dist_{s}"].notna().any()]

    summary, baseline, overall = summarise(pairs, present)
    n_lab = int(df["target"].notna().sum())
    print(f"\n{n_lab} labelled formulations · {len(pairs)} pairs · "
          f"{int(pairs['same_api'].sum())} share an API")
    print(f"disagreement: {overall:.2f} across different-API pairs, "
          f"{baseline:.2f} within the same API\n")
    if not summary.empty:
        cols = ["space", "pairs", "nearest_decile_disagree",
                "all_pairs_disagree", "spearman_rho", "rho_lo", "rho_hi"]
        print(summary[cols].round(3).to_string(index=False))
        summary.to_csv(out / "roughness_summary.csv", index=False)

        best = summary.sort_values("spearman_rho", ascending=False).iloc[0]
        if best["rho_lo"] > 0:
            print(f"\nsmoothest space: {best['space']} "
                  f"(rho {best['spearman_rho']:.2f}, interval excludes zero) "
                  "-- use this distance in tier 6")
        else:
            print("\nNo space has a rho interval excluding zero. Distance "
                  "does not predict outcome here at this sample size; treat "
                  "any applicability-domain threshold as provisional.")

    cliffs = find_cliffs(pairs, a.sim_threshold, a.loading_tol)
    if not cliffs.empty:
        cliffs.to_csv(out / "cliffs.csv", index=False)
        print(f"\n{len(cliffs)} conditional cliff(s):")
        print(cliffs.head(10).round(3).to_string(index=False))
    else:
        print("\nno conditional cliffs at the current thresholds")

    pairs.to_csv(out / "pairs.csv", index=False)
    fig_roughness(pairs, summary, baseline, overall, out)
    fig_cliffs(cliffs, out)
    print(f"\nwrote to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
