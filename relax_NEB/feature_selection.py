#!/usr/bin/env python3
"""
Feature selection stability, for a small grouped dataset.

With 7 APIs you have 7 effectively independent observations, not 37. At that
size a single feature-selection run is a coin toss dressed as a result. This
measures how often each feature survives across leave-one-API-out folds, and
compares that against what happens when the labels are shuffled.

The order of operations is the point:

  1. Unsupervised pruning   near-zero variance, then one of each correlated
                            pair. Touches no labels, so it needs no nesting
                            and costs nothing statistically.
  2. Effective-N audit      API-level features are constant within an API, so
                            they have as many distinct values as there are
                            APIs -- seven -- however many rows the table has.
                            Formulation-level features genuinely vary across
                            37 rows. The two are not equally supportable.
  3. Fold-wise selection    L1 logistic regression, random-forest permutation
                            importance, and Boruta if available, each refitted
                            inside every fold. Selection that sees all folds
                            leaks into the CV estimate it is judged by.
  4. Permutation null       the same pipeline on labels shuffled within API.
                            If real performance sits inside the shuffled
                            distribution there is no signal to select on, and
                            every frequency above is noise.

Outputs
-------
  fs_stability.png        per-feature selection frequency, real vs shuffled
  fs_null.png             CV AUC, real vs permutation null
  selection_stability.csv the frequency table
  pruning_report.csv      what step 1 removed and why

Usage
-----
    python feature_selection.py --features model_features.csv
    python feature_selection.py --features interaction_out/interaction_features.csv

Expects a target column, a group column (default 'api'), and numeric features.
Needs only numpy, pandas, scikit-learn and matplotlib.
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

STABLE, UNSTABLE = "#2a78d6", "#d03b3b"
MUTED, GRID, INK = "#52514e", "#e4e3de", "#0b0b0b"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                     "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})

DROP_ALWAYS = {"api", "polymer", "polymer_name", "target", "truth", "role",
               "source", "smiles", "canonical_smiles", "api_name"}


# ===========================================================================
# Step 1 -- unsupervised pruning
# ===========================================================================
def prune_unsupervised(X, nzv_threshold=0.01, corr_threshold=0.9):
    """
    Drop near-constant and near-duplicate features.

    Uses no labels, so it does not have to live inside the CV loop and cannot
    leak. Doing it first is what makes the supervised step affordable.
    """
    report, keep = [], list(X.columns)

    nunique = X.nunique()
    std = X.std(ddof=0)
    for col in list(keep):
        if nunique[col] <= 1 or std[col] <= nzv_threshold * abs(X[col].mean() or 1):
            keep.remove(col)
            report.append({"feature": col, "reason": "near-zero variance",
                           "detail": f"{nunique[col]} distinct, sd={std[col]:.4g}"})

    corr = X[keep].corr(method="spearman").abs()
    for i, a in enumerate(list(keep)):
        if a not in keep:
            continue
        for b in list(keep)[i + 1:]:
            if b not in keep or a not in corr or b not in corr:
                continue
            r = corr.loc[a, b]
            if r >= corr_threshold:
                # Keep the one with more distinct values; it carries more.
                drop = b if nunique[a] >= nunique[b] else a
                kept = a if drop == b else b
                keep.remove(drop)
                report.append({"feature": drop,
                               "reason": "correlated duplicate",
                               "detail": f"rho={r:.3f} with {kept}"})
                if drop == a:
                    break
    return keep, pd.DataFrame(report)


# ===========================================================================
# Step 2 -- effective N
# ===========================================================================
def effective_n(X, groups):
    """
    How many distinct values each feature really has, per group structure.

    A feature constant within every API varies only across APIs: its effective
    sample size is the number of APIs, not the number of rows. This is the
    difference between a feature 37 rows can support and one only 7 can.
    """
    rows = []
    for col in X.columns:
        within = X.groupby(groups)[col].nunique()
        api_level = bool((within <= 1).all())
        rows.append({
            "feature": col,
            "level": "API" if api_level else "formulation",
            "distinct_values": int(X[col].nunique()),
            "effective_n": int(groups.nunique() if api_level else X[col].nunique()),
        })
    return pd.DataFrame(rows).sort_values(["level", "effective_n"],
                                          ascending=[True, False])


# ===========================================================================
# Step 3 -- fold-wise selection
# ===========================================================================
def fold_selection(X, y, groups, seed=0, top_k=5, fast=False):
    """
    Refit every selector inside every leave-one-API-out fold.

    fast=True drops Boruta and shrinks the forests. The permutation null runs
    the pipeline tens of times, and a full Boruta in each round costs minutes
    for a selector whose null behaviour the cheap methods already capture.

    Returns a frequency table: how many folds picked each feature, by method.
    """
    logo = LeaveOneGroupOut()
    counts = {m: {c: 0 for c in X.columns}
              for m in ("l1_logreg", "rf_permutation", "boruta")}
    n_folds = 0
    aucs = []

    boruta_cls = None if fast else _load_boruta()
    n_trees = 150 if fast else 400
    n_repeats = 5 if fast else 10

    for train_idx, test_idx in logo.split(X, y, groups):
        Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
        Xte, yte = X.iloc[test_idx], y.iloc[test_idx]
        if ytr.nunique() < 2:
            continue
        n_folds += 1

        # L1 logistic regression: non-zero coefficients are the selection.
        pipe = Pipeline([("s", StandardScaler()),
                         ("m", LogisticRegression(penalty="l1", C=0.5,
                                                  solver="liblinear",
                                                  random_state=seed))])
        pipe.fit(Xtr, ytr)
        coef = pipe.named_steps["m"].coef_.ravel()
        for col, c in zip(X.columns, coef):
            if abs(c) > 1e-8:
                counts["l1_logreg"][col] += 1
        if yte.nunique() > 1:
            aucs.append(roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))

        # Random forest, ranked by permutation importance on the training fold.
        rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed,
                                    min_samples_leaf=2, n_jobs=-1)
        rf.fit(Xtr, ytr)
        imp = permutation_importance(rf, Xtr, ytr, n_repeats=n_repeats,
                                     random_state=seed, n_jobs=-1)
        for col in np.array(X.columns)[np.argsort(imp.importances_mean)[::-1][:top_k]]:
            counts["rf_permutation"][col] += 1

        if boruta_cls is not None:
            try:
                est = RandomForestClassifier(n_estimators=200,
                                             random_state=seed,
                                             min_samples_leaf=2, n_jobs=-1)
                bor = boruta_cls(est, n_estimators="auto", random_state=seed,
                                 verbose=0, max_iter=40)
                bor.fit(Xtr.values, ytr.values)
                for col, picked in zip(X.columns, bor.support_):
                    if picked:
                        counts["boruta"][col] += 1
            except Exception:
                pass

    freq = pd.DataFrame(counts)
    freq["n_folds"] = n_folds
    for m in ("l1_logreg", "rf_permutation", "boruta"):
        freq[m + "_frac"] = freq[m] / max(n_folds, 1)
    methods = ["l1_logreg", "rf_permutation"]
    if freq["boruta"].sum() > 0:
        methods.append("boruta")
    freq["mean_frac"] = freq[[m + "_frac" for m in methods]].mean(axis=1)
    freq["cheap_frac"] = freq[["l1_logreg_frac",
                               "rf_permutation_frac"]].mean(axis=1)
    return freq.sort_values("mean_frac", ascending=False), aucs, n_folds


def _load_boruta():
    """Use the project's boruta_py if it is importable; otherwise skip it."""
    for candidate in (Path(__file__).resolve().parent,
                      Path(__file__).resolve().parent.parent / "model"):
        if (candidate / "boruta_py.py").exists():
            sys.path.insert(0, str(candidate))
            try:
                from boruta_py import BorutaPy
                return BorutaPy
            except Exception:
                return None
    return None


# ===========================================================================
# Step 4 -- permutation null
# ===========================================================================
def permutation_null(X, y, groups, n_rounds=20, seed=0):
    """
    Shuffle labels WITHIN each API and rerun the fold-wise pipeline.

    Shuffling within group preserves the group structure and the class
    balance per API, so the null answers the right question: given this
    grouping, how well would we do by chance?
    """
    rng = np.random.default_rng(seed)
    null_aucs, null_freq = [], []
    for r in range(n_rounds):
        y_shuf = y.copy()
        for g in groups.unique():
            m = (groups == g).values
            y_shuf.iloc[m] = rng.permutation(y.iloc[m].values)
        if y_shuf.nunique() < 2:
            continue
        freq, aucs, _ = fold_selection(X, y_shuf, groups, seed=seed + r,
                                       fast=True)
        null_aucs.extend(aucs)
        null_freq.append(freq["mean_frac"])
    # fast mode has no Boruta column, so mean_frac is over two methods;
    # compared like-for-like against the real run's two cheap methods below.
    null_table = (pd.concat(null_freq, axis=1).mean(axis=1)
                  if null_freq else pd.Series(dtype=float))
    return null_aucs, null_table


# ===========================================================================
# Figures
# ===========================================================================
def fig_stability(freq, null_table, out, limit=20):
    f = freq.head(limit).iloc[::-1]
    y = np.arange(len(f))
    fig, ax = plt.subplots(figsize=(9.8, 0.34 * len(f) + 3.0))

    ax.barh(y, f["mean_frac"], color=STABLE, height=0.62, zorder=3,
            label="real labels")
    if not null_table.empty:
        ax.barh(y, null_table.reindex(f.index).values, color=UNSTABLE,
                height=0.26, zorder=4, label="shuffled labels (null)")

    for i, (v, idx) in enumerate(zip(f["mean_frac"], f.index)):
        detail = (f"{int(f.loc[idx, 'l1_logreg'])}/"
                  f"{int(f.loc[idx, 'rf_permutation'])}/"
                  f"{int(f.loc[idx, 'boruta'])}")
        ax.annotate(f"{v:.2f}   L1/RF/Bor {detail}", (v, i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7.2,
                    color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(f.index, fontsize=7.6)
    ax.set_xlim(0, 1.3)
    ax.set_xlabel("fraction of leave-one-API-out folds selecting the feature")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Selection stability", pad=22)
    ax.text(0, 1.01, f"{int(freq['n_folds'].iloc[0])} folds · averaged over "
            "L1 logistic regression, RF permutation importance and Boruta",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.text(0, -0.02, "A feature selected in most folds on real labels and "
             "rarely on shuffled ones is doing work. One whose two bars are "
             "the same length is being chosen by chance, however plausible "
             "its name.", fontsize=7.5, color=MUTED, va="top")
    fig.savefig(out / "fs_stability.png")
    plt.close(fig)


def fig_null(aucs, null_aucs, out):
    if not aucs:
        print("  skipped fs_null.png (no fold had both classes in the test set)")
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bins = np.linspace(0, 1, 21)
    if null_aucs:
        ax.hist(null_aucs, bins=bins, color=UNSTABLE, alpha=0.45,
                label=f"shuffled labels (n={len(null_aucs)})", zorder=2)
    ax.hist(aucs, bins=bins, color=STABLE, alpha=0.75,
            label=f"real labels (n={len(aucs)})", zorder=3)
    ax.axvline(0.5, color=MUTED, ls="--", lw=1.0)
    ax.annotate("chance", (0.5, ax.get_ylim()[1]), xytext=(4, -10),
                textcoords="offset points", fontsize=7.5, color=MUTED)
    ax.axvline(np.mean(aucs), color=STABLE, lw=1.8)
    if null_aucs:
        pval = (1 + sum(n >= np.mean(aucs) for n in null_aucs)) / (1 + len(null_aucs))
        note = (f"mean real AUC {np.mean(aucs):.2f} · "
                f"mean null {np.mean(null_aucs):.2f} · "
                f"permutation p ≈ {pval:.3f}")
    else:
        note = f"mean real AUC {np.mean(aucs):.2f}"
    ax.set_xlabel("per-fold ROC AUC")
    ax.set_ylabel("folds")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Signal against the permutation null", pad=22)
    ax.text(0, 1.01, note, transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.text(0, -0.03, "Per-fold AUC over a handful of held-out rows is very "
             "noisy; the comparison against the shuffled distribution is the "
             "informative part, not the absolute value.",
             fontsize=7.5, color=MUTED, va="top")
    fig.savefig(out / "fs_null.png")
    plt.close(fig)


# ===========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="feature table CSV")
    p.add_argument("--target-col", default="target")
    p.add_argument("--group-col", default="api")
    p.add_argument("--outdir", default="selection_out")
    p.add_argument("--corr-threshold", type=float, default=0.9)
    p.add_argument("--null-rounds", type=int, default=20)
    p.add_argument("--top-k", type=int, default=5,
                   help="features counted as selected per fold by RF importance")
    p.add_argument("--no-null", action="store_true")
    a = p.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.features)
    if "source" in df.columns:
        df = df[df["source"] == "observed"]
    df = df[df[a.target_col].notna()].reset_index(drop=True)

    y = df[a.target_col].astype(int)
    groups = df[a.group_col].astype(str)
    feats = [c for c in df.columns
             if c not in DROP_ALWAYS and c != a.target_col
             and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].copy()
    X = X.loc[:, X.notna().all()]

    print(f"{len(df)} labelled rows · {groups.nunique()} groups "
          f"({a.group_col}) · {X.shape[1]} numeric features")
    print(f"class balance: {int((y == 0).sum())} / {int((y == 1).sum())}\n")

    keep, pruning = prune_unsupervised(X, corr_threshold=a.corr_threshold)
    if not pruning.empty:
        pruning.to_csv(out / "pruning_report.csv", index=False)
        print(f"step 1 -- pruned {len(pruning)} of {X.shape[1]} features")
        for _, r in pruning.head(8).iterrows():
            print(f"    {r['feature']:<28} {r['reason']:<22} {r['detail']}")
        if len(pruning) > 8:
            print(f"    ... {len(pruning) - 8} more in pruning_report.csv")
    X = X[keep]
    print(f"    {X.shape[1]} features remain\n")

    eff = effective_n(X, groups)
    eff.to_csv(out / "effective_n.csv", index=False)
    n_api = int((eff["level"] == "API").sum())
    print(f"step 2 -- {n_api} API-level features (effective n = "
          f"{groups.nunique()}), {len(eff) - n_api} formulation-level")
    print("    API-level features are supported by the number of APIs, not "
          "the number of rows.\n")

    freq, aucs, n_folds = fold_selection(X, y, groups, top_k=a.top_k)
    print(f"step 3 -- selection across {n_folds} leave-one-{a.group_col}-out "
          "folds")
    show = ["l1_logreg", "rf_permutation", "boruta", "mean_frac"]
    print(freq[show].head(10).round(2).to_string())

    null_aucs, null_table = ([], pd.Series(dtype=float))
    if not a.no_null:
        print(f"\nstep 4 -- permutation null, {a.null_rounds} rounds ...")
        null_aucs, null_table = permutation_null(X, y, groups, a.null_rounds)
        if aucs and null_aucs:
            pval = (1 + sum(n >= np.mean(aucs) for n in null_aucs)) / (1 + len(null_aucs))
            print(f"    real AUC {np.mean(aucs):.3f} · "
                  f"null {np.mean(null_aucs):.3f} · p ≈ {pval:.3f}")
            if pval > 0.1:
                print("    Real performance sits inside the null. There is no "
                      "detectable signal at this sample size, so the "
                      "frequencies above should not be read as a feature set.")

    freq["null_mean_frac"] = null_table.reindex(freq.index)
    freq.to_csv(out / "selection_stability.csv")
    fig_stability(freq, null_table, out)
    fig_null(aucs, null_aucs, out)
    print(f"\nwrote to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
