#!/usr/bin/env python3
"""
Simple baseline models: regularized logistic regression and trees.

Built for 7 API groups, not for 37 rows. Everything here follows from that:

  Pooled metrics, not per-fold.   Holding out one API leaves a test fold that
                                  is often single-class, so per-fold AUC is
                                  undefined or meaningless. Out-of-fold
                                  predictions are pooled and scored once.

  Thresholds chosen inside.       find_optimal_threshold on the test fold is
                                  the most common way to inflate an F2. The
                                  threshold is picked by an inner
                                  leave-one-API-out loop over the training
                                  groups only.

  Trivial baselines alongside.    Majority class and loading-only logistic
                                  regression. If the chemistry model does not
                                  beat loading alone, the chemistry is
                                  decoration.

  Median imputation, not sentinels. nan_rep_dict's -1 / -0.25 markers let a
                                  tree split on missingness, which is sound
                                  for trees and wrong for logistic
                                  regression, which reads them as magnitudes.

  Permutation null.               Labels shuffled within API, whole pipeline
                                  rerun. The gap between real and null is the
                                  result; the absolute number is not.

  L2 before L1 for deployment.    Lasso deletes a weakly-supported feature,
                                  so the model can never respond to it. Ridge
                                  shrinks it, so when it moves far outside the
                                  training range it still contributes. With 7
                                  groups L1 will zero most polymer and
                                  complementarity terms -- see --keep.

Outputs
-------
  model_comparison.png    pooled AUC / F2 per model against the baselines
  model_coefficients.png  standardized LR coefficients and tree importances
  oof_predictions.csv     out-of-fold probability for every row
  model_metrics.csv       the summary table

Usage
-----
    python baseline_model.py --features model_features.csv
    python baseline_model.py --features interaction_out/interaction_features.csv \\
        --feature-list g_excess,tg_mix_c,api_loading_percent,pair_hb_match
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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (fbeta_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

STABLE, UNSTABLE = "#2a78d6", "#d03b3b"
MUTED, GRID, INK, ACCENT = "#52514e", "#e4e3de", "#0b0b0b", "#eb6834"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11,
                     "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})

DROP_ALWAYS = {"api", "polymer", "polymer_name", "target", "truth", "role",
               "source", "smiles", "canonical_smiles", "api_name"}

# One feature per mechanism. At 7 groups this is more defensible than a
# data-driven set, and far easier to explain when the held-out results land.
MECHANISTIC = ["g_excess", "tg_mix_c", "api_loading_percent", "pair_hb_match"]


def make_models(seed=0):
    """
    Deliberately small. With 7 effective observations, anything with more
    than a handful of effective parameters overfits in a way this dataset
    cannot detect.
    """
    num = lambda est: Pipeline([                      # noqa: E731
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", est)])
    raw = lambda est: Pipeline([                      # noqa: E731
        ("imp", SimpleImputer(strategy="median")),
        ("m", est)])
    return {
        "logreg_l2": num(LogisticRegression(penalty="l2", C=0.3,
                                            solver="liblinear",
                                            random_state=seed)),
        "logreg_l1": num(LogisticRegression(penalty="l1", C=0.5,
                                            solver="liblinear",
                                            random_state=seed)),
        "tree_d2": raw(DecisionTreeClassifier(max_depth=2, min_samples_leaf=4,
                                              random_state=seed)),
        "rf": raw(RandomForestClassifier(n_estimators=400, max_depth=3,
                                         min_samples_leaf=3, random_state=seed,
                                         n_jobs=-1)),
    }


def pick_threshold(y, prob, target_recall=0.90):
    """
    Threshold at a target recall, and the F2-maximising threshold.

    Mirrors find_optimal_threshold in wet_stability_model_dev_v0.py. Called
    only on inner-loop predictions, never on the outer test fold.
    """
    if len(np.unique(y)) < 2:
        return 0.5, 0.5
    prec, rec, thr = precision_recall_curve(y, prob)
    if len(thr) == 0:
        return 0.5, 0.5
    idx = np.where(rec >= target_recall)[0]
    t_recall = thr[min(idx[-1], len(thr) - 1)] if len(idx) else thr[0]
    f2 = [fbeta_score(y, (prob >= t).astype(int), beta=2, zero_division=0)
          for t in thr]
    return float(t_recall), float(thr[int(np.argmax(f2))])


def cross_validate(X, y, groups, model, seed=0, target_recall=0.90):
    """
    Leave-one-API-out, with the decision threshold chosen by an inner
    leave-one-API-out loop over the training groups only.
    """
    logo = LeaveOneGroupOut()
    oof = np.full(len(y), np.nan)
    thresholds = []

    for tr, te in logo.split(X, y, groups):
        Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr], groups.iloc[tr]
        if ytr.nunique() < 2:
            continue

        # Inner loop: out-of-fold probabilities on the training groups,
        # used only to choose a threshold.
        inner = np.full(len(ytr), np.nan)
        for itr, ite in LeaveOneGroupOut().split(Xtr, ytr, gtr):
            if ytr.iloc[itr].nunique() < 2:
                continue
            m = clone_model(model)
            m.fit(Xtr.iloc[itr], ytr.iloc[itr])
            inner[ite] = m.predict_proba(Xtr.iloc[ite])[:, 1]
        ok = ~np.isnan(inner)
        if ok.sum() > 4:
            thresholds.append(pick_threshold(ytr[ok], inner[ok],
                                             target_recall))

        m = clone_model(model)
        m.fit(Xtr, ytr)
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]

    t_recall = float(np.median([t[0] for t in thresholds])) if thresholds else 0.5
    t_f2 = float(np.median([t[1] for t in thresholds])) if thresholds else 0.5
    return oof, t_recall, t_f2


def clone_model(model):
    from sklearn.base import clone
    return clone(model)


def score(y, oof, t_recall, t_f2):
    ok = ~np.isnan(oof)
    y, p = y[ok], oof[ok]
    out = {"n_scored": int(ok.sum())}
    out["pooled_auc"] = (roc_auc_score(y, p) if y.nunique() > 1 else np.nan)
    for name, t in (("at_recall90", t_recall), ("at_maxF2", t_f2)):
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        out[f"recall_{name}"] = tp / (tp + fn) if (tp + fn) else np.nan
        out[f"precision_{name}"] = tp / (tp + fp) if (tp + fp) else np.nan
        out[f"f2_{name}"] = fbeta_score(y, pred, beta=2, zero_division=0)
        out[f"threshold_{name}"] = t
    return out


def permutation_null(X, y, groups, model, rounds=20, seed=0):
    """Shuffle labels within API; preserves grouping and per-API balance."""
    rng = np.random.default_rng(seed)
    aucs = []
    for r in range(rounds):
        ys = y.copy()
        for g in groups.unique():
            m = (groups == g).values
            ys.iloc[m] = rng.permutation(y.iloc[m].values)
        if ys.nunique() < 2:
            continue
        oof, _, _ = cross_validate(X, ys, groups, model, seed=seed + r)
        ok = ~np.isnan(oof)
        if ok.sum() and ys[ok].nunique() > 1:
            aucs.append(roc_auc_score(ys[ok], oof[ok]))
    return aucs


# ===========================================================================
def fig_comparison(metrics, null_aucs, out):
    order = metrics.sort_values("pooled_auc", ascending=True)
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.45 * len(order) + 3.2),
                             gridspec_kw={"wspace": 0.38})

    ax = axes[0]
    colors = [MUTED if n.startswith("baseline") else STABLE
              for n in order.index]
    ax.barh(y, order["pooled_auc"], color=colors, height=0.6, zorder=3)
    ax.axvline(0.5, color=UNSTABLE, ls="--", lw=1.1)
    ax.annotate("chance", (0.5, len(order) - 0.4), xytext=(4, 0),
                textcoords="offset points", fontsize=7.5, color=UNSTABLE)
    if null_aucs:
        lo, hi = np.percentile(null_aucs, [2.5, 97.5])
        ax.axvspan(lo, hi, color=UNSTABLE, alpha=0.10, zorder=1)
        ax.annotate("permutation null (95%)", ((lo + hi) / 2, -0.75),
                    ha="center", fontsize=7, color=UNSTABLE)
    for i, v in enumerate(order["pooled_auc"]):
        if np.isfinite(v):
            ax.annotate(f"{v:.2f}", (v, i), xytext=(5, 0),
                        textcoords="offset points", va="center",
                        fontsize=7.6, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(order.index, fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("pooled out-of-fold ROC AUC")
    ax.grid(axis="y", visible=False)
    ax.set_title("Discrimination", pad=22)
    ax.text(0, 1.01, "grey = trivial baseline · one AUC over all pooled "
            "out-of-fold predictions", transform=ax.transAxes, fontsize=8,
            color=MUTED)

    ax = axes[1]
    width = 0.38
    ax.barh(y - width / 2, order["f2_at_maxF2"], height=width, color=STABLE,
            zorder=3, label="F2 at max-F2 threshold")
    ax.barh(y + width / 2, order["recall_at_recall90"], height=width,
            color=ACCENT, zorder=3, label="recall at the recall-0.90 threshold")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("score")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Operating points", pad=22)
    ax.text(0, 1.01, "thresholds chosen by an inner leave-one-API-out loop, "
            "never on the test fold", transform=ax.transAxes, fontsize=8,
            color=MUTED)

    fig.text(0, -0.02, "A model inside the shaded null band has not been "
             "shown to work. A model that does not clear the loading-only "
             "baseline has not been shown to need chemistry.",
             fontsize=7.5, color=MUTED, va="top")
    fig.savefig(out / "model_comparison.png")
    plt.close(fig)


def fig_coefficients(X, y, groups, models, out):
    """Standardized LR coefficients and tree importances, refit on all data."""
    panels = []
    for name in ("logreg_l2", "tree_d2", "rf"):
        if name not in models:
            continue
        m = clone_model(models[name])
        m.fit(X, y)
        est = m.named_steps["m"]
        if hasattr(est, "coef_"):
            vals = est.coef_.ravel()
            label = "standardized coefficient"
        else:
            vals = est.feature_importances_
            label = "impurity importance"
        panels.append((name, pd.Series(vals, index=X.columns), label))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels),
                             0.34 * X.shape[1] + 3.0), squeeze=False)
    for ax, (name, vals, label) in zip(axes[0], panels):
        s = vals.reindex(vals.abs().sort_values().index)
        colors = [UNSTABLE if v > 0 else STABLE for v in s]
        ax.barh(np.arange(len(s)), s.values, color=colors, height=0.6,
                zorder=3)
        ax.set_yticks(np.arange(len(s)))
        ax.set_yticklabels(s.index, fontsize=7.6)
        ax.axvline(0, color=GRID, lw=0.9)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        ax.set_title(name, pad=22)
        if hasattr(vals, "name"):
            ax.text(0, 1.01, "red pushes toward unstable (1)",
                    transform=ax.transAxes, fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.text(0, -0.02, "Refit on all rows, for interpretation only — these "
             "coefficients were not validated out of fold, and at 7 groups "
             "their signs can flip when one API is removed.",
             fontsize=7.5, color=MUTED, va="top")
    fig.savefig(out / "model_coefficients.png")
    plt.close(fig)


# ===========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--target-col", default="target")
    p.add_argument("--group-col", default="api")
    p.add_argument("--feature-list", default=None,
                   help="comma-separated features; default is the mechanistic "
                        "preset where available, else the 6 with most variance")
    p.add_argument("--outdir", default="model_out")
    p.add_argument("--target-recall", type=float, default=0.90)
    p.add_argument("--null-rounds", type=int, default=20)
    p.add_argument("--no-null", action="store_true")
    p.add_argument("--keep", default="",
                   help="comma-separated features always included alongside "
                        "the chosen set; a trailing * matches by prefix. Use "
                        "for axes you intend to extrapolate along (poly_*), "
                        "which 7 groups cannot justify on fit alone.")
    a = p.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.features)
    if "source" in df.columns:
        df = df[df["source"] == "observed"]
    df = df[df[a.target_col].notna()].reset_index(drop=True)
    y = df[a.target_col].astype(int)
    groups = df[a.group_col].astype(str)

    numeric = [c for c in df.columns
               if c not in DROP_ALWAYS and c != a.target_col
               and pd.api.types.is_numeric_dtype(df[c])]
    if a.feature_list:
        feats = [c.strip() for c in a.feature_list.split(",")]
        missing = [c for c in feats if c not in df.columns]
        if missing:
            print(f"ERROR: not in the table: {', '.join(missing)}")
            return 1
    else:
        feats = [c for c in MECHANISTIC if c in numeric]
        if len(feats) < 3:
            var = df[numeric].std().sort_values(ascending=False)
            feats = list(var.head(6).index)
            print("mechanistic preset unavailable; using the 6 highest-"
                  "variance features instead")

    protected = []
    for pat in a.keep.split(","):
        pat = pat.strip()
        if not pat:
            continue
        protected += ([c for c in numeric if c.startswith(pat[:-1])]
                      if pat.endswith("*") else
                      ([pat] if pat in numeric else []))
    protected = [c for c in sorted(set(protected)) if c not in feats]
    if protected:
        print(f"protected features added ({len(protected)}): "
              f"{', '.join(protected)}")
        feats = feats + protected
    X = df[feats].copy()

    print(f"{len(df)} rows · {groups.nunique()} groups · {len(feats)} features")
    print(f"features: {', '.join(feats)}")
    print(f"class balance: {int((y == 0).sum())} stable / "
          f"{int((y == 1).sum())} unstable")
    per_group = y.groupby(groups).nunique()
    print(f"groups with both classes: {int((per_group > 1).sum())} of "
          f"{groups.nunique()}  (single-class groups cannot be scored "
          "on their own)\n")

    models = make_models()
    models["baseline_majority"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", DummyClassifier(strategy="prior"))])
    loading_col = ("api_loading_percent" if "api_loading_percent" in df
                   else ("x_api" if "x_api" in df else None))

    rows, oof_store = {}, {}
    for name, model in models.items():
        Xm = X
        if name == "baseline_majority" and loading_col:
            Xm = df[[loading_col]]
        oof, t_rec, t_f2 = cross_validate(Xm, y, groups, model,
                                          target_recall=a.target_recall)
        rows[name] = score(y, oof, t_rec, t_f2)
        oof_store[name] = oof

    if loading_col:
        lo = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler()),
                       ("m", LogisticRegression(random_state=0))])
        oof, t_rec, t_f2 = cross_validate(df[[loading_col]], y, groups, lo,
                                          target_recall=a.target_recall)
        rows[f"baseline_{loading_col}_only"] = score(y, oof, t_rec, t_f2)
        oof_store[f"baseline_{loading_col}_only"] = oof

    metrics = pd.DataFrame(rows).T
    metrics.to_csv(out / "model_metrics.csv")
    show = ["n_scored", "pooled_auc", "f2_at_maxF2", "recall_at_recall90",
            "precision_at_recall90"]
    print(metrics[show].round(3).to_string())

    null_aucs = []
    if not a.no_null:
        print(f"\npermutation null ({a.null_rounds} rounds, logreg_l2) ...")
        null_aucs = permutation_null(X, y, groups, models["logreg_l2"],
                                     a.null_rounds)
        if null_aucs:
            best = metrics["pooled_auc"].max()
            pval = (1 + sum(n >= best for n in null_aucs)) / (1 + len(null_aucs))
            print(f"  best model AUC {best:.3f} · null mean "
                  f"{np.mean(null_aucs):.3f} · p ≈ {pval:.3f}")
            if pval > 0.1:
                print("  Inside the null. No model here has been shown to "
                      "work; do not pick a winner from this table.")

    pd.DataFrame(oof_store).assign(
        api=groups.values, target=y.values).to_csv(
        out / "oof_predictions.csv", index=False)
    fig_comparison(metrics, null_aucs, out)
    fig_coefficients(X, y, groups, models, out)
    print(f"\nwrote to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
