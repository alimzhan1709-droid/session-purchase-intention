"""
uci_online_shoppers_kbtu_pipeline.py

Companion program for the master's thesis "Investigation and prediction of user
behavior in e-commerce applications" (KBTU). It is the executable specification
of the empirical analysis: it validates the UCI CSV, constructs the exact
zero-safe features, fits the pre-declared candidates, selects model family and
threshold only on October, evaluates the frozen pipelines once on
November-December and regenerates the synchronized tables, diagnostics and
figures reported in Chapters 3 and 4.

    python uci_online_shoppers_kbtu_pipeline.py --data online_shoppers_intention.csv \
                                                --output uci_corrected_results \
                                                --figures thesis_corrected_figures

Design controls: SEED = 42; PageValues isolated into a separate feature setting;
raw numeric month never used as a predictor; preprocessing fitted on the training
months only; model family and threshold selected on October alone; no refit after
selection; no test-driven redesign.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

SEED = 42
N_BOOT = 1500
N_PERM_REPEATS = 10

MONTHS = {"Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jul": 7,
          "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

EXPECTED_COLUMNS = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
    "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType",
    "Weekend", "Revenue"]

EXPECTED_ROWS = 12330
EXPECTED_POSITIVES = 1908

CAT = ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]


# ------------------------------------------------------------------ 1. validation
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(df, path):
    """Step 1: schema, counts, labels, missingness, validity and duplicates."""
    report = {}
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    assert not missing, f"missing required columns: {missing}"
    assert len(df) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(df)}"

    positives = int(df["Revenue"].astype(bool).sum())
    assert positives == EXPECTED_POSITIVES, \
        f"expected {EXPECTED_POSITIVES} purchase sessions, got {positives}"

    unknown_months = sorted(set(df["Month"]) - set(MONTHS))
    assert not unknown_months, f"unknown month labels: {unknown_months}"

    numeric = df.select_dtypes(include=[np.number])
    report["rows"] = len(df)
    report["positives"] = positives
    report["prevalence"] = positives / len(df)
    report["missing_cells"] = int(df.isna().sum().sum())
    report["non_finite_cells"] = int((~np.isfinite(numeric.to_numpy(float))).sum())
    report["negative_counts_or_durations"] = int(
        (numeric[[c for c in numeric.columns if "Duration" in c
                  or c in ("Administrative", "Informational", "ProductRelated")]] < 0).sum().sum())
    report["exact_duplicate_rows"] = int(df.duplicated().sum())
    report["months_present"] = sorted(set(df["Month"]), key=lambda m: MONTHS[m])
    report["sha256"] = sha256(path)
    report["file"] = str(path)
    return report


# ------------------------------------------------------------------ 2-3. features
def build(df):
    """Steps 2-3: binary target, month mapping and deterministic zero-safe features."""
    df = df.copy()
    df["Revenue"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    df["month_num"] = df["Month"].map(MONTHS)

    df["total_pages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["total_duration"] = (df["Administrative_Duration"] + df["Informational_Duration"]
                            + df["ProductRelated_Duration"])
    den = df["total_pages"].where(df["total_pages"] > 0, 1)   # zero-safe denominator
    df["duration_per_page"] = df["total_duration"] / den
    df["exit_minus_bounce"] = df["ExitRates"] - df["BounceRates"]
    df["admin_share"] = df["Administrative"] / den
    df["info_share"] = df["Informational"] / den
    df["product_share"] = df["ProductRelated"] / den
    df["log_total_pages"] = np.log1p(df["total_pages"])
    df["log_total_duration"] = np.log1p(df["total_duration"])
    df["log_product_duration"] = np.log1p(df["ProductRelated_Duration"])

    # cyclical month is a sensitivity representation only; raw month_num is never a predictor
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    return df


BEHAVIOR = ["Administrative", "Administrative_Duration", "Informational",
            "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "SpecialDay", "Weekend",
            "total_pages", "total_duration", "duration_per_page", "exit_minus_bounce",
            "admin_share", "info_share", "product_share",
            "log_total_pages", "log_total_duration", "log_product_duration"] + CAT

FULL = BEHAVIOR + ["PageValues"]
CYCLICAL = ["month_sin", "month_cos"]
# engagement-only drops the technical categories and SpecialDay
ENGAGEMENT = [f for f in BEHAVIOR if f not in CAT and f != "SpecialDay"]

FEATURE_SETTINGS = {
    "behavior no month": BEHAVIOR,
    "behavior cyclical month": BEHAVIOR + CYCLICAL,
    "engagement only": ENGAGEMENT,
    "full no month": FULL,
    "full cyclical month": FULL + CYCLICAL,
}

PRIMARY = "behavior no month"
UPPER_BOUND = "full no month"


# ------------------------------------------------------------------ 4. partitions
def partition(df):
    """Step 4: February-September fitting, October validation, frozen November-December test."""
    return (df[df.month_num <= 9].copy(),
            df[df.month_num == 10].copy(),
            df[df.month_num >= 11].copy())


# ------------------------------------------------------------------ 5. pipelines
def make_pipe(model, feats):
    """Step 5: imputation, scaling and one-hot encoding fitted inside the pipeline."""
    cats = [f for f in feats if f in CAT]
    num = [f for f in feats if f not in CAT]
    blocks = [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("sc", StandardScaler())]), num)]
    if cats:
        blocks.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                        ("oh", OneHotEncoder(handle_unknown="ignore"))]), cats))
    return Pipeline([("pre", ColumnTransformer(blocks)), ("clf", model)])


def models():
    """The three pre-declared supervised families and their fixed configurations."""
    return {
        "Logistic Regression": LogisticRegression(C=1.0, class_weight="balanced",
                                                  solver="liblinear", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=80, max_depth=10,
                                                min_samples_leaf=10, max_features="sqrt",
                                                class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=30, colsample_bytree=0.9,
                                   reg_lambda=1.0, class_weight="balanced",
                                   random_state=SEED, verbose=-1),
    }


# ------------------------------------------------------------------ 6. thresholds
GRID = np.arange(0.02, 0.98 + 1e-9, 0.005)


def f1_over_grid(y, p):
    """F1 for the whole threshold grid at once, so the bootstrap stays tractable."""
    y = np.asarray(y).astype(int)
    pred = p[:, None] >= GRID[None, :]
    tp = (pred & (y[:, None] == 1)).sum(0)
    denom = pred.sum(0) + y.sum()
    return np.divide(2 * tp, denom, out=np.zeros(len(GRID)), where=denom > 0)


def best_threshold(y, p):
    """First threshold attaining the maximum F1 — the pre-declared selection convention."""
    scores = f1_over_grid(y, p)
    k = int(np.argmax(scores))
    return float(GRID[k]), float(scores[k])


def metrics(y, p, thr):
    """Threshold and ranking metrics. Brier score is defined only for a calibrated
    probability, so it is left undefined for raw scores such as PageValues."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    is_probability = float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    return dict(precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0),
                f1=f1_score(y, yh, zero_division=0),
                roc_auc=roc_auc_score(y, p),
                ap=average_precision_score(y, p),
                brier=brier_score_loss(y, p) if is_probability else None,
                accuracy=(tp + tn) / len(y),
                n=len(y), positives=int(y.sum()),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


# ------------------------------------------------------------------ 6-7. candidates
def fit_candidates(tr, va, te):
    """Steps 6-7: fit every candidate, select on October, evaluate the frozen pipeline once."""
    rows, store = [], {}
    for setting, feats in FEATURE_SETTINGS.items():
        for name, model in models().items():
            pipe = make_pipe(model, feats).fit(tr[feats], tr.Revenue)
            pv = pipe.predict_proba(va[feats])[:, 1]
            thr, val_f1 = best_threshold(va.Revenue, pv)
            pt = pipe.predict_proba(te[feats])[:, 1]
            rows.append(dict(feature_set=setting, model=name, val_f1=val_f1, threshold=thr,
                             **metrics(te.Revenue.values, pt, thr)))
            store[(setting, name)] = dict(pipe=pipe, feats=feats, threshold=thr,
                                          val_f1=val_f1, val_proba=pv, test_proba=pt)
    return pd.DataFrame(rows), store


def select(cand, setting):
    """October F1 is the only selection criterion; test metrics never influence it."""
    block = cand[cand.feature_set == setting]
    return block.loc[block.val_f1.idxmax(), "model"]


# ------------------------------------------------------------------ 8. baselines
def baselines(tr, va, te):
    """Step 8: majority, fixed and October-tuned PageValues, and prevalence-constant baselines."""
    y = te.Revenue.values
    rows = []

    zero = np.zeros(len(te))
    rows.append(dict(baseline="Majority class", threshold=0.5,
                     **metrics(y, zero, 0.5)))

    # the stated strict rule: PageValues > 0, with the raw score used for ranking metrics
    raw_te = te.PageValues.values.astype(float)
    fixed = metrics(y, raw_te, np.nextafter(0.0, 1.0))
    rows.append(dict(baseline="PageValues > 0 rule / raw score", threshold=0.0, **fixed))

    # October-tuned cut-point over the observed raw values
    raw_va = va.PageValues.values.astype(float)
    cuts = np.unique(raw_va)
    scores = [f1_score(va.Revenue.values, (raw_va > c).astype(int), zero_division=0) for c in cuts]
    k = int(np.argmax(scores))
    cut, val_f1 = float(cuts[k]), float(scores[k])
    tuned = metrics(y, raw_te, np.nextafter(cut, np.inf))
    rows.append(dict(baseline="PageValues tuned rule", val_f1=val_f1, threshold=cut, **tuned))

    prevalence = float(tr.Revenue.mean())
    const = np.full(len(te), prevalence)
    rows.append(dict(baseline=f"Training-prevalence constant (p = {prevalence:.3f})",
                     threshold=0.5, **metrics(y, const, 0.5)))
    return pd.DataFrame(rows), prevalence


# ------------------------------------------------------------------ 9. uncertainty
def stratified_indices(rng, y, size=None):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])


def bootstrap_ci(y, p, thr, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    f1s, rocs, aps = [], [], []
    for _ in range(n_boot):
        idx = stratified_indices(rng, y)
        yb, pb = y[idx], p[idx]
        f1s.append(f1_score(yb, (pb >= thr).astype(int), zero_division=0))
        rocs.append(roc_auc_score(yb, pb))
        aps.append(average_precision_score(yb, pb))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return dict(f1_lo=q(f1s)[0], f1_hi=q(f1s)[1], roc_lo=q(rocs)[0], roc_hi=q(rocs)[1],
                ap_lo=q(aps)[0], ap_hi=q(aps)[1])


def paired_contrast(y, p_model, thr_model, p_rule, thr_rule, n_boot=N_BOOT):
    """Paired stratified bootstrap of model-minus-rule differences on identical resamples."""
    rng = np.random.default_rng(SEED)
    d_f1, d_roc, d_ap = [], [], []
    for _ in range(n_boot):
        idx = stratified_indices(rng, y)
        yb = y[idx]
        a, b = p_model[idx], p_rule[idx]
        d_f1.append(f1_score(yb, (a >= thr_model).astype(int), zero_division=0)
                    - f1_score(yb, (b > thr_rule).astype(int), zero_division=0))
        d_roc.append(roc_auc_score(yb, a) - roc_auc_score(yb, b))
        d_ap.append(average_precision_score(yb, a) - average_precision_score(yb, b))
    out = []
    for label, obs, draws in [
            ("F1", f1_score(y, (p_model >= thr_model).astype(int), zero_division=0)
                   - f1_score(y, (p_rule > thr_rule).astype(int), zero_division=0), d_f1),
            ("ROC-AUC", roc_auc_score(y, p_model) - roc_auc_score(y, p_rule), d_roc),
            ("Average precision", average_precision_score(y, p_model)
                                  - average_precision_score(y, p_rule), d_ap)]:
        draws = np.asarray(draws)
        out.append(dict(metric=label, observed=obs,
                        ci_low=float(np.percentile(draws, 2.5)),
                        ci_high=float(np.percentile(draws, 97.5)),
                        share_positive=float((draws > 0).mean())))
    return pd.DataFrame(out)


def calibration_table(y, p, n_bins=10):
    """Deciles of the predicted score with observed frequency in each bin."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    rows = []
    for i, b in enumerate(bins, 1):
        rows.append(dict(decile=i, n=len(b), mean_predicted=float(p[b].mean()),
                         observed_rate=float(y[b].mean())))
    return pd.DataFrame(rows)


def grouped_permutation_importance(pipe, feats, X, y, n_repeats=N_PERM_REPEATS):
    """Permute each raw column, so all encoded columns derived from it move together."""
    rng = np.random.default_rng(SEED)
    base = average_precision_score(y, pipe.predict_proba(X[feats])[:, 1])
    rows = []
    for col in feats:
        drops = []
        for _ in range(n_repeats):
            shuffled = X[feats].copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(base - average_precision_score(
                y, pipe.predict_proba(shuffled)[:, 1]))
        rows.append(dict(feature=col, ap_drop_mean=float(np.mean(drops)),
                         ap_drop_std=float(np.std(drops))))
    return (pd.DataFrame(rows).sort_values("ap_drop_mean", ascending=False)
            .reset_index(drop=True), base)


def drift_table(tr, te, feats):
    rows = []
    for col in feats:
        if col in CAT:
            continue
        a, b = tr[col].to_numpy(float), te[col].to_numpy(float)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or np.nan
        rows.append(dict(feature=col, train_mean=a.mean(), test_mean=b.mean(),
                         train_median=float(np.median(a)), test_median=float(np.median(b)),
                         standardized_diff=(b.mean() - a.mean()) / pooled,
                         ks_statistic=float(ks_2samp(a, b).statistic)))
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False).reset_index(drop=True)


def error_profile(te, p, thr, label):
    y = te.Revenue.values
    yh = (p >= thr).astype(int)
    group = np.where((y == 0) & (yh == 0), "TN",
             np.where((y == 0) & (yh == 1), "FP",
             np.where((y == 1) & (yh == 0), "FN", "TP")))
    d = te.assign(_group=group)
    rows = []
    for g in ["TN", "FP", "FN", "TP"]:
        part = d[d._group == g]
        rows.append(dict(model=label, group=g, n=len(part),
                         median_product_pages=float(part.ProductRelated.median()),
                         median_product_duration=float(part.ProductRelated_Duration.median()),
                         mean_bounce=float(part.BounceRates.mean()),
                         mean_exit=float(part.ExitRates.mean()),
                         median_pagevalues=float(part.PageValues.median())))
    return pd.DataFrame(rows)


def month_sensitivity(te, p, thr, label):
    rows = []
    for name, part in [("Nov+Dec", te), ("Nov", te[te.month_num == 11]),
                       ("Dec", te[te.month_num == 12])]:
        mask = te.index.isin(part.index)
        rows.append(dict(model=label, period=name, prevalence=float(part.Revenue.mean()),
                         **metrics(part.Revenue.values, p[mask], thr)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 10. figures
def figures(df, tr, va, te, store, sel, base_rows, cal, imp_full, imp_beh, drift, figdir):
    figdir.mkdir(parents=True, exist_ok=True)
    saved = []

    def finish(fig, name, title):
        fig.suptitle(title)
        fig.tight_layout()
        path = figdir / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path.name)

    order = sorted(MONTHS, key=lambda m: MONTHS[m])
    present = [m for m in order if m in set(df.Month)]

    fig, ax = plt.subplots(figsize=(5, 4))
    df.Revenue.value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_xticklabels(["No purchase", "Purchase"], rotation=0); ax.set_ylabel("Sessions")
    finish(fig, "fig01_class_balance.png", "Figure 1 - Target class balance")

    fig, ax = plt.subplots(figsize=(8, 2.6))
    spans = [("Train Feb-Sep", 2, 9, "#4c72b0"), ("Val Oct", 10, 10, "#dd8452"),
             ("Test Nov-Dec", 11, 12, "#55a868")]
    for label, a, b, c in spans:
        ax.barh(0, b - a + 1, left=a, color=c, edgecolor="white")
        ax.text((a + b + 1) / 2, 0, label, ha="center", va="center", color="white", fontsize=9)
    ax.set_yticks([]); ax.set_xlabel("Month number"); ax.set_xlim(2, 13)
    finish(fig, "fig02_holdout_design.png", "Figure 2 - Frozen month-based selection and holdout design")

    fig, ax = plt.subplots(figsize=(7, 4))
    rate = df.groupby("Month").Revenue.mean().reindex(present)
    rate.plot(kind="bar", ax=ax, color="#4c72b0"); ax.set_ylabel("Conversion rate")
    finish(fig, "fig03_monthly_conversion.png", "Figure 3 - Conversion rate by month")

    fig, ax = plt.subplots(figsize=(7, 4))
    df.Month.value_counts().reindex(present).plot(kind="bar", ax=ax, color="#4c72b0")
    ax.set_ylabel("Sessions")
    finish(fig, "fig04_monthly_sessions.png", "Figure 4 - Session count by month")

    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby("VisitorType").Revenue.mean().plot(kind="bar", ax=ax, color="#4c72b0")
    ax.set_ylabel("Conversion rate"); ax.set_xlabel("")
    finish(fig, "fig05_visitor_type.png", "Figure 5 - Conversion rate by visitor type")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    df.groupby("Weekend").Revenue.mean().plot(kind="bar", ax=axes[0], color="#4c72b0")
    axes[0].set_title("Weekend"); axes[0].set_ylabel("Conversion rate")
    df.groupby("SpecialDay").Revenue.mean().plot(kind="bar", ax=axes[1], color="#dd8452")
    axes[1].set_title("SpecialDay"); axes[1].set_ylabel("")
    finish(fig, "fig06_weekend_specialday.png", "Figure 6 - Conversion by weekend and special day")

    def by_outcome(col, name, num, title, logy=False):
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [df[df.Revenue == 0][col], df[df.Revenue == 1][col]]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["No purchase", "Purchase"])
        ax.set_ylabel(col)
        if logy:
            ax.set_yscale("symlog")
        finish(fig, name, title)

    by_outcome("ProductRelated", "fig07_product_pages.png", 7,
               "Figure 7 - Product-related pages by outcome")
    by_outcome("ProductRelated_Duration", "fig08_product_duration.png", 8,
               "Figure 8 - Product-related duration by outcome")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col in zip(axes, ["BounceRates", "ExitRates"]):
        ax.boxplot([df[df.Revenue == 0][col], df[df.Revenue == 1][col]], showfliers=False)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["No purchase", "Purchase"])
        ax.set_title(col)
    finish(fig, "fig09_bounce_exit.png", "Figure 9 - Bounce and exit rates by outcome")

    by_outcome("PageValues", "fig10_pagevalues.png", 10,
               "Figure 10 - PageValues by outcome", logy=True)

    def confusion_fig(label, name, title):
        s = store[(sel[label]["setting"], sel[label]["model"])]
        yh = (s["test_proba"] >= s["threshold"]).astype(int)
        cm = confusion_matrix(te.Revenue.values, yh, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"]); ax.set_yticks([0, 1], ["True 0", "True 1"])
        finish(fig, name, title)

    confusion_fig("full", "fig11_confusion_full.png",
                  "Figure 11 - Confusion matrix, validation-selected full model")
    confusion_fig("behavior", "fig12_confusion_behavior.png",
                  "Figure 12 - Confusion matrix, validation-selected behavior-only model")

    y = te.Revenue.values
    curves = [("Full " + sel["full"]["model"], store[(sel["full"]["setting"], sel["full"]["model"])]["test_proba"]),
              ("Behavior-only " + sel["behavior"]["model"],
               store[(sel["behavior"]["setting"], sel["behavior"]["model"])]["test_proba"]),
              ("Full Logistic Regression", store[(UPPER_BOUND, "Logistic Regression")]["test_proba"]),
              ("Raw PageValues", te.PageValues.values.astype(float))]

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, p in curves:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{label} ({roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=8)
    finish(fig, "fig13_roc.png", "Figure 13 - ROC curves on the frozen test period")

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, p in curves:
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=f"{label} (AP {average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"Prevalence ({y.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(fontsize=8)
    finish(fig, "fig14_precision_recall.png", "Figure 14 - Precision-recall curves")

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, part in cal.groupby("model"):
        ax.plot(part.mean_predicted, part.observed_rate, "o-", label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed rate"); ax.legend(fontsize=8)
    finish(fig, "fig15_calibration.png", "Figure 15 - Calibration by predicted-probability decile")

    for imp, name, title in [
            (imp_full, "fig16_importance_full.png",
             "Figure 16 - Grouped permutation importance, full model"),
            (imp_beh, "fig17_importance_behavior.png",
             "Figure 17 - Grouped permutation importance, behavior-only model")]:
        top = imp.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top.feature, top.ap_drop_mean, xerr=top.ap_drop_std, color="#4c72b0")
        ax.set_xlabel("Mean drop in average precision")
        finish(fig, name, title)

    s = store[(sel["full"]["setting"], sel["full"]["model"])]
    prof = error_profile(te, s["test_proba"], s["threshold"], "full")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(prof.group, prof.median_product_pages, color="#4c72b0")
    ax.set_ylabel("Median product-related pages")
    finish(fig, "fig18_error_depth.png", "Figure 18 - Session depth across prediction outcomes")

    top = drift.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.feature, top.ks_statistic, color="#dd8452")
    ax.set_xlabel("Kolmogorov-Smirnov statistic, training vs frozen test")
    finish(fig, "fig19_drift.png", "Figure 19 - Distribution shift between periods")

    return saved


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="online_shoppers_intention.csv")
    ap.add_argument("--output", default="uci_corrected_results")
    ap.add_argument("--figures", default="thesis_corrected_figures")
    a = ap.parse_args()

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    figdir = Path(a.figures)

    raw = pd.read_csv(a.data)
    report = validate(raw, Path(a.data))
    print(f"[1] input validated: {report['rows']} rows, {report['positives']} purchases, "
          f"{report['exact_duplicate_rows']} exact duplicate rows")

    df = build(raw)
    tr, va, te = partition(df)
    print(f"[4] partitions: train {len(tr)} / val {len(va)} / test {len(te)}; "
          f"prevalence {tr.Revenue.mean():.4f} / {va.Revenue.mean():.4f} / {te.Revenue.mean():.4f}")

    cand, store = fit_candidates(tr, va, te)
    cand.to_csv(out / "candidates.csv", index=False)
    print("[6] candidates fitted and evaluated once on the frozen test period")

    sel = {}
    for key, setting in [("behavior", PRIMARY), ("full", UPPER_BOUND)]:
        model = select(cand, setting)
        sel[key] = dict(setting=setting, model=model)
        s = store[(setting, model)]
        print(f"    selected {key:9} -> {model} (Oct F1 {s['val_f1']:.3f}, thr {s['threshold']:.3f})")

    base, train_prev = baselines(tr, va, te)
    base.to_csv(out / "baselines.csv", index=False)
    print("[8] baselines computed")

    y = te.Revenue.values
    raw_pv = te.PageValues.values.astype(float)

    ci_rows = []
    for key in ("full", "behavior"):
        s = store[(sel[key]["setting"], sel[key]["model"])]
        ci_rows.append(dict(predictor=f"{key} {sel[key]['model']}",
                            f1=f1_score(y, (s["test_proba"] >= s["threshold"]).astype(int),
                                        zero_division=0),
                            roc_auc=roc_auc_score(y, s["test_proba"]),
                            ap=average_precision_score(y, s["test_proba"]),
                            **bootstrap_ci(y, s["test_proba"], s["threshold"])))
    ci_rows.append(dict(predictor="PageValues score / > 0 rule",
                        f1=f1_score(y, (raw_pv > 0).astype(int), zero_division=0),
                        roc_auc=roc_auc_score(y, raw_pv),
                        ap=average_precision_score(y, raw_pv),
                        **bootstrap_ci(y, raw_pv, np.nextafter(0.0, 1.0))))
    pd.DataFrame(ci_rows).to_csv(out / "bootstrap_intervals.csv", index=False)

    sfull = store[(sel["full"]["setting"], sel["full"]["model"])]
    contrast = paired_contrast(y, sfull["test_proba"], sfull["threshold"], raw_pv, 0.0)
    contrast.to_csv(out / "paired_contrast.csv", index=False)
    print("[9] bootstrap intervals and paired contrast computed")

    cal = pd.concat([
        calibration_table(y, store[(sel[k]["setting"], sel[k]["model"])]["test_proba"])
        .assign(model=f"{k} {sel[k]['model']}") for k in ("full", "behavior")])
    cal.to_csv(out / "calibration.csv", index=False)

    imp = {}
    for key in ("full", "behavior"):
        s = store[(sel[key]["setting"], sel[key]["model"])]
        imp[key], base_ap = grouped_permutation_importance(s["pipe"], s["feats"], te, y)
        imp[key].to_csv(out / f"importance_{key}.csv", index=False)
    print("[9] grouped permutation importance computed")

    drift = drift_table(tr, te, [f for f in FULL if f not in CAT])
    drift.to_csv(out / "drift.csv", index=False)

    profiles = pd.concat([error_profile(te, store[(sel[k]["setting"], sel[k]["model"])]["test_proba"],
                                        store[(sel[k]["setting"], sel[k]["model"])]["threshold"], k)
                          for k in ("full", "behavior")])
    profiles.to_csv(out / "error_profiles.csv", index=False)

    months = pd.concat([month_sensitivity(te, store[(sel[k]["setting"], sel[k]["model"])]["test_proba"],
                                          store[(sel[k]["setting"], sel[k]["model"])]["threshold"], k)
                        for k in ("full", "behavior")])
    months.to_csv(out / "nov_dec_sensitivity.csv", index=False)

    saved = figures(df, tr, va, te, store, sel, base, cal, imp["full"], imp["behavior"], drift, figdir)
    print(f"[10] {len(saved)} figures written to {figdir}/")

    meta = dict(seed=SEED, n_bootstrap=N_BOOT, permutation_repeats=N_PERM_REPEATS,
                python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                scikit_learn=sklearn.__version__, scipy=scipy.__version__,
                matplotlib=matplotlib.__version__,
                feature_settings={k: v for k, v in FEATURE_SETTINGS.items()},
                selected=sel, training_prevalence=train_prev, input=report, figures=saved)
    try:
        import lightgbm
        meta["lightgbm"] = lightgbm.__version__
    except Exception:
        meta["lightgbm"] = None
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"\nDone. Tables and diagnostics in {out}/, figures in {figdir}/")


if __name__ == "__main__":
    main()
