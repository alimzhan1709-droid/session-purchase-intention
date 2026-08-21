"""
thesis_gaps_addon.py - companion to uci_online_shoppers_kbtu_pipeline.py

Covers five points that could not be settled by editing the text,
because they require real computation on the data.

Run:
    python thesis_gaps_addon.py --data online_shoppers_intention.csv --out gaps_results

What it computes:
  1. Separate November and December results          -> nov_dec_split.csv
  2. Bootstrap of October F1 (selection instability) -> selection_stability.csv
  3. Platt / isotonic calibration                    -> calibration_fix.csv
  4. Sensitivity without the 125 duplicates          -> duplicates_sensitivity.csv
  5. "Literature-style" protocol (random split + SMOTE) -> literature_inflation.csv

Point 5 is the main one: it shows numerically that results published on this
dataset are inflated. That is the core contribution and needs to be in numbers.
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
try:                                   # sklearn >= 1.6
    from sklearn.frozen import FrozenEstimator
    _FROZEN = True
    def freeze(est): return FrozenEstimator(est)
except ImportError:                    # sklearn < 1.6 used cv="prefit"
    _FROZEN = False
    def freeze(est): return est
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss)
from lightgbm import LGBMClassifier

SEED = 42
MONTHS = {"Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jul": 7,
          "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

CAT = ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]


# ---------------------------------------------------------------- data
def build(df):
    """The same deterministic features as in the main pipeline."""
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
    return df


BEHAVIOR = ["Administrative", "Administrative_Duration", "Informational",
            "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "SpecialDay", "Weekend",
            "total_pages", "total_duration", "duration_per_page", "exit_minus_bounce",
            "admin_share", "info_share", "product_share",
            "log_total_pages", "log_total_duration", "log_product_duration"] + CAT

FULL = BEHAVIOR + ["PageValues"]


def make_pipe(model, feats):
    num = [f for f in feats if f not in CAT]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def models():
    return {
        "LogisticRegression": LogisticRegression(C=1.0, class_weight="balanced",
                                                 solver="liblinear", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=80, max_depth=10,
                                               min_samples_leaf=10, max_features="sqrt",
                                               class_weight="balanced_subsample",
                                               random_state=SEED, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=30, colsample_bytree=0.9,
                                   reg_lambda=1.0, class_weight="balanced",
                                   random_state=SEED, verbose=-1),
    }


GRID = np.arange(0.02, 0.98 + 1e-9, 0.005)


def f1_over_grid(y, p):
    """F1 for the whole threshold grid at once, no loop - otherwise the bootstrap takes hours."""
    y = np.asarray(y).astype(int)
    pred = p[:, None] >= GRID[None, :]          # (n_sessions, n_thresholds)
    tp = (pred & (y[:, None] == 1)).sum(0)
    predicted_pos = pred.sum(0)
    actual_pos = y.sum()
    denom = predicted_pos + actual_pos
    return np.divide(2 * tp, denom, out=np.zeros(len(GRID)), where=denom > 0)


def best_threshold(y, p):
    """First threshold reaching maximum F1 - same convention as in the main script."""
    scores = f1_over_grid(y, p)
    k = int(np.argmax(scores))
    return float(GRID[k]), float(scores[k])


def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    return dict(precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0),
                f1=f1_score(y, yh, zero_division=0),
                roc_auc=roc_auc_score(y, p),
                ap=average_precision_score(y, p),
                brier=brier_score_loss(y, p),
                n=len(y), positives=int(y.sum()))


def split(df):
    tr = df[df.month_num <= 9]
    va = df[df.month_num == 10]
    te = df[df.month_num >= 11]
    return tr, va, te


# ------------------------------------------------------- 1. November / December
def nov_dec(df, out):
    tr, va, te = split(df)
    rows = []
    for setting, feats in [("full_no_month", FULL), ("behavior_no_month", BEHAVIOR)]:
        for name, m in models().items():
            pipe = make_pipe(m, feats).fit(tr[feats], tr.Revenue)
            thr, _ = best_threshold(va.Revenue, pipe.predict_proba(va[feats])[:, 1])
            for label, part in [("Nov+Dec", te), ("Nov", te[te.month_num == 11]),
                                ("Dec", te[te.month_num == 12])]:
                p = pipe.predict_proba(part[feats])[:, 1]
                rows.append(dict(setting=setting, model=name, period=label,
                                 threshold=thr, prevalence=part.Revenue.mean(),
                                 **metrics(part.Revenue.values, p, thr)))
    pd.DataFrame(rows).to_csv(out / "nov_dec_split.csv", index=False)
    print("[1] nov_dec_split.csv - table for section 4.3, backs 'Nov/Dec differ' with numbers")


# ------------------------------------------- 2. model selection instability
def selection_stability(df, out, n_boot=2000):
    """Resample October and see how often each model wins."""
    tr, va, te = split(df)
    rows = []
    for setting, feats in [("behavior_no_month", BEHAVIOR), ("full_no_month", FULL)]:
        probs, fitted = {}, {}
        for name, m in models().items():
            pipe = make_pipe(m, feats).fit(tr[feats], tr.Revenue)
            probs[name] = pipe.predict_proba(va[feats])[:, 1]
            fitted[name] = pipe
        y = va.Revenue.values
        rng = np.random.default_rng(SEED)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        wins = {k: 0 for k in probs}
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(pos, len(pos), True),
                                  rng.choice(neg, len(neg), True)])
            best, bf1 = None, -1.0
            for name, p in probs.items():
                f = float(f1_over_grid(y[idx], p[idx]).max())
                if f > bf1:
                    best, bf1 = name, f
            wins[best] += 1
        for name, w in wins.items():
            rows.append(dict(setting=setting, model=name,
                             win_share=w / n_boot,
                             point_val_f1=best_threshold(y, probs[name])[1]))
    pd.DataFrame(rows).to_csv(out / "selection_stability.csv", index=False)
    print("[2] selection_stability.csv - win share of each model on the October bootstrap")
    print("    if nobody exceeds 0.6, the selection is statistically insignificant - a separate finding")


# --------------------------------------------------------- 3. calibration
def calibration_fix(df, out):
    tr, va, te = split(df)
    rows = []
    for setting, feats in [("full_no_month", FULL), ("behavior_no_month", BEHAVIOR)]:
        base = make_pipe(LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31,
                                        min_child_samples=30, colsample_bytree=0.9,
                                        reg_lambda=1.0, class_weight="balanced",
                                        random_state=SEED, verbose=-1), feats)
        base.fit(tr[feats], tr.Revenue)
        thr, _ = best_threshold(va.Revenue, base.predict_proba(va[feats])[:, 1])
        rows.append(dict(setting=setting, calibration="none",
                         **metrics(te.Revenue.values, base.predict_proba(te[feats])[:, 1], thr)))
        # calibrate on October, training part untouched (model is frozen)
        for method in ["sigmoid", "isotonic"]:
            cal = (CalibratedClassifierCV(freeze(base), method=method)
                   if _FROZEN else CalibratedClassifierCV(base, method=method, cv="prefit"))
            cal.fit(va[feats], va.Revenue)
            pv = cal.predict_proba(va[feats])[:, 1]
            t2, _ = best_threshold(va.Revenue, pv)
            rows.append(dict(setting=setting, calibration=method,
                             **metrics(te.Revenue.values,
                                       cal.predict_proba(te[feats])[:, 1], t2)))
    pd.DataFrame(rows).to_csv(out / "calibration_fix.csv", index=False)
    print("[3] calibration_fix.csv - Brier before and after Platt/isotonic")
    print("    section 5.6 admits overconfidence but does not fix it; now there is a number")


# ------------------------------------------------------- 4. duplicates
def duplicates_sensitivity(df, out):
    src = [c for c in df.columns if c in FULL or c in ("Month", "Revenue")]
    dedup = df.drop_duplicates(subset=src, keep="first")
    rows = []
    for label, d in [("with_duplicates", df), ("without_duplicates", dedup)]:
        tr, va, te = split(d)
        for setting, feats in [("full_no_month", FULL), ("behavior_no_month", BEHAVIOR)]:
            pipe = make_pipe(models()["LightGBM"], feats).fit(tr[feats], tr.Revenue)
            thr, _ = best_threshold(va.Revenue, pipe.predict_proba(va[feats])[:, 1])
            rows.append(dict(variant=label, setting=setting, n_total=len(d), threshold=thr,
                             **metrics(te.Revenue.values,
                                       pipe.predict_proba(te[feats])[:, 1], thr)))
    pd.DataFrame(rows).to_csv(out / "duplicates_sensitivity.csv", index=False)
    print("[4] duplicates_sensitivity.csv - 125 repeats removed, metrics recomputed")


# ------------------------------ 5. MAIN: inflation under the literature-style protocol
def literature_inflation(df, out):
    """
    Run it 'as in the papers': random 70/30 split, PageValues included,
    SMOTE, threshold 0.5. Then the honest protocol. The gap is the contribution.
    """
    rows = []

    # (a) literature-style protocol
    X, y = df[FULL], df.Revenue
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        num = [f for f in FULL if f not in CAT]
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
        pipe = ImbPipeline([("pre", pre), ("sm", SMOTE(random_state=SEED)),
                            ("clf", RandomForestClassifier(n_estimators=200, random_state=SEED,
                                                           n_jobs=-1))])
        tag = "random_split + SMOTE + RF + PageValues (as in the literature)"
    except ImportError:
        pipe = make_pipe(RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1), FULL)
        tag = "random_split + RF + PageValues (SMOTE unavailable: pip install imbalanced-learn)"
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    rows.append(dict(protocol=tag, **metrics(yte.values, p, 0.5)))

    # (b) honest protocol
    tr, va, te = split(df)
    for setting, feats in [("out-of-time + PageValues (upper bound)", FULL),
                           ("out-of-time, without PageValues (primary result)", BEHAVIOR)]:
        pipe = make_pipe(models()["LightGBM"], feats).fit(tr[feats], tr.Revenue)
        thr, _ = best_threshold(va.Revenue, pipe.predict_proba(va[feats])[:, 1])
        rows.append(dict(protocol=setting,
                         **metrics(te.Revenue.values,
                                   pipe.predict_proba(te[feats])[:, 1], thr)))

    res = pd.DataFrame(rows)
    res.to_csv(out / "literature_inflation.csv", index=False)
    print("[5] literature_inflation.csv - MAIN table")
    print(res[["protocol", "f1", "roc_auc", "ap"]].to_string(index=False))
    print("    the gap between row 1 and row 3 is the scientific contribution")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="online_shoppers_intention.csv")
    ap.add_argument("--out", default="gaps_results")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = build(pd.read_csv(a.data))
    assert len(df) == 12330, f"expected 12330 rows, got {len(df)}"
    assert "PageValues" not in BEHAVIOR and "month_num" not in BEHAVIOR

    nov_dec(df, out)
    selection_stability(df, out)
    calibration_fix(df, out)
    duplicates_sensitivity(df, out)
    literature_inflation(df, out)

    (out / "run_meta.json").write_text(json.dumps(
        {"seed": SEED, "rows": len(df), "n_behavior_features": len(BEHAVIOR)}, indent=2))
    print(f"\nDone. Results in {out}/")


if __name__ == "__main__":
    main()
