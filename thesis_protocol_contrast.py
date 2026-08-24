"""
thesis_protocol_contrast.py — companion script for Section 4.3 of the thesis.

Measures how much of the performance reported on the Online Shoppers benchmark
comes from the evaluation protocol rather than from the data. The prior-work
machinery is held constant — stratified oversampling with SMOTE on the training
part only, an unconstrained 200-tree Random Forest, a fixed 0.5 threshold — and
two factors are varied one at a time:

    partition : random 70/30 over all months   vs   Feb-Sep fit / Nov-Dec test
    features  : full (with PageValues)         vs   behavior-only

The first cell reproduces the arrangement used in recent benchmark studies; the
row-to-row differences are the protocol contrast reported in Table 19.

Run:
    python thesis_protocol_contrast.py --data online_shoppers_intention.csv \
                                       --out protocol_contrast.csv

Requires imbalanced-learn in addition to the main pipeline's dependencies.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, confusion_matrix)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 42
THRESHOLD = 0.5                        # fixed, as in the prior-work protocol
MONTHS = {"Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jul": 7,
          "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

CAT = ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]


def build(df):
    """The same deterministic features as the main pipeline."""
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


def prior_work_pipe(feats):
    """Random Forest with SMOTE — the arrangement used in the cited studies.

    SMOTE sits inside the imbalanced-learn pipeline, so it resamples only during
    fit; the held-out sessions keep their natural class distribution.
    """
    num = [f for f in feats if f not in CAT]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    return ImbPipeline([("pre", pre),
                        ("sm", SMOTE(random_state=SEED)),
                        ("clf", RandomForestClassifier(n_estimators=200, random_state=SEED,
                                                       n_jobs=-1))])


def metrics(y, p, thr=THRESHOLD):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    prevalence = float(np.mean(y))
    ap = average_precision_score(y, p)
    return dict(n=len(y), positives=int(np.sum(y)), prevalence=prevalence,
                precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0),
                f1=f1_score(y, yh, zero_division=0),
                roc_auc=roc_auc_score(y, p),
                ap=ap, ap_lift=ap / prevalence,
                brier=brier_score_loss(y, p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def random_split(df, feats):
    Xtr, Xte, ytr, yte = train_test_split(df[feats], df.Revenue, test_size=0.3,
                                          stratify=df.Revenue, random_state=SEED)
    return Xtr, ytr, Xte, yte


def out_of_time_split(df, feats):
    tr, te = df[df.month_num <= 9], df[df.month_num >= 11]
    return tr[feats], tr.Revenue, te[feats], te.Revenue


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--data", default="online_shoppers_intention.csv")
    ap_.add_argument("--out", default="protocol_contrast.csv")
    a = ap_.parse_args()

    df = build(pd.read_csv(a.data))
    assert len(df) == 12330, f"expected 12330 rows, got {len(df)}"
    assert "PageValues" not in BEHAVIOR and "month_num" not in FULL

    rows = []
    for fname, feats in [("full (with PageValues)", FULL), ("behavior-only", BEHAVIOR)]:
        for sname, splitter in [("random 70/30", random_split),
                                ("out-of-time Nov-Dec", out_of_time_split)]:
            Xtr, ytr, Xte, yte = splitter(df, feats)
            pipe = prior_work_pipe(feats).fit(Xtr, ytr)
            rows.append(dict(partition=sname, features=fname, threshold=THRESHOLD,
                             **metrics(np.asarray(yte), pipe.predict_proba(Xte)[:, 1])))

    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)

    pd.set_option("display.width", 200)
    print(res[["partition", "features", "n", "prevalence", "precision", "recall",
               "f1", "roc_auc", "ap", "ap_lift", "brier"]].round(4).to_string(index=False))

    def cell(part, feat, m):
        return float(res[(res.partition == part) & (res.features == feat)][m].iloc[0])

    lit = "random 70/30", "full (with PageValues)"
    print(f"\nprior-work protocol           : F1 {cell(*lit,'f1'):.3f}  "
          f"ROC-AUC {cell(*lit,'roc_auc'):.3f}  AP {cell(*lit,'ap'):.3f}")
    print(f"partition effect (ROC-AUC)    : "
          f"{cell(*lit,'roc_auc') - cell('out-of-time Nov-Dec','full (with PageValues)','roc_auc'):+.3f}")
    print(f"PageValues effect (ROC-AUC)   : "
          f"{cell(*lit,'roc_auc') - cell('random 70/30','behavior-only','roc_auc'):+.3f}")
    print(f"\nWritten to {a.out}")


if __name__ == "__main__":
    main()
