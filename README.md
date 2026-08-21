# Session-Level Purchase-Intention Prediction

**Target-Adjacent Features and Temporal Validation in Session-Level Purchase-Intention Prediction: How Much of a Benchmark Result Survives?**

## Authors

- Arsen S. Alimzhan - School of Information Technology and Engineering, Kazakh-British Technical University, Almaty, Kazakhstan - ar_alimzhan@kbtu.kz - ORCID [0009-0008-7229-0533](https://orcid.org/0009-0008-7229-0533)
- Irina M. Ualiyeva - Department of Computer Science, Al-Farabi Kazakh National University, Almaty, Kazakhstan - ualiyeva.irina@kaznu.kz - ORCID [0000-0003-3853-8896](https://orcid.org/0000-0003-3853-8896)

## Abstract

This study asks how much of the strong performance commonly reported for the UCI Online Shoppers Purchasing Intention benchmark reflects genuine behavioral signal rather than a single analytics-derived, target-adjacent feature. Using 12,330 anonymized sessions (1,908 purchases, 15.47%), we fit Logistic Regression, Random Forest, and LightGBM under a chronological, out-of-time protocol: February-September for training, October for model and threshold selection, and the frozen pipeline evaluated once on November-December. Two pre-declared feature settings isolate the effect of `PageValues`, an analytics-platform-computed conversion-propensity score: a **full** setting that includes it as an upper bound, and a primary **behavior-only** setting that excludes it together with the raw month indicator.

## Key results

| Setting | F1 | ROC-AUC | AP |
|---|---|---|---|
| Full LightGBM (PageValues retained) | 0.609 | 0.835 | 0.660 |
| Single-rule baseline `PageValues > 0` | 0.659 | - | - |
| Behavior-only LightGBM (primary) | 0.422 | 0.695 | 0.340 |
| Prior-work-style protocol (random split, SMOTE, 0.5) | - | 0.895 | 0.653 |

- The full model is significantly **outperformed on threshold F1** by the single-rule baseline `PageValues > 0` (paired difference -0.0507, 95% bootstrap CI [-0.0671, -0.0339]).
- Once `PageValues` is withheld, AP falls to 0.340 and ROC-AUC to 0.695, and the three algorithm families become **statistically indistinguishable** (test F1 0.411-0.422, inside the bootstrap interval of the selected model [0.408, 0.433]).
- The winning family is **unstable across pre-declared feature-timing settings** - LightGBM, Random Forest, and Logistic Regression are each selected in turn from the same 549-session validation month.
- Random splitting inflates separability; withholding `PageValues` accounts for the loss in precision-recall area.

## Interpretation

The field's typical protocol on this benchmark measures an analytics artifact rather than the predictability of ordinary browsing behavior. We propose **target-adjacent-feature control with frozen out-of-time evaluation** as a minimal reporting standard for this dataset.

## Method

- **Data:** UCI Online Shoppers Purchasing Intention Dataset - 12,330 anonymized sessions, 15.47% positive class.
- **Split:** chronological - train Feb-Sep, validate Oct (549 sessions), test Nov-Dec once, no refitting.
- **Models:** Logistic Regression, Random Forest, LightGBM with class-weighted fitting.
- **Analysis:** stratified bootstrap intervals, paired comparison, calibration diagnostics, grouped permutation importance, error profiling.

## Keywords

e-commerce; purchase-intention prediction; target leakage; temporal validation; class imbalance; probability calibration

## Status

Manuscript in preparation for journal submission.
