# Session-Level Purchase-Intention Prediction

**Target-Adjacent Features and Temporal Validation in Session-Level Purchase-Intention Prediction: How Much of a Benchmark Result Survives?**

## Authors

- Arsen S. Alimzhan - School of Information Technology and Engineering, Kazakh-British Technical University, Almaty, Kazakhstan - ar_alimzhan@kbtu.kz - ORCID [0009-0008-7229-0533](https://orcid.org/0009-0008-7229-0533)
- Irina M. Ualiyeva - Department of Computer Science, Al-Farabi Kazakh National University, Almaty, Kazakhstan - ualiyeva.irina@kaznu.kz - ORCID [0000-0003-3853-8896](https://orcid.org/0000-0003-3853-8896)

## Abstract

This study asks how much of the strong performance commonly reported for the UCI Online Shoppers Purchasing Intention benchmark reflects genuine behavioral signal rather than a single analytics-derived, target-adjacent feature. Using 12,330 anonymized sessions (1,908 purchases, 15.47%), we fit Logistic Regression, Random Forest, and LightGBM under a chronological, out-of-time protocol: February-September for training, October for model and threshold selection, and the frozen pipeline evaluated once on November-December. Two pre-declared feature settings isolate the effect of `PageValues`, an analytics-platform-computed conversion-propensity score: a **full** setting that includes it as an upper bound, and a primary **behavior-only** setting that excludes it together with the raw month indicator.

## Key results

| Setting | F1 | ROC-AUC | AP | Test prevalence |
|---|---|---|---|---|
| Full LightGBM (PageValues retained) | 0.609 | 0.835 | 0.660 | 0.207 |
| Single-rule baseline `PageValues > 0` | 0.659 | 0.807 | 0.577 | 0.207 |
| Behavior-only LightGBM (primary) | 0.422 | 0.695 | 0.340 | 0.207 |
| Prior-work-style protocol (random split, SMOTE, 0.5) | 0.650 | 0.895 | 0.654 | 0.155 |

- The full model is significantly **outperformed on threshold F1** by the single-rule baseline `PageValues > 0` (paired difference -0.0507, 95% bootstrap CI [-0.0671, -0.0339]).
- Once `PageValues` is withheld, AP falls to 0.340 and ROC-AUC to 0.695, and the three algorithm families become **statistically indistinguishable** (test F1 0.411-0.422, inside the bootstrap interval of the selected model [0.408, 0.433]).
- The winning family is **unstable across pre-declared feature-timing settings** - LightGBM, Random Forest, and Logistic Regression are each selected in turn from the same 549-session validation month.
- Random splitting inflates separability. Holding the prior-work machinery constant and changing one factor at a time attributes about **0.058** of ROC-AUC to the random partition and about **0.156** to retaining `PageValues`, so the admitted feature matters roughly three times as much as the split.
- Average precision is not comparable across the two designs at face value: a random split leaves test prevalence at 0.155 while the frozen period carries 0.207. Against its own no-skill baseline the prior-work protocol reaches 4.2x prevalence and the frozen full model 3.2x.

## Interpretation

The field's typical protocol on this benchmark measures an analytics artifact rather than the predictability of ordinary browsing behavior. We propose **target-adjacent-feature control with frozen out-of-time evaluation** as a minimal reporting standard for this dataset.

## Method

- **Data:** UCI Online Shoppers Purchasing Intention Dataset - 12,330 anonymized sessions, 15.47% positive class.
- **Split:** chronological - train Feb-Sep (7,056 sessions, 11.58% purchase rate), validate Oct (549 sessions, 115 purchases), test Nov-Dec once (4,725 sessions, 20.66%), no refitting.
- **Feature settings:** five, declared in advance - behavior-only (primary), behavior + cyclical month, engagement-only, full (adds `PageValues`, upper bound), full + cyclical month. Raw numeric month is never a predictor, since it would encode the partition boundary.
- **Models:** Logistic Regression, Random Forest, LightGBM with class-weighted fitting.
- **Analysis:** stratified bootstrap intervals, paired comparison, calibration diagnostics, grouped permutation importance, error profiling.
- `SEED = 42` throughout; preprocessing is fitted inside each pipeline on the training months only.

## Repository contents

| File | Purpose |
|---|---|
| `uci_online_shoppers_kbtu_pipeline.py` | Main pipeline. Validates the input, builds features, fits all candidates, selects on October, evaluates the frozen pipelines once, writes tables, diagnostics and 19 figures. |
| `thesis_protocol_contrast.py` | The 2x2 protocol contrast: {random 70/30, out-of-time} x {full, behavior-only}, with the prior-work machinery held constant. |
| `thesis_gaps_addon.py` | Supplementary analyses: November/December split, selection stability under bootstrap, Platt/isotonic calibration, duplicate sensitivity, and the prior-work inflation run. |
| `fetch_data.py` | Downloads the benchmark from UCI and verifies its SHA-256. |

## Running

```bash
pip install -r requirements.txt
python fetch_data.py
python uci_online_shoppers_kbtu_pipeline.py --data online_shoppers_intention.csv \
                                            --output uci_corrected_results \
                                            --figures thesis_corrected_figures
python thesis_protocol_contrast.py --data online_shoppers_intention.csv --out protocol_contrast.csv
python thesis_gaps_addon.py --data online_shoppers_intention.csv --out gaps_results
```

The dataset is not committed. It is public, it downloads in seconds, and pinning it by digest is
more useful than storing a copy: `fetch_data.py` refuses to proceed unless the upstream file is
byte-identical to the one used for the analysis (`b3055ee3...f358e0d9`, 12,330 rows, 1,908
purchases, 125 exact duplicate rows).

## Reproduction status

`uci_online_shoppers_kbtu_pipeline.py` is a **reconstruction**. The original script was lost, and
this one was rebuilt from the recorded execution algorithm and validated against the published
tables. The whole pipeline was executed on Python 3.9.6 / scikit-learn 1.6.1 / LightGBM 4.6.0,
which is the published LightGBM but not the published scikit-learn.

**Reproduced exactly** (to three decimals):

- Partition sizes and prevalences: 7,056 / 549 / 4,725 at 11.58% / 20.95% / 20.66%.
- Row and class counts, and the 125 exact duplicate rows.
- Every baseline: `PageValues > 0` at P 0.617 / R 0.708 / F1 0.659 / ROC-AUC 0.807 / AP 0.577 with
  691 true and 429 false positives; the October-tuned cut-point 6.887; prevalence-constant Brier
  0.172; majority-class accuracy 0.7934.
- All five Logistic Regression rows of the candidate table, and the engagement-only LightGBM row.

**Reproduced within 0.009**, which is far inside the reported bootstrap intervals:

- Ranking metrics for every one of the fourteen candidate rows. The largest deviation in ROC-AUC or
  AP anywhere in the table is 0.009; LightGBM rows land within 0.006.
- Bootstrap intervals: full LightGBM F1 0.607 [0.582, 0.631] against 0.609 [0.583, 0.632];
  behavior-only 0.421 [0.406, 0.436] against 0.422 [0.408, 0.433]; the `PageValues` row is exact.
- The paired contrast and its conclusion: F1 difference -0.0527 [-0.0703, -0.0359] against
  -0.0507 [-0.0671, -0.0339], with 0% of resamples positive; ROC-AUC +0.0274 against +0.0281 and
  AP +0.0846 against +0.0836, both 100% positive. The rejection of the F1 hypothesis is unchanged.
- Brier scores: 0.120 / 0.205 / 0.172 / 0.207 against 0.121 / 0.204 / 0.172 / 0.207.
- Grouped permutation importance: `PageValues` 0.432 against 0.424, `ExitRates` 0.023 against
  0.022, `BounceRates` 0.017 exactly; behavior-only `ExitRates` 0.045 against 0.043, with
  `duration_per_page` and `OperatingSystems` exact. The dominance ordering is unchanged.

**Does not reproduce exactly: everything downstream of the October threshold.** The selection
argmax moves under a different scikit-learn version - for behavior-only LightGBM from 0.305 to
0.400, for full Random Forest from 0.550 to 0.555 - and the confusion matrices move with it. The
full model stays close (3,448 / 301 / 420 / 556 against 3,422 / 327 / 406 / 570), the behavior-only
model does not (2,105 / 1,644 / 278 / 698 against 1,737 / 2,012 / 178 / 798). The error-group
profiles inherit the same shift, though within-group medians agree closely where the groups
overlap: full-model false-positive and false-negative median product durations match at 1,467.7
and 2,356.9 seconds.

This is not a discrepancy in the analysis so much as a measurement of it. October carries 549
sessions and 115 purchases, its F1 surface is flat near the optimum, and a small change in tree
construction between library versions is enough to move the selected point. The study reports this
selection instability as a finding; the reconstruction encounters it from the other direction.

Pin to the reference environment below to reproduce the published operating points.

Note for macOS: LightGBM's wheel links `@rpath/libomp.dylib` but does not ship it, and its only
rpath entries point at Homebrew locations. Installing `libomp` (`brew install libomp`) is the
supported fix.

## Reference environment

Python 3.13.5, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, LightGBM 4.6.0, SciPy 1.17.0,
Matplotlib 3.10.8, imbalanced-learn 0.12.4.

## Data

C. O. Sakar, S. O. Polat, M. Katircioglu and Y. Kastro, "Real-time prediction of online shoppers
purchasing intention using multilayer perceptron and LSTM recurrent neural networks," *Neural
Computing and Applications*, vol. 31, no. 10, pp. 6893-6908, 2019.
Dataset: UCI Machine Learning Repository, [doi:10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).

## Limitations

One website, one incomplete year, month-level rather than event-level time, model and threshold
selected on a single validation month of 115 purchases, `PageValues` and completed-session
aggregates that would not be available at a live scoring checkpoint, no product or cart variables,
and no fitted calibration layer. These results study evaluation design on a public benchmark; they
are not a deployable conversion model.

## Keywords

e-commerce; purchase-intention prediction; target leakage; temporal validation; class imbalance; probability calibration

## Status

Manuscript in preparation for journal submission.

## License

MIT - see `LICENSE`.
