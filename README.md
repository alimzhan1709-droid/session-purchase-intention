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
| Full LightGBM (PageValues retained) | 0.607 | 0.834 | 0.661 | 0.207 |
| Single-rule baseline `PageValues > 0` | 0.659 | 0.807 | 0.577 | 0.207 |
| Behavior-only LightGBM (primary) | 0.421 | 0.693 | 0.341 | 0.207 |
| Prior-work-style protocol (random split, SMOTE, 0.5) | 0.643 | 0.894 | 0.653 | 0.155 |

- The full model is significantly **outperformed on threshold F1** by the single-rule baseline `PageValues > 0` (paired difference -0.0527, 95% bootstrap CI [-0.0703, -0.0359]).
- Once `PageValues` is withheld, AP falls to 0.341 and ROC-AUC to 0.693, and the three algorithm families become **statistically indistinguishable** (test F1 0.414-0.425, inside the bootstrap interval of the selected model [0.406, 0.436]). In the primary setting the selected family is not even the strongest on the frozen period: October picks LightGBM on validation F1, while Random Forest attains the higher test F1 of 0.425.
- The winning family is **unstable across pre-declared feature-timing settings** - the same protocol selects LightGBM under no-month features and Logistic Regression under both cyclical-month settings and in the engagement-only ablation, all from the same 549-session validation month of 115 purchases.
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
this one was rebuilt from the recorded execution algorithm. The reported numbers were then
regenerated from it inside the declared reference environment, so the study text and this code now
agree by construction rather than by comparison.

**Reproduced exactly from the original tables** (to three decimals): partition sizes and
prevalences (7,056 / 549 / 4,725 at 11.58% / 20.95% / 20.66%), row and class counts, the 125 exact
duplicate rows, every baseline (`PageValues > 0` at P 0.617 / R 0.708 / F1 0.659 / ROC-AUC 0.807 /
AP 0.577 with 691 true and 429 false positives; the October-tuned cut-point 6.887;
prevalence-constant Brier 0.172; majority-class accuracy 0.7934), and all five Logistic Regression
rows of the candidate table.

**Not recovered: the original tree-model configuration.** The published Random Forest and LightGBM
results could not be reproduced exactly, and three explanations were ruled out in turn:

- *Library version.* The declared environment was rebuilt exactly and re-run. It selects the same
  thresholds as scikit-learn 1.6.1, so the version was not the cause.
- *Hyperparameters.* A 256-point LightGBM sweep and a 96-point Random Forest sweep over plausible
  settings failed to reach the published values. The closest LightGBM candidate matched two figures
  exactly but moved the upper-bound setting further away, which is the signature of fitting to a few
  targets rather than recovering an original.
- *Data, features, splits, threshold rule, metrics.* All exact, as the Logistic Regression and
  baseline rows above demonstrate.

The configuration documented in the surviving add-on script was therefore kept rather than tuned
toward the target numbers, and the study was recomputed from it. Across every candidate row the
reconstruction and the original agree to within 0.009 on ranking metrics; what moves is the October
threshold argmax and everything downstream of it. October carries 115 purchases and a nearly flat F1
surface near the optimum, so the operating point is far less sharply determined than its decimals
suggest - the selection instability the study reports as a finding, met from the other direction.

**macOS note.** The LightGBM wheel links `@rpath/libomp.dylib` without shipping it, and its only
rpath entries point at Homebrew locations. Installing `libomp` (`brew install libomp`) is the
supported fix.

## Reference environment

Python 3.13.5, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, LightGBM 4.6.0, SciPy 1.17.0,
Matplotlib 3.10.8, imbalanced-learn 0.12.4. The reported results were produced in exactly this
environment; `requirements.txt` pins it.

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
