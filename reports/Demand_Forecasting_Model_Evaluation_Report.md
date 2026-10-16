# Model Evaluation Report — Demand Forecasting System

**Dataset:** M5 Forecasting Accuracy  
**Scope:** 14 representative series  
**Forecast horizon:** 28 days  
**Core candidate:** LightGBM V2 without `rolling_std_5`  
**Primary aggregate metric:** WAPE

## 1. Executive summary

The current core candidate is a recursive LightGBM V2 model without `rolling_std_5`. Its validation WAPE is **28.247%**, better than the previously recorded 7-day moving-average baseline (31.572%) and LightGBM V1 strict-recursive validation result (30.908%).

An optional rule sets the entire 28-day forecast to zero when the previous 14 available days of a series sum to zero. It improved aggregate WAPE in the limited rolling-origin comparison, validation, and final test. On final test, WAPE decreased from **29.297% to 29.058%** and MAE decreased from **7.041 to 6.984**. RMSE was unchanged to displayed precision.

However, the rule suppressed **17 units of real test demand** across three activated series. Keep LightGBM V2 without the rule as the core candidate and treat zero-14 as an optional policy whose suitability depends on inventory and service-level costs.

## 2. Experimental setup

- Training features: 25,606 rows.
- Validation: 392 rows (14 series × 28 days).
- Test: 392 rows (14 series × 28 days).
- Forecast horizon: 28 days.
- Forecast method: recursive one-step-ahead predictions, appending predictions to history for subsequent steps.
- Actual future sales are used for scoring only, not recursive feature construction.
- The test period has been evaluated and must not be reused for further threshold selection.

These are results for a representative subset, not full-dataset M5 benchmark results.

### Model configuration

| Parameter | Value |
|---|---:|
| `n_estimators` | 500 |
| `learning_rate` | 0.03 |
| `num_leaves` | 31 |
| `max_depth` | -1 |
| `min_child_samples` | 20 |
| `subsample` | 0.9 |
| `subsample_freq` | 1 |
| `colsample_bytree` | 0.9 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 0.1 |
| `random_state` | 42 |
| `n_jobs` | -1 |

Categorical features: `item_id`, `dept_id`, `cat_id`, `store_id`, and `state_id`.

## 3. Candidate-model history

| Model/strategy | Evaluation | WAPE |
|---|---|---:|
| Moving average, 7-day | Earlier baseline evaluation | 31.572% |
| LightGBM V1, strict recursive | Validation | 30.908% |
| LightGBM V2 with `rolling_std_5` | Validation | 45.250% |
| LightGBM V2 without `rolling_std_5` | Validation | 28.247% |

The recorded comparisons support using the no-`rolling_std_5` V2 model as the current core candidate.

## 4. Rolling-origin zero-demand threshold experiment

Four origins were evaluated, each forecasting 28 days for 14 series. Each strategy produced 1,568 predictions across the four origins.

| Strategy | MAE | RMSE | WAPE | sMAPE | Predicted total | Bias |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.382 | 17.372 | 36.644% | 128.501% | 37,565.537 | +1,699.537 |
| Zero-14 | 7.907 | 17.345 | 34.569% | 61.694% | 36,771.008 | +905.008 |
| Zero-28 | 7.990 | 17.354 | 34.930% | 72.026% | 36,900.484 | +1,034.484 |
| Zero-42 | 8.021 | 17.356 | 35.068% | 78.434% | 36,951.302 | +1,085.302 |

WAPE by origin:

| Forecast origin | Baseline WAPE | Zero-14 WAPE |
|---|---:|---:|
| 2015-12-06 | 55.580% | 53.730% |
| 2016-01-03 | 32.915% | 30.395% |
| 2016-01-31 | 30.781% | 27.514% |
| 2016-02-28 | 31.512% | 30.779% |

Zero-14 beat baseline at all four origins. Its activation counts were 5, 4, 6, and 6 series, respectively. This is promising but limited evidence given the small subset and number of origins.

## 5. Validation results

Actual validation total: **9,661 units**.

| Metric | Baseline LightGBM V2 | V2 + zero-14 |
|---|---:|---:|
| Rows | 392 | 392 |
| Predicted total | 10,896.539 | 10,859.741 |
| Total bias | +1,235.539 | +1,198.741 |
| MAE | 6.961 | 6.876 |
| RMSE | 13.181 | 13.179 |
| WAPE | 28.247% | 27.901% |
| sMAPE | 109.548% | 71.612% |

The rule activated for three of 14 series:

| Series | Actual total | Baseline forecast | Zero-14 forecast | Interpretation |
|---|---:|---:|---:|---|
| `HOBBIES_2_146_CA_3` | 9 | 10.651 | 0 | Suppressed 9 actual units |
| `FOODS_3_778_CA_2` | 0 | 17.397 | 0 | Removed a false-positive forecast |
| `HOUSEHOLD_1_336_CA_3` | 0 | 8.750 | 0 | Removed a false-positive forecast |

Validation WAPE improved by 0.346 percentage points and aggregate positive bias decreased by 36.798 units.

## 6. Final test results

Actual test total: **9,421 units**.

| Metric | Baseline LightGBM V2 | V2 + zero-14 |
|---|---:|---:|
| Rows | 392 | 392 |
| Predicted total | 10,855.965 | 10,826.073 |
| Total bias | +1,434.965 | +1,405.073 |
| MAE | 7.041 | 6.984 |
| RMSE | 13.817 | 13.817 |
| WAPE | 29.297% | 29.058% |
| sMAPE | 110.647% | 76.599% |

### Change from baseline

- MAE decreased by **0.057**.
- WAPE decreased by **0.239 percentage points**.
- Aggregate bias decreased by **29.892 units**.
- RMSE was unchanged to displayed precision.

The rule activated for three series, all of which had some real demand during the test horizon:

| Series | Actual total | Baseline forecast | Zero-14 forecast | Actual units suppressed |
|---|---:|---:|---:|---:|
| `HOBBIES_2_146_CA_3` | 8 | 10.829 | 0 | 8 |
| `HOUSEHOLD_1_336_CA_3` | 6 | 8.577 | 0 | 6 |
| `FOODS_3_778_CA_2` | 3 | 10.486 | 0 | 3 |
| **Total** | **17** | **29.892** | **0** | **17** |

Although zero-14 reduced absolute error across these activated series overall, it eliminated 17 units of actual demand. Stockout and missed-demand costs must be considered in an operational setting.

## 7. Metric definitions

- **MAE:** mean absolute error.
- **RMSE:** square root of mean squared error.
- **WAPE:** sum of absolute errors divided by sum of absolute actual demand, expressed as a percentage.
- **sMAPE:** symmetric mean absolute percentage error; can be volatile for sparse/intermittent demand.
- **Total bias:** predicted total minus actual total; positive values indicate aggregate over-forecasting.

WAPE is the primary aggregate metric, supplemented by MAE, RMSE, bias, and per-series results.

## 8. Recommendations

1. **Keep LightGBM V2 without `rolling_std_5` as the core candidate.** It has the strongest recorded validation WAPE among the compared versions.
2. **Keep zero-14 optional.** It provides a small aggregate gain but suppresses true demand.
3. **Do not tune further using the final test.** Use training-only rolling-origin evaluation and validation for selection, and reserve a new untouched holdout for a future final assessment.
4. **Expand evaluation.** Include more series and origins, and analyze performance by forecast horizon and demand volume.
5. **Measure business cost.** Compare stockout/under-forecast costs with excess-inventory/over-forecast costs before enabling the rule.
6. **Be precise about scope.** These results do not establish full-M5 benchmark performance or production readiness.

## 9. Result artifacts

Evaluation scripts save outputs under `data/processed/model_results/`, including rolling-origin, validation, and test summaries, row-level predictions, rule-activation details, and per-series metrics. Retain these CSVs alongside this report for reproducibility and auditing.
