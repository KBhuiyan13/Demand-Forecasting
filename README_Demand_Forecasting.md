# Demand Forecasting System

A time-series forecasting project using the M5 Forecasting Accuracy dataset. The current experiment compares a recursive LightGBM model with an optional zero-demand post-processing rule.

> **Core candidate:** LightGBM V2 without `rolling_std_5`  
> **Validation WAPE:** 28.247% baseline; 27.901% with zero-14  
> **Final test WAPE:** 29.297% baseline; 29.058% with zero-14

The optional rule gives a small aggregate improvement but suppresses some real demand. Keep it optional until inventory and service-level costs are evaluated.

## Scope

The reported experiment covers **14 representative series** and a 28-day forecast horizon, not the full M5 dataset.

- Training feature table: 25,606 rows
- Validation: 392 rows (14 series × 28 days)
- Test: 392 rows (14 series × 28 days)

## Results

### Validation

Actual total: **9,661 units**.

| Strategy | MAE | RMSE | WAPE | sMAPE | Predicted total | Total bias |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM V2 | 6.961 | 13.181 | 28.247% | 109.548% | 10,896.539 | +1,235.539 |
| V2 + zero-14 | 6.876 | 13.179 | 27.901% | 71.612% | 10,859.741 | +1,198.741 |

### Final test

Actual total: **9,421 units**.

| Strategy | MAE | RMSE | WAPE | sMAPE | Predicted total | Total bias |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM V2 | 7.041 | 13.817 | 29.297% | 110.647% | 10,855.965 | +1,434.965 |
| V2 + zero-14 | 6.984 | 13.817 | 29.058% | 76.599% | 10,826.073 | +1,405.073 |

Zero-14 improved test WAPE by **0.239 percentage points**, reduced MAE by 0.057, and reduced aggregate over-forecast bias by 29.892 units. RMSE was unchanged to displayed precision.

### Rolling-origin threshold comparison

Four origins were evaluated, with 14 series and a 28-day horizon per origin (1,568 predictions per strategy).

| Strategy | MAE | RMSE | WAPE | sMAPE | Predicted total | Bias |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.382 | 17.372 | 36.644% | 128.501% | 37,565.537 | +1,699.537 |
| Zero-14 | 7.907 | 17.345 | 34.569% | 61.694% | 36,771.008 | +905.008 |
| Zero-28 | 7.990 | 17.354 | 34.930% | 72.026% | 36,900.484 | +1,034.484 |
| Zero-42 | 8.021 | 17.356 | 35.068% | 78.434% | 36,951.302 | +1,085.302 |

Zero-14 beat baseline WAPE at all four origins, but the evidence is limited by the small number of series and origins.

## Methodology

### Feature engineering

The feature set includes lagged sales, rolling statistics, calendar features, event/SNAP indicators, and static product/store identifiers. The selected V2 model excludes `rolling_std_5`, which performed poorly in the earlier validation comparison.

### Recursive forecasting

Forecasts are generated one day at a time. Each new forecast is appended to the history used to build the next forecast's features. Actual future validation/test sales are used for scoring only, not for recursive feature construction.

### Model configuration

Model script: `src/models/lightgbm_v2_no_std5.py`.

- `n_estimators=500`, `learning_rate=0.03`, `num_leaves=31`, `max_depth=-1`
- `min_child_samples=20`, `subsample=0.9`, `subsample_freq=1`
- `colsample_bytree=0.9`, `reg_alpha=0.1`, `reg_lambda=0.1`
- `random_state=42`, `n_jobs=-1`
- Categorical features: `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`

### Optional zero-demand rule

At forecast origin, if the previous 14 available days sum to zero, the series' entire 28-day forecast is set to zero. Otherwise, normal recursive predictions are used. This is applied once per series, not recalculated every forecast day.

On final test, the rule activated for three series, all of which had some actual demand; **17 actual units** were suppressed. Business stockout costs must be considered before using this rule operationally.

## Metrics

- **MAE:** mean absolute error.
- **RMSE:** root mean squared error, more sensitive to large errors.
- **WAPE:** total absolute error divided by total absolute actual demand.
- **sMAPE:** symmetric mean absolute percentage error; can be difficult to interpret for intermittent demand.
- **Total bias:** predicted total minus actual total; positive means aggregate over-forecasting.

## Indicative repository layout

```text
Demand-Forecasting/
├── data/raw/
├── data/processed/features_v2/
├── data/processed/model_results/
├── notebooks/
├── reports/
├── src/preprocessing.py
├── src/models/
├── README.md
└── requirements.txt
```

Adjust this illustrative layout to match the files in your actual checkout.

## Reproduce evaluation

Run from the project root in PowerShell with the project virtual environment activated:

```powershell
python src\models\rolling_origin_thresholds.py
python src\models\validate_zero14.py
python src\models\evaluate_test_zero14.py
```

Outputs are written under `data/processed/model_results/`, including rolling-origin, validation, and test summaries, predictions, rule-activation details, and per-series metrics.

## Limitations and next steps

- This is a 14-series subset, not a full M5 benchmark.
- Four rolling origins provide limited evidence; evaluate more series and origins.
- Zero-14 has a small test gain but suppresses true demand.
- The final test has already been used for final evaluation. **Do not use it to select new thresholds or tune settings.** Use training-only rolling-origin evaluation and validation for further selection, and reserve a new untouched holdout for a future final assessment.
- Compare stockout costs with excess-inventory costs before adopting post-processing.
- Future work may include broader series coverage, additional model families, experiment tracking, API serving, monitoring, drift detection, and scheduled retraining.

## Status

Recursive LightGBM V2 candidate evaluated; optional zero-demand rule documented; validation and final test results recorded. Full-dataset evaluation, production deployment, and production monitoring are not claimed.
