# Freight Rate Prediction Challenge

**Author:** Musab Alhusamiah

Predicts the posted rate of freight loads. Final model: a blend of a shallow **LightGBM** (Huber loss) and a **spline + Ridge** model with a linear time-trend term, both predicting **log(rate per mile)**.

| Rolling-origin validation (3 future windows, mean) | MAE | MAPE | median APE |
|---|---|---|---|
| Baseline: median $/mile by equipment x distance band | $159 | 6.9% | 4.2% |
| **Final blend** | **$98** | **4.3%** | **1.5%** |

Full methodology, data-quality handling, split strategy and the December chart: [`reports/Freight_Rate_Report.pdf`](reports/Freight_Rate_Report.pdf).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # optional
python -m pip install -r requirements.txt

# 1) put the assessment files into data/ (renames them to the names used by the code)
python -m src.prepare_data path/to/machineLearningJOBassesmint.zip     # or the extracted folder

# 2) run everything (about 90 seconds): EDA -> CV -> final fit -> predictions -> scorer -> figures -> report
make all DATA=path/to/machineLearningJOBassesmint.zip
```

Or step by step:

```bash
python -m src.run_all          # validation, final model, validation_predictions.csv, outputs/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv
python -m src.make_figures && python -m src.make_report
```

Note: the completed December file is written to `outputs/december_chart_inputs.csv` (the original input in `data/` is left untouched), so pass that path to `score.py` instead of the `data/december_chart_inputs.csv` path shown in the assessment README.

Outputs: `validation_predictions.csv` (12,000 rows, `load_id,predicted_rate`), `outputs/december_chart_inputs.csv`, `scorer_results/candidate_december.png`, `reports/*.json|csv|pdf`.
Runs are deterministic (fixed seed 42); outputs were verified identical across independent runs.

## Repository layout

```
src/features.py      cleaning + feature engineering (FeatureBuilder: fit on train only, transform everywhere)
src/models.py        feature assembly, the 3 models, blend, metrics
src/run_all.py       rolling CV, last-window evaluation, final fit, predictions, December inputs
src/eda.py           exploratory data analysis: prints every number quoted in the report (log: reports/eda_log.txt)
src/make_figures.py  report figures and reports/results_summary.json
src/make_report.py   builds the PDF report
src/prepare_data.py  copies/renames the assessment files into data/
experiments/         scripts behind each modelling decision (see experiments/README.md)
score.py             provided scorer (unchanged)
```

## Approach in brief

- **Target:** log(rate / distance); prediction = exp(output) x distance.
- **Cleaning:** negative weights -> absolute value (sign errors); weight capped at 47,500 -> flag; missing weight -> median per equipment (train only) + flag; missing market_index -> NaN for LightGBM / median + flag for Ridge; price spikes (0.7% with rate/mile > $5) are kept and handled with Huber loss.
- **Validation:** expanding-window, rolling-origin CV over three future windows (May-Jun, Jul-Aug, Sep-Oct). No random splits; imputers/encoders fit on training folds only.
- **Findings that shaped the model:** calendar/holiday features hurt (removed); lane target-encoding gave no gain (12% of validation lanes are unseen); same-day market means help; tree models drift low in later months, fixed by a linear trend term in the Ridge component.
- **December chart:** the December input file has no `market_index`/`quote_signal`, so each date uses the average of those features over that date's loads in `validation.csv` (features only, no labels).

## Honest notes / limitations

- Sep-Oct was meant to be an untouched hold-out, but it revealed the temporal drift and guided the fix, so it is reported as the last rolling-validation fold, not an independent test.
- The linear trend is an assumption when extrapolating to Nov-Dec; the held-out metrics of the grader may differ.
- The 0.7% price spikes keep RMSE high (~630) for every model; MAE/MAPE/median APE are the informative metrics.
