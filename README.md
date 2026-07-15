---
title: ARIMA Informed LSTM
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# ARIMA-Informed LSTM

A web app that forecasts a time series by combining a classical **ARIMA** model with an
**LSTM** neural network, where the LSTM's training loss is additionally informed by the
ARIMA forecast (weighted by a tunable λ).

## Navigation

- **Home** — overview of the tool and workflow
- **Model**
  1. **Data** — upload a CSV/Excel file, pick the time column and the study variable
  2. **Summary Statistics** — mean, median, std, variance, min, max, skewness, kurtosis,
     plus a line plot and a boxplot
  3. **ARIMA** — auto order selection (`auto_arima`) or manual `p, d, q`; adjustable
     train/test split (default 0.80)
  4. **LSTM Hyperparameters** — manual entry, or auto-tuning via **Grid Search** or
     **Bayesian Optimization** (Gaussian Process)
  5. **ARIMA-Informed LSTM** — trains the standard LSTM (λ=0) plus one model per λ value
     you list, reports RMSE / MAE / MAPE for each, plots predictions vs. actuals, and
     produces a downloadable Excel workbook
- **Instructions** — usage guide (expand as needed)
- **Developers** — credits / contact (fill in your details)

## Notes

- Data must contain a numeric study-variable column; non-numeric rows are dropped.
- For small datasets (fewer than ~30 points) ARIMA/LSTM results are less reliable —
  more history generally gives better forecasts.
- Bayesian Search uses `scikit-optimize`'s `gp_minimize` — a Gaussian Process surrogate
  model with expected-improvement acquisition — to choose each hyperparameter
  evaluation based on prior results, rather than sampling at random.
