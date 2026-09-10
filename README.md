# Momentum and Machine Learning Research

Python notebooks and helper modules for exploring **financial time series, technical indicators, and classification models**. This is a research workspace rather than a deployable trading system.

## Repository map

| Path | Purpose |
| --- | --- |
| [momentum.ipynb](momentum.ipynb) | Momentum and modeling experiments. |
| [stock_analysis.ipynb](stock_analysis.ipynb) | Stock-series exploration. |
| [process_data.py](process_data.py) | Indicator signals, targets, and feature preparation. |
| [ml.py](ml.py) | Model factories and walk-forward experimentation. |
| [utils.py](utils.py) | Data loading, downloads, stationarity, and Hurst helpers. |
| [assets](assets) | CSV inputs managed with Git LFS. |
| [backtest.py](backtest.py) | Empty placeholder; no backtest engine is implemented here. |

## Prepare the data

After cloning, install Git LFS and retrieve the actual CSV objects:

```sh
git lfs install
git lfs pull
```

Run notebooks from the repository root so relative paths such as `assets/stock_info.csv`, `assets/stocks.csv`, and `assets/selected_stocks.csv` resolve. A small text file beginning with a Git LFS specification URL is a pointer, not the dataset.

## Python environment

The repository does not include a pinned environment. Imports reference NumPy, pandas, Matplotlib, seaborn, scikit-learn, statsmodels, Keras, TA-Lib, hurst, and yfinance. Install these in an isolated Python environment along with Jupyter; resolve the Keras backend and TA-Lib platform requirements before running the notebooks.

```sh
python -m pip install jupyterlab
python -m jupyterlab
```

Review notebook cells before running them: utilities can download market data and save CSV files. Package/API differences may require code changes, so these steps are an environment outline rather than a verified reproducible install.

## Research workflow and limits

1. Load and inspect the available time series.
2. Explore signals and target construction in `process_data.py`.
3. Inspect model choices in `ml.py`, including random forests, SVC, logistic regression, and a Keras network.
4. Review time ordering and the walk-forward implementation before interpreting validation output.

The repository has no automated test suite, pinned data snapshot, or verified performance report. No model training, live downloads, or trading operations were run for this documentation update. Notebook output is experimental and does not establish a reliable trading strategy.
