# momentum-trading-ml

Experimentos de análisis de series financieras, momentum, aprendizaje automático y backtesting. Combina notebooks con módulos de preparación de datos y modelos.

## Estructura

- [assets](assets)
- [backtest.py](backtest.py)
- [ml.py](ml.py)
- [process_data.py](process_data.py)
- [utils.py](utils.py)

## Preparación y uso

Los CSV utilizan Git LFS: instala Git LFS y ejecuta `git lfs pull` para recuperar los datos. Los notebooks son `momentum.ipynb` y `stock_analysis.ipynb`. Las importaciones incluyen pandas, NumPy, matplotlib, scikit-learn, statsmodels, Keras, TA-Lib, hurst y yfinance. No se incluye una lista de versiones reproducible; TA-Lib puede requerir componentes nativos.

## Validación y estado

Esta guía se contrastó con el árbol de archivos y los manifiestos del repositorio. No se ha validado una ejecución completa contra servicios externos, bases de datos o hardware. Las versiones y los scripts mostrados describen el código actual; no implican que sus dependencias antiguas sigan siendo compatibles.
