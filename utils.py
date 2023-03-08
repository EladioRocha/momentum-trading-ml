import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import datetime
import os
from hurst import compute_Hc
from typing import List

def get_stocks(tickers: List[str], start_date: str = '2000-01-01', 
               end_date: str = datetime.datetime.today().strftime('%Y-%m-%d')) -> pd.DataFrame:
    """
    Download historical stock data for the specified tickers.

    Parameters:
    ----------
    tickers : List[str]
        A list of ticker symbols to download data for.
    start_date : str, optional (default='2000-01-01')
        The start date for the data download in YYYY-MM-DD format.
    end_date : str, optional (default=today's date in YYYY-MM-DD format)
        The end date for the data download in YYYY-MM-DD format.

    Returns:
    -------
    pd.DataFrame
        A Pandas DataFrame containing the Open, High, Low, Close, and Volume data
        for the specified tickers and date range.
    """
    ohlc = yf.download(tickers, start=start_date, end=end_date, interval='1d')
    return ohlc

def remove_sparse_columns(df: pd.DataFrame, min_rows: int = 5000) -> pd.DataFrame:
    """
    Remove sparse columns from a Pandas DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        A Pandas DataFrame to remove sparse columns from.
    min_rows : int, optional (default=5000)
        The minimum number of non-null values a column must have to be kept.

    Returns:
    -------
    pd.DataFrame
        A new Pandas DataFrame with sparse columns removed.
    """
    df_copy = df.copy()

    # Get columns with at least `min_rows` non-null values. More data is better.
    columns_with_non_null_values = df_copy.count()[df_copy.count() >= min_rows].index

    # Remove columns with null values
    df_copy = df_copy[columns_with_non_null_values]
    df_copy = df_copy.dropna(axis=1)

    return df_copy

def normalize_df(df: pd.DataFrame, column: str = 'Adj Close', column_index: str = 'Date') -> pd.DataFrame:
    """
    Normalize a Pandas DataFrame by renaming columns and the index.

    Parameters:
    ----------
    df : pd.DataFrame
        A Pandas DataFrame to be normalized.
    column : str, optional (default='Adj Close')
        The column name to use for the normalized DataFrame.
    column_index : str, optional (default='Date')
        The name to use for the index of the normalized DataFrame.

    Returns:
    -------
    pd.DataFrame
        A new Pandas DataFrame with normalized column names and index.
    """
    df_copy = df.copy()

    # If the DataFrame has multi-level columns, select the specified column
    if df.columns.nlevels > 1:
        df_copy = df_copy[column]
        df_copy.index = df_copy.index.get_level_values(column_index)

    # Rename columns and index to remove special characters and convert to lowercase
    df_copy.columns = df_copy.columns.str.replace('[^0-9a-zA-Z]+', '_', regex=True).str.lower()
    df_copy.index.name = df_copy.index.name.lower()

    return df_copy

def get_non_stationary_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and remove non-stationary stock timeseries from a DataFrame using the Augmented Dickey-Fuller test.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the stock timeseries data.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing only the stationary stock timeseries data.
    
    Raises
    ------
    ValueError
        If the input DataFrame is empty or contains only non-numeric data.
    """
    selected_stocks = df.copy()

    count = 0

    for ticker in df.columns:
        pvalue = adfuller(df[ticker])[1]

        if pvalue < 0.05:
            selected_stocks = selected_stocks.drop(ticker, axis=1)
            count += 1

    print(f'{count} non-stationary stock timeseries removed')

    return selected_stocks

def get_trending_stocks(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Identify and remove non-trending stock timeseries from a DataFrame using the Hurst exponent.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the stock timeseries data.
    threshold : float, optional
        Threshold value for the Hurst exponent below which a stock timeseries is considered non-trending.
        Default is 0.5.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing only the trending stock timeseries data.
    
    Raises
    ------
    ValueError
        If the input DataFrame is empty or contains only non-numeric data.
    """
    selected_stocks = df.copy()

    count = 0

    for ticker in df.columns:
        try:
            # Compute the Hurst exponent using the random walk method
            H, _, _ = compute_Hc(df[ticker], kind='random_walk', simplified=True)
        except:
            # If an exception is raised (usually due to negative prices), set H to 0.0
            H = 0.0

        if H <= threshold:
            selected_stocks = selected_stocks.drop(ticker, axis=1)
            count += 1

    print(f'{count} non-trending stock timeseries removed')

    return selected_stocks

def load_stocks(load_selected_stocks=False, min_rows=5000):
    """
    Load stock data from CSV file and return as a pandas DataFrame.

    Parameters
    ----------
    load_selected_stocks : bool, optional
        Whether to load the selected stocks or all stocks, by default False.
        
    min_rows : int, optional
        The minimum number of non-NaN rows a column must have to be kept in the resulting DataFrame, by default 5000.

    Returns
    -------
    stocks : pandas.DataFrame
        A DataFrame containing the stock data.
    """

    if load_selected_stocks:
        return pd.read_csv('assets/selected_stocks.csv', index_col=0)

    tickers = pd.read_csv('assets/stock_info.csv')[['Ticker']].rename(columns={'Ticker': 'ticker'}).values.flatten().tolist()

    loaded_from_file = False
    if os.path.exists('assets/stocks.csv'):
        stocks = pd.read_csv('assets/stocks.csv', index_col=0)
        loaded_from_file = True
    else:
        stocks = get_stocks(tickers)
        stocks.to_csv('assets/stocks.csv')

    stocks.index = pd.to_datetime(stocks.index, utc=True).strftime('%Y-%m-%d')

    if not loaded_from_file:
        stocks = normalize_df(stocks)

    stocks = remove_sparse_columns(stocks, min_rows=min_rows)

    return stocks