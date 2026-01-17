"""
Data Fetcher Module for CLSTM-PPO Stock Trading System.

This module handles:
1. Downloading historical stock data from Yahoo Finance
2. Computing technical indicators (MACD, RSI, CCI, ADX)
3. Preparing 181-dimensional feature vectors

Based on: "A Novel Deep Reinforcement Learning Based Automated Stock Trading System 
Using Cascaded LSTM Networks" (Zou et al., 2023)
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Technical indicator library
import ta
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator

from config import (
    DOW_30_TICKERS, 
    TRAIN_START_DATE, 
    TEST_END_DATE,
    DATA_DIR
)


def download_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        save_path: Optional path to save the data
        
    Returns:
        DataFrame with OHLCV data for all tickers
    """
    print(f"Downloading data for {len(tickers)} stocks from {start_date} to {end_date}...")
    
    # Add buffer days for technical indicator calculation
    buffer_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    
    all_data = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=buffer_start, end=end_date)
            
            if df.empty:
                print(f"  Warning: No data for {ticker}")
                continue
            
            # Reset index to get date as column
            df = df.reset_index()
            df['tic'] = ticker
            
            # Rename columns to standard format
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Select relevant columns
            df = df[['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]
            
            # Remove timezone info if present
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            
            all_data.append(df)
            print(f"  Downloaded {ticker}: {len(df)} days")
            
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data downloaded for any ticker!")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    print(f"Total records: {len(combined_df)}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        combined_df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
    
    return combined_df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the stock data.
    
    Indicators (Paper Section 3.1.1, Line 430-468):
    - MACD: Moving Average Convergence Divergence
    - RSI: Relative Strength Index
    - CCI: Commodity Channel Index
    - ADX: Average Directional Index
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    print("Computing technical indicators...")
    
    df = df.copy()
    result_dfs = []
    
    tickers = df['tic'].unique()
    
    for ticker in tickers:
        ticker_df = df[df['tic'] == ticker].copy()
        ticker_df = ticker_df.sort_values('date').reset_index(drop=True)
        
        # Ensure we have enough data
        if len(ticker_df) < 30:
            print(f"  Warning: {ticker} has less than 30 days of data, skipping")
            continue
        
        try:
            # MACD (Moving Average Convergence Divergence)
            # Default: fast=12, slow=26, signal=9
            macd_indicator = MACD(
                close=ticker_df['close'],
                window_slow=26,
                window_fast=12,
                window_sign=9
            )
            ticker_df['macd'] = macd_indicator.macd()
            
            # RSI (Relative Strength Index)
            # Default: window=14
            rsi_indicator = RSIIndicator(
                close=ticker_df['close'],
                window=14
            )
            ticker_df['rsi'] = rsi_indicator.rsi()
            
            # CCI (Commodity Channel Index)
            # Default: window=20
            cci_indicator = CCIIndicator(
                high=ticker_df['high'],
                low=ticker_df['low'],
                close=ticker_df['close'],
                window=20
            )
            ticker_df['cci'] = cci_indicator.cci()
            
            # ADX (Average Directional Index)
            # Default: window=14
            adx_indicator = ADXIndicator(
                high=ticker_df['high'],
                low=ticker_df['low'],
                close=ticker_df['close'],
                window=14
            )
            ticker_df['adx'] = adx_indicator.adx()
            
            result_dfs.append(ticker_df)
            
        except Exception as e:
            print(f"  Error computing indicators for {ticker}: {e}")
            continue
    
    if not result_dfs:
        raise ValueError("No data after computing technical indicators!")
    
    result_df = pd.concat(result_dfs, ignore_index=True)
    result_df = result_df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    # Fill NaN values (from indicator warmup period)
    result_df = result_df.fillna(method='bfill')
    result_df = result_df.fillna(0)
    
    print(f"Technical indicators computed. Shape: {result_df.shape}")
    
    return result_df


def prepare_features(
    df: pd.DataFrame,
    start_date: str,
    end_date: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature data for the trading environment.
    
    Args:
        df: DataFrame with OHLCV and technical indicators
        start_date: Start date to filter data
        end_date: End date to filter data
        
    Returns:
        Tuple of (filtered DataFrame, list of trading dates)
    """
    df = df.copy()
    
    # Convert date column to datetime if not already
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter by date range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df = df[mask].copy()
    
    if df.empty:
        raise ValueError(f"No data between {start_date} and {end_date}")
    
    # Get unique trading dates
    trading_dates = sorted(df['date'].unique())
    
    # Ensure we have all 30 stocks for each date
    # Pivot and forward fill missing data
    dates_with_all_stocks = []
    for date in trading_dates:
        date_df = df[df['date'] == date]
        if len(date_df['tic'].unique()) == 30:
            dates_with_all_stocks.append(date)
    
    if not dates_with_all_stocks:
        print("Warning: No dates have all 30 stocks. Using available data.")
        dates_with_all_stocks = trading_dates
    
    df = df[df['date'].isin(dates_with_all_stocks)]
    trading_dates = [str(d)[:10] for d in sorted(df['date'].unique())]
    
    print(f"Prepared data: {len(trading_dates)} trading days, {df['tic'].nunique()} stocks")
    
    return df, trading_dates


def get_state_vector(df: pd.DataFrame, date: str, tickers: List[str]) -> np.ndarray:
    """
    Construct the 181-dimensional state vector for a given date.
    
    State Space (Paper Section 3.1.1, Line 393-468):
    [b_t(1), p_t(30), h_t(30), M_t(30), R_t(30), C_t(30), X_t(30)] = 181 dim
    
    Note: This function returns only the market data part (180 dim).
    Balance is added by the environment.
    
    Args:
        df: DataFrame with stock data and indicators
        date: Date string
        tickers: List of ticker symbols (must be in consistent order)
        
    Returns:
        180-dimensional numpy array (excluding balance)
    """
    date_df = df[df['date'] == date].copy()
    date_df = date_df.set_index('tic')
    
    # Ensure consistent ordering
    prices = []
    macd_values = []
    rsi_values = []
    cci_values = []
    adx_values = []
    
    for ticker in tickers:
        if ticker in date_df.index:
            row = date_df.loc[ticker]
            prices.append(row['close'])
            macd_values.append(row['macd'])
            rsi_values.append(row['rsi'])
            cci_values.append(row['cci'])
            adx_values.append(row['adx'])
        else:
            # Use 0 if ticker missing (shouldn't happen with proper data prep)
            prices.append(0)
            macd_values.append(0)
            rsi_values.append(0)
            cci_values.append(0)
            adx_values.append(0)
    
    # State vector: prices(30) + macd(30) + rsi(30) + cci(30) + adx(30) = 150 dim
    # Note: shares(30) will be added by environment
    state = np.array(prices + macd_values + rsi_values + cci_values + adx_values)
    
    return state


def calculate_turbulence(
    df: pd.DataFrame,
    current_date: str,
    lookback_days: int = 252
) -> float:
    """
    Calculate turbulence index for market risk assessment.
    
    Turbulence (Paper Equation 3, Line 558-567):
    turbulence_t = (y_t - μ)' Σ^{-1} (y_t - μ)
    
    Args:
        df: DataFrame with stock data
        current_date: Current date
        lookback_days: Number of historical days for covariance calculation
        
    Returns:
        Turbulence index value
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    current_date = pd.to_datetime(current_date)
    
    # Get historical returns
    historical_df = df[df['date'] < current_date].copy()
    historical_df = historical_df.tail(lookback_days * 30)  # Approximate for 30 stocks
    
    if len(historical_df) < 100:
        return 0.0
    
    # Pivot to get returns matrix
    pivot_df = historical_df.pivot(index='date', columns='tic', values='close')
    returns = pivot_df.pct_change().dropna()
    
    if len(returns) < 30:
        return 0.0
    
    # Current day returns
    current_df = df[df['date'] == current_date]
    prev_date = returns.index[-1]
    
    if len(current_df) < 30:
        return 0.0
    
    try:
        current_prices = current_df.set_index('tic')['close']
        prev_prices = pivot_df.loc[prev_date]
        current_returns = (current_prices / prev_prices - 1).values
        
        # Calculate turbulence
        historical_mean = returns.mean().values
        historical_cov = returns.cov().values
        
        diff = current_returns - historical_mean
        
        # Add small regularization to covariance matrix for numerical stability
        cov_inv = np.linalg.inv(historical_cov + np.eye(len(historical_cov)) * 1e-6)
        
        turbulence = np.dot(np.dot(diff, cov_inv), diff)
        
        return float(turbulence)
        
    except Exception as e:
        print(f"Error calculating turbulence: {e}")
        return 0.0


def get_turbulence_threshold(df: pd.DataFrame, percentile: int = 90) -> float:
    """
    Calculate turbulence threshold based on historical percentile.
    
    Paper Line 578-579: "We set the turbulence threshold to 90th percentile 
    of all historical turbulence indexes"
    
    Args:
        df: DataFrame with stock data
        percentile: Percentile for threshold (default 90)
        
    Returns:
        Turbulence threshold value
    """
    dates = sorted(df['date'].unique())
    turbulence_values = []
    
    # Calculate turbulence for each date (skip first year for warmup)
    warmup_dates = dates[:252]  # Skip first year
    
    for date in dates[252:]:
        turb = calculate_turbulence(df, str(date)[:10])
        if turb > 0:
            turbulence_values.append(turb)
    
    if not turbulence_values:
        return float('inf')  # Never trigger if no valid turbulence
    
    threshold = np.percentile(turbulence_values, percentile)
    print(f"Turbulence threshold ({percentile}th percentile): {threshold:.2f}")
    
    return threshold


def fetch_and_prepare_data(
    tickers: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    save_dir: str = None
) -> Tuple[pd.DataFrame, float]:
    """
    Main function to fetch data and prepare for training/testing.
    
    Args:
        tickers: List of stock tickers (default: DOW_30)
        start_date: Start date (default: TRAIN_START_DATE)
        end_date: End date (default: TEST_END_DATE)
        save_dir: Directory to save data (default: DATA_DIR)
        
    Returns:
        Tuple of (prepared DataFrame, turbulence threshold)
    """
    tickers = tickers or DOW_30_TICKERS
    start_date = start_date or TRAIN_START_DATE
    end_date = end_date or TEST_END_DATE
    save_dir = save_dir or DATA_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if data already exists
    raw_data_path = os.path.join(save_dir, "raw_stock_data.csv")
    processed_data_path = os.path.join(save_dir, "processed_stock_data.csv")
    
    if os.path.exists(processed_data_path):
        print(f"Loading existing processed data from {processed_data_path}")
        df = pd.read_csv(processed_data_path)
        df['date'] = pd.to_datetime(df['date'])
    else:
        # Download raw data
        if os.path.exists(raw_data_path):
            print(f"Loading existing raw data from {raw_data_path}")
            df = pd.read_csv(raw_data_path)
        else:
            df = download_stock_data(tickers, start_date, end_date, raw_data_path)
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Save processed data
        df.to_csv(processed_data_path, index=False)
        print(f"Processed data saved to {processed_data_path}")
    
    # Calculate turbulence threshold
    turbulence_threshold = get_turbulence_threshold(df)
    
    return df, turbulence_threshold


if __name__ == "__main__":
    # Test data fetching
    print("=" * 60)
    print("CLSTM-PPO Data Fetcher Test")
    print("=" * 60)
    
    # Fetch and prepare data
    df, turb_threshold = fetch_and_prepare_data()
    
    print("\n" + "=" * 60)
    print("Data Summary:")
    print("=" * 60)
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of stocks: {df['tic'].nunique()}")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Turbulence threshold: {turb_threshold:.2f}")
    
    # Test state vector construction
    print("\n" + "=" * 60)
    print("State Vector Test:")
    print("=" * 60)
    
    test_date = df['date'].iloc[100]
    state = get_state_vector(df, str(test_date)[:10], DOW_30_TICKERS)
    print(f"State vector shape: {state.shape}")
    print(f"State vector sample (first 10): {state[:10]}")
    
    print("\nData fetcher test completed successfully!")
