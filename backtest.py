"""
Backtesting Script for CLSTM-PPO Stock Trading System.

This script implements the backtesting pipeline and performance evaluation
as described in the paper.

Paper Reference:
- Section 4.4 (Evaluation Measures)
- Table 4, 5, 6 (Trading results)
- Figure 7, 8 (Equity curves)

Evaluation Metrics:
- CR: Cumulative Return
- MER: Maximum Earning Rate
- MPB: Maximum Pullback (Drawdown)
- APPT: Average Profitability Per Trade
- SR: Sharpe Ratio
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sb3_contrib import RecurrentPPO

from config import (
    DOW_30_TICKERS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    INITIAL_CAPITAL,
    MODEL_DIR,
    RESULTS_DIR,
)
from data_fetcher import fetch_and_prepare_data, prepare_features
from envs.multi_stock_trading_env import MultiStockTradingEnv


def calculate_metrics(portfolio_values: np.ndarray, initial_capital: float = INITIAL_CAPITAL) -> Dict:
    """
    Calculate performance metrics as defined in the paper.
    
    Paper Section 4.4 (Line 1133-1240):
    
    - CR (Cumulative Return): (P_end - P_0) / P_0
    - MER (Maximum Earning Rate): max((A_x - A_y) / A_y) where x > y, A_y < A_x
    - MPB (Maximum Pullback): max((A_x - A_y) / A_y) where x > y, A_y > A_x
    - APPT (Average Profitability Per Trade): (P_end - P_0) / N_trades
    - SR (Sharpe Ratio): (E(R_P) - R_f) / σ_P
    
    Args:
        portfolio_values: Array of daily portfolio values
        initial_capital: Initial capital
        
    Returns:
        Dictionary of metrics
    """
    portfolio_values = np.array(portfolio_values)
    
    if len(portfolio_values) < 2:
        return {
            'CR': 0, 'MER': 0, 'MPB': 0, 'APPT': 0, 'SR': 0,
            'final_value': initial_capital, 'initial_value': initial_capital
        }
    
    # Cumulative Return (Equation 7)
    final_value = portfolio_values[-1]
    cr = (final_value - initial_capital) / initial_capital
    
    # Maximum Earning Rate (Equation 8)
    # Find maximum percentage gain from any trough to subsequent peak
    mer = 0
    for i in range(len(portfolio_values)):
        for j in range(i + 1, len(portfolio_values)):
            if portfolio_values[i] > 0 and portfolio_values[j] > portfolio_values[i]:
                rate = (portfolio_values[j] - portfolio_values[i]) / portfolio_values[i]
                mer = max(mer, rate)
    
    # Maximum Pullback (Equation 9)
    # Find maximum percentage loss from any peak to subsequent trough
    mpb = 0
    peak = portfolio_values[0]
    for i in range(len(portfolio_values)):
        if portfolio_values[i] > peak:
            peak = portfolio_values[i]
        if peak > 0:
            drawdown = (peak - portfolio_values[i]) / peak
            mpb = max(mpb, drawdown)
    
    # Daily returns for Sharpe Ratio
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    daily_returns = daily_returns[~np.isnan(daily_returns) & ~np.isinf(daily_returns)]
    
    # APPT placeholder (needs trade count)
    # Will be updated with actual trade count
    appt = final_value - initial_capital
    
    # Sharpe Ratio (Equation 11)
    # Annualized: uses 252 trading days
    # Assumes risk-free rate = 0 for simplicity
    if len(daily_returns) > 0 and np.std(daily_returns) > 0:
        annualized_return = np.mean(daily_returns) * 252
        annualized_vol = np.std(daily_returns) * np.sqrt(252)
        sr = annualized_return / annualized_vol
    else:
        sr = 0
    
    return {
        'CR': cr,
        'MER': mer,
        'MPB': mpb,
        'APPT': appt,
        'SR': sr,
        'final_value': final_value,
        'initial_value': initial_capital,
        'total_return': final_value - initial_capital,
        'annualized_return': np.mean(daily_returns) * 252 if len(daily_returns) > 0 else 0,
        'annualized_volatility': np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0,
    }


def backtest(
    model: RecurrentPPO,
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    turbulence_threshold: float,
    deterministic: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run backtest on test data.
    
    Args:
        model: Trained RecurrentPPO model
        df: Stock data DataFrame
        start_date: Backtest start date
        end_date: Backtest end date
        turbulence_threshold: Turbulence threshold
        deterministic: Use deterministic actions
        
    Returns:
        Tuple of (portfolio history DataFrame, metrics dictionary)
    """
    print(f"Backtesting from {start_date} to {end_date}...")
    
    # Prepare test data
    test_df, _ = prepare_features(df, start_date, end_date)
    
    # Check if we have enough data for the lookback window
    window_size = 30  # Default window size
    if len(test_df['date'].unique()) <= window_size:
        print(f"Warning: Not enough data for backtest ({len(test_df['date'].unique())} days < {window_size}). Skipping.")
        return pd.DataFrame(), {'CR': 0, 'SR': 0, 'total_trades': 0}

    # Create test environment
    test_env = MultiStockTradingEnv(
        df=test_df,
        turbulence_threshold=turbulence_threshold,
        mode='test',
        print_verbosity=0,
    )
    
    # Run backtest episode
    obs, info = test_env.reset()
    
    # For RecurrentPPO, need to handle LSTM states
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    done = False
    while not done:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        episode_starts = np.array([done])
    
    # Get results
    portfolio_history = test_env.get_portfolio_history()
    total_trades = test_env.trades
    
    # Calculate metrics
    portfolio_values = portfolio_history['portfolio_value'].values
    metrics = calculate_metrics(portfolio_values)
    
    # Update APPT with actual trade count (Equation 10)
    if total_trades > 0:
        metrics['APPT'] = metrics['total_return'] / total_trades
    else:
        metrics['APPT'] = 0
    
    metrics['total_trades'] = total_trades
    metrics['total_cost'] = test_env.cost
    
    return portfolio_history, metrics


def calculate_buy_and_hold(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_capital: float = INITIAL_CAPITAL,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate buy-and-hold benchmark (DJI Index).
    
    Args:
        df: Stock data DataFrame
        start_date: Start date
        end_date: End date
        initial_capital: Initial capital
        
    Returns:
        Tuple of (portfolio history DataFrame, metrics dictionary)
    """
    # Prepare data
    test_df, trading_dates = prepare_features(df, start_date, end_date)
    
    # Get first and last day prices
    dates = sorted(test_df['date'].unique())
    
    portfolio_values = []
    portfolio_dates = []
    
    # Calculate equal-weighted portfolio value each day
    # Invest equal amount in each stock at start
    first_day = test_df[test_df['date'] == dates[0]]
    first_prices = first_day.set_index('tic')['close']
    
    shares_per_stock = {}
    invest_per_stock = initial_capital / len(DOW_30_TICKERS)
    
    for ticker in DOW_30_TICKERS:
        if ticker in first_prices.index and first_prices[ticker] > 0:
            shares_per_stock[ticker] = invest_per_stock / first_prices[ticker]
        else:
            shares_per_stock[ticker] = 0
    
    for date in dates:
        day_df = test_df[test_df['date'] == date].set_index('tic')
        
        portfolio_value = 0
        for ticker in DOW_30_TICKERS:
            if ticker in day_df.index:
                portfolio_value += shares_per_stock[ticker] * day_df.loc[ticker, 'close']
        
        portfolio_values.append(portfolio_value)
        portfolio_dates.append(str(date)[:10])
    
    # Create DataFrame
    portfolio_history = pd.DataFrame({
        'date': portfolio_dates,
        'portfolio_value': portfolio_values,
    })
    
    # Calculate metrics
    metrics = calculate_metrics(np.array(portfolio_values), initial_capital)
    metrics['strategy'] = 'Buy and Hold (DJI)'
    
    return portfolio_history, metrics


def plot_equity_curves(
    results: Dict[str, pd.DataFrame],
    save_path: str = None,
    title: str = "Trading Strategy Comparison",
):
    """
    Plot equity curves for comparison.
    
    Based on Figure 7 from the paper.
    
    Args:
        results: Dictionary of {strategy_name: portfolio_history_df}
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(14, 8))
    
    colors = {
        'CLSTM-PPO': 'blue',
        'Buy and Hold (DJI)': 'green',
        'PPO': 'orange',
        'Ensemble': 'red',
    }
    
    for strategy_name, portfolio_df in results.items():
        portfolio_df = portfolio_df.copy()
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        
        # Normalize to percentage return
        initial_value = portfolio_df['portfolio_value'].iloc[0]
        portfolio_df['return'] = (portfolio_df['portfolio_value'] / initial_value - 1) * 100
        
        color = colors.get(strategy_name, 'gray')
        plt.plot(portfolio_df['date'], portfolio_df['return'], 
                 label=strategy_name, color=color, linewidth=2)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def print_metrics_table(metrics_dict: Dict[str, Dict]):
    """
    Print metrics in table format similar to Table 4 in paper.
    
    Args:
        metrics_dict: Dictionary of {strategy_name: metrics}
    """
    print("\n" + "=" * 80)
    print("Performance Metrics Comparison")
    print("=" * 80)
    
    # Header
    print(f"{'Metric':<15}", end="")
    for strategy in metrics_dict.keys():
        print(f"{strategy:<20}", end="")
    print()
    print("-" * 80)
    
    # Metrics rows
    metric_names = ['CR', 'MER', 'MPB', 'APPT', 'SR']
    metric_labels = {
        'CR': 'CR (%)',
        'MER': 'MER (%)', 
        'MPB': 'MPB (%)',
        'APPT': 'APPT ($)',
        'SR': 'Sharpe Ratio',
    }
    
    for metric in metric_names:
        print(f"{metric_labels[metric]:<15}", end="")
        for strategy, metrics in metrics_dict.items():
            value = metrics.get(metric, 0)
            if metric in ['CR', 'MER', 'MPB']:
                print(f"{value*100:>17.2f}%", end="  ")
            elif metric == 'APPT':
                print(f"${value:>15.2f}", end="  ")
            else:
                print(f"{value:>18.4f}", end="  ")
        print()
    
    print("-" * 80)
    
    # Additional info
    print(f"{'Final Value':<15}", end="")
    for strategy, metrics in metrics_dict.items():
        print(f"${metrics.get('final_value', 0):>14,.2f}", end="  ")
    print()
    
    print(f"{'Total Trades':<15}", end="")
    for strategy, metrics in metrics_dict.items():
        trades = metrics.get('total_trades', 'N/A')
        if isinstance(trades, (int, float)):
            print(f"{trades:>18}", end="  ")
        else:
            print(f"{trades:>18}", end="  ")
    print()
    
    print("=" * 80)


def main():
    """Main backtesting entry point."""
    parser = argparse.ArgumentParser(description="Backtest CLSTM-PPO Stock Trading Agent")
    parser.add_argument(
        "--model_path", type=str, 
        default=os.path.join(MODEL_DIR, "clstm_ppo_final.zip"),
        help="Path to trained model"
    )
    parser.add_argument(
        "--start_date", type=str, default=TEST_START_DATE,
        help=f"Backtest start date (default: {TEST_START_DATE})"
    )
    parser.add_argument(
        "--end_date", type=str, default=TEST_END_DATE,
        help=f"Backtest end date (default: {TEST_END_DATE})"
    )
    parser.add_argument(
        "--save_dir", type=str, default=RESULTS_DIR,
        help=f"Directory to save results (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Skip plotting"
    )
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("CLSTM-PPO Stock Trading System - Backtesting")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first using train.py")
        return
    
    # Fetch and prepare data
    print("\nStep 1: Fetching and preparing data...")
    df, turbulence_threshold = fetch_and_prepare_data()
    
    print(f"\nData summary:")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Stocks: {df['tic'].nunique()}")
    print(f"  Turbulence threshold: {turbulence_threshold:.2f}")
    
    # Load model
    print(f"\nStep 2: Loading model from {args.model_path}...")
    model = RecurrentPPO.load(args.model_path)
    
    # Run backtest
    print("\nStep 3: Running backtest...")
    results = {}
    metrics_dict = {}
    
    # CLSTM-PPO strategy
    print("\n--- CLSTM-PPO Strategy ---")
    portfolio_history, metrics = backtest(
        model=model,
        df=df,
        start_date=args.start_date,
        end_date=args.end_date,
        turbulence_threshold=turbulence_threshold,
    )
    results['CLSTM-PPO'] = portfolio_history
    metrics['strategy'] = 'CLSTM-PPO'
    metrics_dict['CLSTM-PPO'] = metrics
    
    # Buy and Hold benchmark
    print("\n--- Buy and Hold (DJI) Benchmark ---")
    bh_portfolio, bh_metrics = calculate_buy_and_hold(
        df=df,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    results['Buy and Hold (DJI)'] = bh_portfolio
    metrics_dict['Buy and Hold (DJI)'] = bh_metrics
    
    # Print metrics comparison
    print_metrics_table(metrics_dict)
    
    # Save results
    print(f"\nStep 4: Saving results to {args.save_dir}...")
    
    # Save portfolio histories
    for strategy_name, portfolio_df in results.items():
        safe_name = strategy_name.replace(' ', '_').replace('(', '').replace(')', '')
        csv_path = os.path.join(args.save_dir, f"portfolio_{safe_name}.csv")
        portfolio_df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_path = os.path.join(args.save_dir, "metrics_comparison.csv")
    metrics_df.to_csv(metrics_path)
    print(f"  Saved {metrics_path}")
    
    # Plot equity curves
    if not args.no_plot:
        print("\nStep 5: Plotting equity curves...")
        plot_path = os.path.join(args.save_dir, "equity_curves.png")
        plot_equity_curves(
            results=results,
            save_path=plot_path,
            title=f"CLSTM-PPO vs Buy-and-Hold ({args.start_date} to {args.end_date})"
        )
    
    print("\n" + "=" * 60)
    print("Backtesting completed!")
    print("=" * 60)
    
    # Print paper comparison
    print("\n" + "=" * 60)
    print("Comparison with Paper Results (Table 4)")
    print("=" * 60)
    print("Expected CLSTM-PPO results on DJI (2016-01-01 to 2020-05-08):")
    print("  CR: 90.81%")
    print("  MER: 113.50%")
    print("  MPB: 46.51%")
    print("  APPT: 35.27")
    print("  SR: 1.1540")
    print("\nYour results:")
    print(f"  CR: {metrics_dict['CLSTM-PPO']['CR']*100:.2f}%")
    print(f"  MER: {metrics_dict['CLSTM-PPO']['MER']*100:.2f}%")
    print(f"  MPB: {metrics_dict['CLSTM-PPO']['MPB']*100:.2f}%")
    print(f"  APPT: {metrics_dict['CLSTM-PPO']['APPT']:.2f}")
    print(f"  SR: {metrics_dict['CLSTM-PPO']['SR']:.4f}")


if __name__ == "__main__":
    main()
