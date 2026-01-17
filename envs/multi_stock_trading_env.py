"""
Multi-Stock Trading Environment for CLSTM-PPO.

This module implements an OpenAI Gym compatible environment for multi-stock trading.
Based on Yang[14] and the CLSTM-PPO paper specifications.

Paper Reference: "A Novel Deep Reinforcement Learning Based Automated Stock Trading System 
Using Cascaded LSTM Networks" (Zou et al., 2023)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import (
    DOW_30_TICKERS,
    INITIAL_CAPITAL,
    HMAX,
    TRANSACTION_COST_PCT,
    REWARD_SCALING,
    STATE_DIM,
    LSTM_WINDOW_SIZE,
)


class MultiStockTradingEnv(gym.Env):
    """
    Multi-stock trading environment following OpenAI Gym interface.
    
    State Space (Paper Section 3.1.1, 181 dimensions):
    [b_t(1), p_t(30), h_t(30), M_t(30), R_t(30), C_t(30), X_t(30)]
    - b_t: Available balance (1 dim)
    - p_t: Adjusted close prices (30 dim)
    - h_t: Shares owned (30 dim)
    - M_t: MACD values (30 dim)
    - R_t: RSI values (30 dim)
    - C_t: CCI values (30 dim)
    - X_t: ADX values (30 dim)
    
    Action Space (Paper Section 3.1.2):
    Continuous [-1, 1] for 30 stocks
    - Positive: Buy shares
    - Negative: Sell shares
    - Magnitude × hmax = number of shares
    
    Reward (Paper Section 3.1.3, Equation 1):
    Return_t = (b_{t+1} + p_{t+1}^T h_{t+1}) - (b_t + p_t^T h_t) - c_t
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        tickers: List[str] = None,
        initial_capital: float = INITIAL_CAPITAL,
        hmax: int = HMAX,
        transaction_cost_pct: float = TRANSACTION_COST_PCT,
        reward_scaling: float = REWARD_SCALING,
        window_size: int = LSTM_WINDOW_SIZE,
        turbulence_threshold: float = float('inf'),
        mode: str = 'train',
        print_verbosity: int = 1,
    ):
        """
        Initialize the multi-stock trading environment.
        
        Args:
            df: DataFrame with columns ['date', 'tic', 'close', 'macd', 'rsi', 'cci', 'adx']
            tickers: List of stock ticker symbols (default: DOW_30)
            initial_capital: Starting capital (default: $1M)
            hmax: Maximum shares per trade (default: 100)
            transaction_cost_pct: Transaction cost percentage (default: 0.1%)
            reward_scaling: Reward scaling factor (default: 1e-4)
            window_size: Size of observation window for LSTM (default: 30)
            turbulence_threshold: Threshold for turbulence index
            mode: 'train' or 'test'
            print_verbosity: Verbosity level for printing
        """
        super().__init__()
        
        self.df = df.copy()
        self.tickers = tickers or DOW_30_TICKERS
        self.num_stocks = len(self.tickers)
        
        # Environment parameters (Paper Section 3.1.5)
        self.initial_capital = initial_capital
        self.hmax = hmax
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        self.turbulence_threshold = turbulence_threshold
        self.mode = mode
        self.print_verbosity = print_verbosity
        
        # Prepare data
        self._prepare_data()
        
        # State dimension: 181 for single step, or 181 * window_size for LSTM
        self.state_dim = 1 + self.num_stocks * 6  # balance + 30*(price+shares+4 indicators)
        
        # Define action and observation spaces
        # Action space: continuous [-1, 1] for each stock
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.num_stocks,), 
            dtype=np.float32
        )
        
        # Observation space: window_size * state_dim for LSTM input
        # Flattened: (window_size * 181,)
        obs_dim = self.window_size * self.state_dim
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _prepare_data(self):
        """Prepare data structures for efficient access."""
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.dates = sorted(self.df['date'].unique())
        self.num_days = len(self.dates)
        
        # Create lookup dictionaries for fast access
        self.price_data = {}  # date -> {ticker: price}
        self.indicator_data = {}  # date -> {ticker: {macd, rsi, cci, adx}}
        self.turbulence_data = {} # date -> turbulence_value
        
        for date in self.dates:
            date_df = self.df[self.df['date'] == date].set_index('tic')
            
            self.price_data[date] = {}
            self.indicator_data[date] = {}
            
            # Store turbulence (same for all tickers on this date)
            # Take the first available turbulence value for this date
            if 'turbulence' in date_df.columns:
                self.turbulence_data[date] = date_df['turbulence'].iloc[0]
            else:
                self.turbulence_data[date] = 0.0
            
            for ticker in self.tickers:
                if ticker in date_df.index:
                    row = date_df.loc[ticker]
                    self.price_data[date][ticker] = row['close']
                    self.indicator_data[date][ticker] = {
                        'macd': row.get('macd', 0),
                        'rsi': row.get('rsi', 0),
                        'cci': row.get('cci', 0),
                        'adx': row.get('adx', 0),
                    }
                else:
                    self.price_data[date][ticker] = 0
                    self.indicator_data[date][ticker] = {
                        'macd': 0, 'rsi': 0, 'cci': 0, 'adx': 0
                    }
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset portfolio state
        self.balance = self.initial_capital
        self.shares = np.zeros(self.num_stocks, dtype=np.float32)
        self.cost = 0
        self.trades = 0
        
        # Reset time
        self.day_index = self.window_size - 1  # Start after enough history for window
        
        # Track portfolio values
        self.asset_memory = [self.initial_capital]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [str(self.dates[self.day_index])[:10]]
        
        # Previous portfolio value for reward calculation
        self.prev_portfolio_value = self.initial_capital
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_prices(self, date) -> np.ndarray:
        """Get prices for all stocks on a given date."""
        prices = np.array([
            self.price_data[date].get(ticker, 0) 
            for ticker in self.tickers
        ], dtype=np.float32)
        return prices
    
    def _get_indicators(self, date) -> Dict[str, np.ndarray]:
        """Get technical indicators for all stocks on a given date."""
        indicators = {
            'macd': np.zeros(self.num_stocks, dtype=np.float32),
            'rsi': np.zeros(self.num_stocks, dtype=np.float32),
            'cci': np.zeros(self.num_stocks, dtype=np.float32),
            'adx': np.zeros(self.num_stocks, dtype=np.float32),
        }
        
        for i, ticker in enumerate(self.tickers):
            ind = self.indicator_data[date].get(ticker, {})
            indicators['macd'][i] = ind.get('macd', 0)
            indicators['rsi'][i] = ind.get('rsi', 0)
            indicators['cci'][i] = ind.get('cci', 0)
            indicators['adx'][i] = ind.get('adx', 0)
        
        return indicators
    
    def _get_state(self, date) -> np.ndarray:
        """
        Construct single-step state vector (181 dim).
        
        State = [balance(1), price(30), shares(30), macd(30), rsi(30), cci(30), adx(30)]
        """
        prices = self._get_prices(date)
        indicators = self._get_indicators(date)
        
        # Construct state vector
        state = np.concatenate([
            [self.balance],           # 1 dim
            prices,                   # 30 dim
            self.shares,              # 30 dim
            indicators['macd'],       # 30 dim
            indicators['rsi'],        # 30 dim
            indicators['cci'],        # 30 dim
            indicators['adx'],        # 30 dim
        ]).astype(np.float32)
        
        return state
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation with window_size history for LSTM.
        
        Returns:
            Flattened observation of shape (window_size * 181,)
        """
        observations = []
        
        for i in range(self.window_size):
            day_idx = self.day_index - self.window_size + 1 + i
            if day_idx < 0:
                day_idx = 0
            
            date = self.dates[day_idx]
            state = self._get_state(date)
            observations.append(state)
        
        # Stack and flatten
        observation = np.concatenate(observations).astype(np.float32)
        
        return observation
    
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        current_date = self.dates[self.day_index]
        prices = self._get_prices(current_date)
        stock_value = np.dot(prices, self.shares)
        return self.balance + stock_value
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Array of shape (30,) with values in [-1, 1]
                   Positive = buy, Negative = sell
                   |action| * hmax = number of shares
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Clip actions to valid range
        action = np.clip(action, -1, 1)
        
        # Store action for logging
        self.actions_memory.append(action.copy())
        
        # Get current date and prices
        current_date = self.dates[self.day_index]
        prices = self._get_prices(current_date)
        
        # Calculate portfolio value before trade
        prev_portfolio = self._get_portfolio_value()
        
        # Execute trades
        # Action in [-1, 1] -> shares to trade = action * hmax
        shares_to_trade = (action * self.hmax).astype(np.int32)
        
        # Turbulence Logic (Market Risk Filter)
        # Check if turbulence exceeds threshold
        current_turbulence = self.turbulence_data.get(current_date, 0)
        
        if self.turbulence_threshold is not None and current_turbulence > self.turbulence_threshold:
            # Force Sell All (Close Position)
            # Paper Line 578: "all the stocks will be sold to avoid risk"
            shares_to_trade = -self.shares.astype(np.int32)
            
            if self.print_verbosity >= 1:
                print(f"Turbulence triggered ({current_turbulence:.2f} > {self.turbulence_threshold:.2f}) at {current_date}. Selling all.")
        
        
        # Process sells first (to free up capital)
        sell_mask = shares_to_trade < 0
        for i in np.where(sell_mask)[0]:
            shares_to_sell = min(-shares_to_trade[i], int(self.shares[i]))
            if shares_to_sell > 0 and prices[i] > 0:
                sell_amount = shares_to_sell * prices[i]
                transaction_cost = sell_amount * self.transaction_cost_pct
                self.balance += sell_amount - transaction_cost
                self.shares[i] -= shares_to_sell
                self.cost += transaction_cost
                self.trades += 1
        
        # Process buys
        buy_mask = shares_to_trade > 0
        for i in np.where(buy_mask)[0]:
            shares_to_buy = shares_to_trade[i]
            if prices[i] > 0:
                # Calculate maximum affordable shares
                max_affordable = int(self.balance / (prices[i] * (1 + self.transaction_cost_pct)))
                shares_to_buy = min(shares_to_buy, max_affordable)
                
                if shares_to_buy > 0:
                    buy_amount = shares_to_buy * prices[i]
                    transaction_cost = buy_amount * self.transaction_cost_pct
                    self.balance -= buy_amount + transaction_cost
                    self.shares[i] += shares_to_buy
                    self.cost += transaction_cost
                    self.trades += 1
        
        # Move to next day
        self.day_index += 1
        terminated = self.day_index >= self.num_days - 1
        truncated = False
        
        if not terminated:
            # Calculate reward (Paper Equation 1)
            new_portfolio = self._get_portfolio_value()
            reward = (new_portfolio - prev_portfolio) * self.reward_scaling
            
            # Update previous portfolio value
            self.prev_portfolio_value = new_portfolio
            
            # Store for logging
            self.asset_memory.append(new_portfolio)
            self.rewards_memory.append(reward)
            self.date_memory.append(str(self.dates[self.day_index])[:10])
        else:
            reward = 0
        
        # Get new observation
        observation = self._get_observation() if not terminated else np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        return {
            'balance': self.balance,
            'shares': self.shares.copy(),
            'portfolio_value': self._get_portfolio_value(),
            'total_cost': self.cost,
            'total_trades': self.trades,
            'day_index': self.day_index,
            'current_date': str(self.dates[self.day_index])[:10] if self.day_index < len(self.dates) else None,
        }
    
    def render(self, mode='human'):
        """Render the environment state."""
        if self.print_verbosity > 0:
            info = self._get_info()
            print(f"Day: {info['day_index']}, Date: {info['current_date']}")
            print(f"  Balance: ${info['balance']:,.2f}")
            print(f"  Portfolio Value: ${info['portfolio_value']:,.2f}")
            print(f"  Total Trades: {info['total_trades']}")
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get portfolio value history as DataFrame.
        
        Returns:
            DataFrame with date and portfolio value
        """
        return pd.DataFrame({
            'date': self.date_memory,
            'portfolio_value': self.asset_memory[:len(self.date_memory)]
        })
    
    def close(self):
        """Clean up resources."""
        pass


def make_env(
    df: pd.DataFrame,
    mode: str = 'train',
    **kwargs
) -> MultiStockTradingEnv:
    """
    Factory function to create trading environment.
    
    Args:
        df: Stock data DataFrame
        mode: 'train' or 'test'
        **kwargs: Additional arguments for environment
        
    Returns:
        MultiStockTradingEnv instance
    """
    return MultiStockTradingEnv(df, mode=mode, **kwargs)


if __name__ == "__main__":
    # Test the environment
    print("=" * 60)
    print("Multi-Stock Trading Environment Test")
    print("=" * 60)
    
    # Create dummy data for testing
    import sys
    sys.path.append('..')
    
    from data_fetcher import fetch_and_prepare_data, prepare_features
    from config import TRAIN_START_DATE, TRAIN_END_DATE
    
    # Load data
    print("\nLoading data...")
    df, turb_threshold = fetch_and_prepare_data()
    
    # Prepare training data
    train_df, _ = prepare_features(df, TRAIN_START_DATE, TRAIN_END_DATE)
    
    # Create environment
    print("\nCreating environment...")
    env = MultiStockTradingEnv(
        df=train_df,
        turbulence_threshold=turb_threshold,
        mode='train'
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test step
    print("\nTesting steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, portfolio=${info['portfolio_value']:,.2f}")
        
        if terminated:
            break
    
    print("\nEnvironment test completed successfully!")
