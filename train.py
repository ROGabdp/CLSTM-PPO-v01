"""
Training Script for CLSTM-PPO Stock Trading System.

This script implements the training pipeline with rolling window retraining
as described in the paper.

Paper Reference:
- Section 4.1 (Data splitting and rolling window training)
- Algorithm 2 (PPO with LSTM)
- Table 1 (PPO hyperparameters)

Training Strategy (Line 1018-1027):
- Initial training: 2009-01-01 to 2016-01-01
- Retrain every 3 months with accumulated data
- Continue training during test period
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    DOW_30_TICKERS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    ROLLING_WINDOW_MONTHS,
    PPO_PARAMS,
    LSTM_HIDDEN_SIZE_PPO,
    LSTM_FEATURE_DIM,
    LSTM_WINDOW_SIZE,
    TOTAL_TIMESTEPS,
    MODEL_DIR,
    SEED,
)
from data_fetcher import fetch_and_prepare_data, prepare_features
from envs.multi_stock_trading_env import MultiStockTradingEnv
from envs.multi_stock_trading_env import MultiStockTradingEnv
from models.lstm_feature_extractor import LSTMFeatureExtractor

# Import backtest function for rolling validation
from backtest import backtest


class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics to tensorboard."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.portfolio_values = []
    
    def _on_step(self) -> bool:
        # Log episode reward if available
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info:
                self.logger.record('rollout/ep_reward', ep_info['r'])
        
        return True


def create_train_env(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    turbulence_threshold: float,
    **kwargs
) -> DummyVecEnv:
    """
    Create training environment.
    
    Args:
        df: Full stock data DataFrame
        start_date: Training start date
        end_date: Training end date
        turbulence_threshold: Turbulence threshold for risk control
        **kwargs: Additional environment arguments
        
    Returns:
        Vectorized environment
    """
    # Filter data for date range
    train_df, _ = prepare_features(df, start_date, end_date)
    
    def make_env():
        env = MultiStockTradingEnv(
            df=train_df,
            turbulence_threshold=turbulence_threshold,
            mode='train',
            **kwargs
        )
        return env
    
    return DummyVecEnv([make_env])


def create_model(
    env: DummyVecEnv,
    tensorboard_log: str = None,
    seed: int = SEED,
) -> RecurrentPPO:
    """
    Create CLSTM-PPO model with paper specifications.
    
    Uses RecurrentPPO from sb3-contrib which provides LSTM policy.
    Combines with custom LSTM Feature Extractor for cascaded architecture.
    
    Paper Reference:
    - Table 1 (PPO hyperparameters)
    - Table 3 (LSTM hidden size = 512)
    - Section 3.2.2 (LSTM as feature extractor)
    
    Args:
        env: Training environment
        tensorboard_log: Path for tensorboard logs
        seed: Random seed
        
    Returns:
        Configured RecurrentPPO model
    """
    # Policy kwargs for LSTM policy
    # Paper Table 3: Best hidden size is 512 for PPO LSTM
    policy_kwargs = dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=LSTM_FEATURE_DIM,        # 128 dim output
            window_size=LSTM_WINDOW_SIZE,          # T=30
        ),
        lstm_hidden_size=LSTM_HIDDEN_SIZE_PPO,     # 512 (Table 3)
        n_lstm_layers=1,
        shared_lstm=False,                          # Separate LSTM for actor and critic
        enable_critic_lstm=True,                    # LSTM in critic as well
        net_arch=dict(pi=[256, 128], vf=[256, 128]),  # MLP after LSTM
    )
    
    # Create RecurrentPPO model with paper hyperparameters (Table 1)
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=PPO_PARAMS['learning_rate'],      # 3e-4
        n_steps=PPO_PARAMS['n_steps'],                  # 128
        batch_size=64,                                   # Minibatch size
        n_epochs=10,                                     # Number of epochs per update
        gamma=PPO_PARAMS['gamma'],                      # 0.99 (discount factor)
        gae_lambda=0.95,                                # GAE lambda
        clip_range=PPO_PARAMS['clip_range'],            # 0.2
        ent_coef=PPO_PARAMS['ent_coef'],                # 0.01 (entropy coefficient)
        vf_coef=PPO_PARAMS['vf_coef'],                  # 0.5 (value function coefficient)
        max_grad_norm=PPO_PARAMS['max_grad_norm'],      # 0.5
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=1,
    )
    
    return model


def generate_rolling_windows(
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    rolling_months: int = ROLLING_WINDOW_MONTHS,
) -> List[Tuple[str, str, str, str]]:
    """
    Generate rolling window train/test periods.
    
    Paper (Line 1022-1027):
    "The first training was started from 01/01/2009 until 01/01/2016, 
    and then traded on the test set for three months. 
    The second training will start from 01/01/2009 until 03/01/2016, 
    then trade from 03/02/2016 until 06/01/2016. 
    and so on until the last quarter of the test set."
    
    Args:
        train_start: Training start date
        train_end: Initial training end date
        test_start: Testing start date
        test_end: Testing end date
        rolling_months: Months between retraining
        
    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    windows = []
    
    train_start_dt = datetime.strptime(train_start, "%Y-%m-%d")
    current_train_end = datetime.strptime(train_end, "%Y-%m-%d")
    test_end_dt = datetime.strptime(test_end, "%Y-%m-%d")
    
    current_test_start = current_train_end + timedelta(days=1)
    
    while current_test_start < test_end_dt:
        # Calculate test end (3 months later)
        current_test_end = current_test_start + relativedelta(months=rolling_months)
        
        if current_test_end > test_end_dt:
            current_test_end = test_end_dt
        
        windows.append((
            train_start,
            current_train_end.strftime("%Y-%m-%d"),
            current_test_start.strftime("%Y-%m-%d"),
            current_test_end.strftime("%Y-%m-%d"),
        ))
        
        # Move to next window
        current_train_end = current_test_end
        current_test_start = current_test_end + timedelta(days=1)
    
    return windows


def train_single_period(
    model: RecurrentPPO,
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    turbulence_threshold: float,
    timesteps: int,
    reset_num_timesteps: bool = True,
) -> RecurrentPPO:
    """
    Train model for a single period.
    
    Args:
        model: RecurrentPPO model
        df: Stock data DataFrame
        train_start: Training start date
        train_end: Training end date  
        turbulence_threshold: Turbulence threshold
        timesteps: Number of training timesteps
        reset_num_timesteps: Whether to reset timestep counter
        
    Returns:
        Trained model
    """
    # Create training environment
    train_env = create_train_env(df, train_start, train_end, turbulence_threshold)
    
    # Update model environment
    model.set_env(train_env)
    
    # Train
    print(f"Training from {train_start} to {train_end} for {timesteps:,} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=reset_num_timesteps,
        callback=TensorboardCallback(),
        progress_bar=True,
    )
    
    return model


def train_with_rolling_window(
    df: pd.DataFrame,
    turbulence_threshold: float,
    timesteps_per_window: int = TOTAL_TIMESTEPS,
    save_dir: str = MODEL_DIR,
) -> RecurrentPPO:
    """
    Train model with rolling window strategy.
    
    Paper Section 4.1: Uses rolling window retraining every 3 months.
    
    Args:
        df: Stock data DataFrame
        turbulence_threshold: Turbulence threshold for risk control
        timesteps_per_window: Training timesteps per window
        save_dir: Directory to save models
        
    Returns:
        Trained model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate rolling windows
    windows = generate_rolling_windows(
        TRAIN_START_DATE, TRAIN_END_DATE,
        TEST_START_DATE, TEST_END_DATE,
        ROLLING_WINDOW_MONTHS
    )
    
    print(f"\n{'='*60}")
    print(f"Rolling Window Training")
    print(f"{'='*60}")
    print(f"Number of windows: {len(windows)}")
    print(f"Timesteps per window: {timesteps_per_window:,}")
    
    # Create initial environment and model
    initial_env = create_train_env(
        df, TRAIN_START_DATE, TRAIN_END_DATE, turbulence_threshold
    )
    
    model = create_model(
        env=initial_env,
        tensorboard_log=os.path.join(save_dir, "logs"),
    )
    
    # Train on each window
    for i, (t_start, t_end, test_start, test_end) in enumerate(windows):
        print(f"\n{'='*60}")
        print(f"Window {i+1}/{len(windows)}")
        print(f"Train: {t_start} to {t_end}")
        print(f"Test:  {test_start} to {test_end}")
        print(f"{'='*60}")
        
        # Train on this window
        model = train_single_period(
            model=model,
            df=df,
            train_start=t_start,
            train_end=t_end,
            turbulence_threshold=turbulence_threshold,
            timesteps=timesteps_per_window,
            reset_num_timesteps=(i == 0),  # Only reset on first window
        )
        
        # Save model after each window
        model_path = os.path.join(save_dir, f"clstm_ppo_window_{i+1}.zip")
        model.save(model_path)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Paper: "and then traded on the test set for three months"
        # Perform out-sample testing for this window
        print(f"\nPerforming Out-Sample Test ({test_start} to {test_end})...")
        
        # Create test environment
        test_df, _ = prepare_features(df, test_start, test_end)
        
        # Use deterministic=True for testing (Parameter Freezing)
        portfolio_df, metrics = backtest(
            model=model,
            df=df,  # Pass full df, backtest handles slicing
            start_date=test_start,
            end_date=test_end,
            turbulence_threshold=turbulence_threshold,
            deterministic=True  # Freezing parameters (no exploration)
        )
        
        print(f"Window {i+1} Results:")
        print(f"  CR: {metrics['CR']*100:.2f}%")
        print(f"  SR: {metrics['SR']:.4f}")
        print(f"  Trades: {metrics['total_trades']}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "clstm_ppo_final.zip")
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    return model


def train_simple(
    df: pd.DataFrame,
    turbulence_threshold: float,
    timesteps: int = TOTAL_TIMESTEPS,
    save_dir: str = MODEL_DIR,
) -> RecurrentPPO:
    """
    Simple training without rolling window (for testing).
    
    Args:
        df: Stock data DataFrame
        turbulence_threshold: Turbulence threshold
        timesteps: Total training timesteps
        save_dir: Directory to save model
        
    Returns:
        Trained model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Simple Training (No Rolling Window)")
    print(f"{'='*60}")
    print(f"Train period: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    print(f"Timesteps: {timesteps:,}")
    
    # Create training environment
    train_env = create_train_env(
        df, TRAIN_START_DATE, TRAIN_END_DATE, turbulence_threshold
    )
    
    # Create model
    model = create_model(
        env=train_env,
        tensorboard_log=os.path.join(save_dir, "logs"),
    )
    
    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=TensorboardCallback(),
        progress_bar=True,
    )
    
    # Save model
    model_path = os.path.join(save_dir, "clstm_ppo_simple.zip")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    return model


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train CLSTM-PPO Stock Trading Agent")
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS})"
    )
    parser.add_argument(
        "--rolling", action="store_true",
        help="Use rolling window training strategy"
    )
    parser.add_argument(
        "--test_run", action="store_true",
        help="Quick test run with minimal timesteps"
    )
    parser.add_argument(
        "--save_dir", type=str, default=MODEL_DIR,
        help=f"Directory to save models (default: {MODEL_DIR})"
    )
    
    args = parser.parse_args()
    
    if args.test_run:
        args.timesteps = 1000
        print("Test run mode: using 1000 timesteps")
    
    print("=" * 60)
    print("CLSTM-PPO Stock Trading System - Training")
    print("=" * 60)
    
    # Fetch and prepare data
    print("\nStep 1: Fetching and preparing data...")
    df, turbulence_threshold = fetch_and_prepare_data()
    
    print(f"\nData summary:")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Stocks: {df['tic'].nunique()}")
    print(f"  Turbulence threshold: {turbulence_threshold:.2f}")
    
    # Train model
    print("\nStep 2: Training model...")
    
    if args.rolling:
        model = train_with_rolling_window(
            df=df,
            turbulence_threshold=turbulence_threshold,
            timesteps_per_window=args.timesteps // 5,  # Divide by number of windows
            save_dir=args.save_dir,
        )
    else:
        model = train_simple(
            df=df,
            turbulence_threshold=turbulence_threshold,
            timesteps=args.timesteps,
            save_dir=args.save_dir,
        )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
