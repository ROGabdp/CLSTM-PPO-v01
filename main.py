"""
Main entry point for CLSTM-PPO Stock Trading System.

This is a unified entry point for running training and backtesting.

Usage:
    python main.py train [--timesteps N] [--rolling] [--test_run]
    python main.py backtest [--model_path PATH] [--start_date DATE] [--end_date DATE]
    python main.py test   # Quick test to verify setup
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'train':
        from train import main as train_main
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove 'train' from args
        train_main()
        
    elif command == 'backtest':
        from backtest import main as backtest_main
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove 'backtest' from args
        backtest_main()
        
    elif command == 'test':
        run_quick_test()
        
    elif command in ['help', '-h', '--help']:
        print_help()
        
    else:
        print(f"Unknown command: {command}")
        print_help()


def print_help():
    """Print help message."""
    help_text = """
================================================================================
CLSTM-PPO Stock Trading System
================================================================================

Based on: "A Novel Deep Reinforcement Learning Based Automated Stock Trading 
System Using Cascaded LSTM Networks" (Zou et al., 2023)

Usage:
    python main.py <command> [options]

Commands:
    train       Train the CLSTM-PPO model
    backtest    Backtest a trained model
    test        Run quick tests to verify setup
    help        Show this help message

Training Options:
    --timesteps N       Total training timesteps (default: 100000)
    --rolling           Use rolling window training strategy
    --test_run          Quick test run with 1000 timesteps
    --save_dir PATH     Directory to save models

Backtest Options:
    --model_path PATH   Path to trained model
    --start_date DATE   Backtest start date (default: 2016-01-01)
    --end_date DATE     Backtest end date (default: 2020-05-08)
    --save_dir PATH     Directory to save results
    --no_plot           Skip plotting equity curves

Examples:
    # Run quick test
    python main.py test
    
    # Train with default settings
    python main.py train
    
    # Train with rolling window and 500k timesteps
    python main.py train --rolling --timesteps 500000
    
    # Quick training test
    python main.py train --test_run
    
    # Backtest trained model
    python main.py backtest --model_path models_saved/clstm_ppo_final.zip
    
================================================================================
"""
    print(help_text)


def run_quick_test():
    """Run quick tests to verify the setup."""
    print("=" * 60)
    print("CLSTM-PPO Quick Test")
    print("=" * 60)
    
    errors = []
    
    # Test 1: Import modules
    print("\n[Test 1] Importing modules...")
    try:
        from config import DOW_30_TICKERS, PPO_PARAMS
        print(f"  ✓ config.py loaded ({len(DOW_30_TICKERS)} tickers)")
    except Exception as e:
        errors.append(f"  ✗ config.py: {e}")
        print(errors[-1])
    
    try:
        from data_fetcher import fetch_and_prepare_data
        print("  ✓ data_fetcher.py loaded")
    except Exception as e:
        errors.append(f"  ✗ data_fetcher.py: {e}")
        print(errors[-1])
    
    try:
        from envs.multi_stock_trading_env import MultiStockTradingEnv
        print("  ✓ multi_stock_trading_env.py loaded")
    except Exception as e:
        errors.append(f"  ✗ multi_stock_trading_env.py: {e}")
        print(errors[-1])
    
    try:
        from models.lstm_feature_extractor import LSTMFeatureExtractor
        print("  ✓ lstm_feature_extractor.py loaded")
    except Exception as e:
        errors.append(f"  ✗ lstm_feature_extractor.py: {e}")
        print(errors[-1])
    
    # Test 2: Check dependencies
    print("\n[Test 2] Checking dependencies...")
    required_packages = [
        'stable_baselines3',
        'sb3_contrib', 
        'gymnasium',
        'yfinance',
        'torch',
        'pandas',
        'numpy',
        'ta',
        'matplotlib',
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            errors.append(f"  ✗ {package}: {e}")
            print(errors[-1])
    
    # Test 3: RecurrentPPO
    print("\n[Test 3] Testing RecurrentPPO...")
    try:
        from sb3_contrib import RecurrentPPO
        print("  ✓ RecurrentPPO available")
    except Exception as e:
        errors.append(f"  ✗ RecurrentPPO: {e}")
        print(errors[-1])
    
    # Test 4: LSTM Feature Extractor
    print("\n[Test 4] Testing LSTM Feature Extractor...")
    try:
        import torch
        import numpy as np
        from gymnasium import spaces
        from models.lstm_feature_extractor import LSTMFeatureExtractor
        from config import STATE_DIM, LSTM_WINDOW_SIZE, LSTM_FEATURE_DIM
        
        obs_dim = LSTM_WINDOW_SIZE * STATE_DIM
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        extractor = LSTMFeatureExtractor(observation_space=obs_space)
        dummy_obs = torch.randn(4, obs_dim)
        features = extractor(dummy_obs)
        
        assert features.shape == (4, LSTM_FEATURE_DIM), f"Expected (4, {LSTM_FEATURE_DIM}), got {features.shape}"
        print(f"  ✓ Feature extraction works: input={dummy_obs.shape} -> output={features.shape}")
    except Exception as e:
        errors.append(f"  ✗ LSTM Feature Extractor: {e}")
        print(errors[-1])
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"Tests completed with {len(errors)} error(s)")
        for error in errors:
            print(error)
        print("\nPlease fix the errors above before training.")
    else:
        print("All tests passed! ✓")
        print("\nYou can now run:")
        print("  python main.py train --test_run")
    print("=" * 60)


if __name__ == "__main__":
    main()
