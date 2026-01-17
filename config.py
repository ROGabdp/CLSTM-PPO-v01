"""
Configuration constants for CLSTM-PPO Stock Trading System.

Based on: "A Novel Deep Reinforcement Learning Based Automated Stock Trading System 
Using Cascaded LSTM Networks" (Zou et al., 2023)
"""

# =============================================================================
# DJI 30 Component Stocks (FinRL's validated list for 2009-2020)
# Reference: Yang[14] - Deep Reinforcement Learning for Automated Stock Trading
# These 30 stocks have complete data coverage for the paper's date range.
# =============================================================================
DOW_30_TICKERS = [
    'AXP',   # American Express
    'AAPL',  # Apple Inc.
    'BA',    # Boeing
    'CAT',   # Caterpillar
    'CSCO',  # Cisco Systems
    'CVX',   # Chevron
    'DD',    # DuPont (before DOW spinoff)
    'DIS',   # Walt Disney
    'GE',    # General Electric
    'GS',    # Goldman Sachs
    'HD',    # Home Depot
    'IBM',   # IBM
    'INTC',  # Intel
    'JNJ',   # Johnson & Johnson
    'JPM',   # JPMorgan Chase
    'KO',    # Coca-Cola
    'MCD',   # McDonald's
    'MMM',   # 3M
    'MRK',   # Merck
    'MSFT',  # Microsoft
    'NKE',   # Nike
    'PFE',   # Pfizer
    'PG',    # Procter & Gamble
    'TRV',   # Travelers
    'UNH',   # UnitedHealth
    'HON',   # Honeywell
    'V',     # Visa
    'VZ',    # Verizon
    'WMT',   # Walmart
    'XOM',   # ExxonMobil
]

# =============================================================================
# Time Periods (Paper Section 4.1, Line 1016-1017)
# =============================================================================
TRAIN_START_DATE = "2009-01-01"
TRAIN_END_DATE = "2015-12-31"
TEST_START_DATE = "2016-01-01"
TEST_END_DATE = "2020-05-08"

# Rolling window retraining interval (months)
ROLLING_WINDOW_MONTHS = 3

# =============================================================================
# Environment Parameters (Paper Section 3.1.5, Line 593-598)
# =============================================================================
INITIAL_CAPITAL = 1_000_000       # $1 million
HMAX = 100                        # Maximum shares per trade
TRANSACTION_COST_PCT = 0.001      # 0.1% transaction cost
REWARD_SCALING = 1e-4             # Reward scaling factor

# Turbulence threshold percentile (Paper Line 578-579)
TURBULENCE_THRESHOLD_PERCENTILE = 90

# =============================================================================
# Model Architecture Parameters
# =============================================================================
# LSTM Feature Extractor (Paper Line 876-883)
LSTM_WINDOW_SIZE = 30             # Time window T=30 (Table 2)
LSTM_FEATURE_DIM = 128            # Output feature dimension
LSTM_HIDDEN_SIZE_EXTRACTOR = 128  # Hidden size for feature extractor

# PPO LSTM Policy (Paper Table 3)
LSTM_HIDDEN_SIZE_PPO = 512        # Hidden size for PPO policy LSTM

# State space dimension
STATE_DIM = 181  # 1 (balance) + 30*6 (price, shares, MACD, RSI, CCI, ADX)

# =============================================================================
# PPO Hyperparameters (Paper Table 1, Line 1077-1089)
# =============================================================================
PPO_PARAMS = {
    'gamma': 0.99,                # Reward Discount Factor
    'n_steps': 128,               # Update Frequency
    'vf_coef': 0.5,               # Loss Function Weight of Critic
    'ent_coef': 0.01,             # Loss Function Weight of Distribution Entropy
    'clip_range': 0.2,            # Clip Range
    'max_grad_norm': 0.5,         # Maximum of Gradient Truncation
    'learning_rate': 3e-4,        # Learning Rate
}

# Adam optimizer parameters
ADAM_PARAMS = {
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
}

# =============================================================================
# Training Parameters
# =============================================================================
TOTAL_TIMESTEPS = 100_000         # Default training timesteps
SEED = 42                         # Random seed for reproducibility

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = "data"
MODEL_DIR = "models_saved"
RESULTS_DIR = "results"
