"""
LSTM Feature Extractor for CLSTM-PPO.

This module implements the first LSTM layer that extracts time-series features
from the observation window before feeding to the PPO policy.

Paper Reference: Algorithm 1 (Line 722-766), Figure 3, Section 3.2.2

Architecture (Line 876-883):
- Input: [T=30 days × 181 features]
- LSTM Layer: input=181, hidden=128
- 3 Linear Layers: (input_dim, 128), (128, 128), (128, 128) with Tanh
- Output: 128-dim feature vector
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from config import (
    STATE_DIM,
    LSTM_WINDOW_SIZE,
    LSTM_FEATURE_DIM,
    LSTM_HIDDEN_SIZE_EXTRACTOR,
)


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for time-series data.
    
    This implements Algorithm 1 from the paper:
    1. Takes last T-day state sequences
    2. Passes through LSTM to capture temporal patterns
    3. Applies 3 linear layers with Tanh activation
    4. Outputs feature vector for PPO agent
    
    Paper Reference:
    - Algorithm 1 (one-day LSTM feature extractor)
    - Figure 3 (network architecture)
    - Line 876-883 (network dimensions)
    """
    
    def __init__(
        self, 
        observation_space: spaces.Box,
        features_dim: int = LSTM_FEATURE_DIM,
        window_size: int = LSTM_WINDOW_SIZE,
        state_dim: int = STATE_DIM,
        lstm_hidden_size: int = LSTM_HIDDEN_SIZE_EXTRACTOR,
    ):
        """
        Initialize the LSTM feature extractor.
        
        Args:
            observation_space: Gym observation space
            features_dim: Output feature dimension (default: 128)
            window_size: Time window size T (default: 30)
            state_dim: Single step state dimension (default: 181)
            lstm_hidden_size: LSTM hidden size (default: 128)
        """
        super().__init__(observation_space, features_dim)
        
        self.window_size = window_size
        self.state_dim = state_dim
        self.lstm_hidden_size = lstm_hidden_size
        self._output_dim = features_dim  # Use different name to avoid conflict with parent property
        
        # Verify observation space dimension
        expected_obs_dim = window_size * state_dim
        actual_obs_dim = np.prod(observation_space.shape)
        
        if actual_obs_dim != expected_obs_dim:
            # Adjust state_dim based on actual observation
            self.state_dim = actual_obs_dim // window_size
            print(f"Adjusted state_dim to {self.state_dim} based on observation space")
        
        # LSTM layer for time-series feature extraction
        # Paper: "LSTM Layer (input=181, hidden=128)"
        self.lstm = nn.LSTM(
            input_size=self.state_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        
        # Three linear layers with Tanh activation
        # Paper (Line 878-883):
        # "Linear layer 1 is (15×128, 128) and then passes the Tanh activation function.
        #  Linear layer2 and 3 are the same two layers of (128, 128), and then pass Tanh."
        # Note: We use lstm_hidden_size as input to first linear layer
        self.linear1 = nn.Linear(self.lstm_hidden_size, self._output_dim)
        self.linear2 = nn.Linear(self._output_dim, self._output_dim)
        self.linear3 = nn.Linear(self._output_dim, self._output_dim)
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for linear in [self.linear1, self.linear2, self.linear3]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observation sequence.
        
        Algorithm 1 implementation:
        1. Get last N-day states list
        2. Initialize LSTM hidden and cell states
        3. For each state, pass through LSTM
        4. Extract features from last LSTM output
        5. Pass through linear layers with Tanh
        
        Args:
            observations: Flattened observations of shape (batch, window_size * state_dim)
            
        Returns:
            Features of shape (batch, features_dim)
        """
        batch_size = observations.shape[0]
        
        # Reshape to (batch, window_size, state_dim)
        x = observations.view(batch_size, self.window_size, self.state_dim)
        
        # Normalize input for stability
        x = self._normalize_input(x)
        
        # Pass through LSTM
        # lstm_out: (batch, window_size, lstm_hidden_size)
        # h_n: (1, batch, lstm_hidden_size) - final hidden state
        # c_n: (1, batch, lstm_hidden_size) - final cell state
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last timestep's output (current state features)
        # Paper: "Extract features from the last LSTM layer"
        features = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)
        
        # Pass through three linear layers with Tanh activation
        features = self.tanh(self.linear1(features))
        features = self.tanh(self.linear2(features))
        features = self.tanh(self.linear3(features))
        
        return features
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input for training stability.
        
        Args:
            x: Input tensor of shape (batch, window_size, state_dim)
            
        Returns:
            Normalized tensor
        """
        # Simple standardization per feature
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        return (x - mean) / std


class LSTMFeatureExtractorWithMemory(LSTMFeatureExtractor):
    """
    LSTM Feature Extractor with persistent hidden state memory.
    
    This variant maintains LSTM hidden states across episodes,
    allowing the model to remember longer-term patterns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_state = None
        self.cell_state = None
    
    def reset_memory(self):
        """Reset LSTM hidden states."""
        self.hidden_state = None
        self.cell_state = None
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward with persistent memory."""
        batch_size = observations.shape[0]
        
        # Reshape to (batch, window_size, state_dim)
        x = observations.view(batch_size, self.window_size, self.state_dim)
        x = self._normalize_input(x)
        
        # Initialize or resize hidden states if needed
        if self.hidden_state is None or self.hidden_state.shape[1] != batch_size:
            device = observations.device
            self.hidden_state = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
            self.cell_state = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        
        # Pass through LSTM with persistent state
        lstm_out, (h_n, c_n) = self.lstm(x, (self.hidden_state.detach(), self.cell_state.detach()))
        
        # Update hidden states
        self.hidden_state = h_n
        self.cell_state = c_n
        
        # Use last timestep output
        features = lstm_out[:, -1, :]
        
        # Linear layers with Tanh
        features = self.tanh(self.linear1(features))
        features = self.tanh(self.linear2(features))
        features = self.tanh(self.linear3(features))
        
        return features


if __name__ == "__main__":
    # Test the feature extractor
    print("=" * 60)
    print("LSTM Feature Extractor Test")
    print("=" * 60)
    
    # Create dummy observation space
    window_size = LSTM_WINDOW_SIZE
    state_dim = STATE_DIM
    obs_dim = window_size * state_dim
    
    observation_space = spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(obs_dim,), 
        dtype=np.float32
    )
    
    # Create feature extractor
    print(f"\nCreating LSTM Feature Extractor...")
    print(f"  Window size: {window_size}")
    print(f"  State dim: {state_dim}")
    print(f"  Observation dim: {obs_dim}")
    
    extractor = LSTMFeatureExtractor(
        observation_space=observation_space,
        features_dim=LSTM_FEATURE_DIM,
        window_size=window_size,
        state_dim=state_dim,
    )
    
    print(f"\nNetwork architecture:")
    print(extractor)
    
    # Count parameters
    total_params = sum(p.numel() for p in extractor.parameters())
    trainable_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 16
    dummy_obs = torch.randn(batch_size, obs_dim)
    
    features = extractor(dummy_obs)
    print(f"Input shape: {dummy_obs.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected output shape: ({batch_size}, {LSTM_FEATURE_DIM})")
    
    assert features.shape == (batch_size, LSTM_FEATURE_DIM), "Output shape mismatch!"
    
    print("\nLSTM Feature Extractor test completed successfully!")
