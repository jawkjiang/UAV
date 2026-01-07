# STRICT IMPLEMENTATION SPECIFICATION: Multi-Model GPS Spoofing Detection

## PROJECT STRUCTURE

```
step3_multiModel/
├── config_step3.py          # COPY from step2, NO modifications
├── multi_attack_injector.py # SYMLINK or COPY from step2, NO modifications
├── model.py                 # NEW FILE - All model implementations
├── main.py                  # NEW FILE - Multi-model multi-attack runner
├── compare_models.py        # NEW FILE - Model comparison analysis
└── output/                  # Auto-generated results
    ├── {attack_type}/
    │   ├── {model_type}/
    │   │   ├── best_model.pth
    │   │   ├── test_metrics.json
    │   │   ├── training_history.json
    │   │   └── test_predictions.npz
    │   └── model_comparison.csv
    └── overall_comparison.csv
```

---

## CRITICAL CONSTRAINTS - MUST NOT VIOLATE

### 🚫 FORBIDDEN MODIFICATIONS

**The following components are IMMUTABLE:**

1. **Data Processing Pipeline**
   ```python
   # From step1_benchmark - DO NOT MODIFY
   - load_flights_data()
   - split_flights()
   - get_flight_subset()
   - compute_delta_t()
   - convert_to_local_coordinates()
   ```

2. **Feature Engineering**
   ```python
   # From step1_benchmark - DO NOT MODIFY
   - compute_all_features()
   - get_feature_columns()
   - normalize_features()
   - compute_position_velocity_residual()
   - compute_velocity_acceleration_residual()
   ```

3. **Attack Injection**
   ```python
   # From step2_multiAttack - DO NOT MODIFY
   - MultiAttackInjector class (entire file)
   - inject_attack_to_flight()
   - inject_attacks_to_dataset()
   - All attack types: step, drift, delay, replay, takeover
   ```

4. **Labeling**
   ```python
   # From step1_benchmark - DO NOT MODIFY
   - generate_point_labels()
   ```

5. **Window Creation**
   ```python
   # From step1_benchmark - DO NOT MODIFY
   - create_windows_from_dataset()
   - balance_windows()
   - WindowDataset class
   ```

6. **Training Infrastructure**
   ```python
   # From step1_benchmark - DO NOT MODIFY
   - train_model() function signature and core logic
   - Loss function: BCELoss with pos_weight
   - Optimizer: Adam
   - Learning rate scheduling: ReduceLROnPlateau
   - Early stopping logic
   ```

7. **Evaluation Infrastructure**
   ```python
   # From step1_benchmark - DO NOT MODIFY
   - evaluate_model() function
   - All metrics: AUC-ROC, AUC-PR, F1, Precision, Recall
   - Per-flight aggregation logic
   - Confusion matrix computation
   ```

8. **Configuration Parameters**
   ```python
   # From step2_multiAttack/config_step2.py - DO NOT MODIFY
   WINDOW_SIZE = 50
   STEP_SIZE = 5
   BATCH_SIZE = 64
   LEARNING_RATE = 0.001
   NUM_EPOCHS = 50
   EARLY_STOPPING_PATIENCE = 10
   CONSISTENCY_MODES = ['pos_vel_acc']  # Unified setting
   # ... all attack parameters ...
   ```

---

## ✅ ALLOWED MODIFICATIONS - ONLY THIS

### MODEL IMPLEMENTATION ONLY

**File: `step3_multiModel/model.py`**

You are ONLY allowed to:
1. Define new model classes
2. Modify the `create_model()` factory function
3. Add model-specific helper functions

**All models MUST conform to this EXACT interface:**

```python
class YourModelName(nn.Module):
    """
    Model description.
    
    REQUIRED INTERFACE - DO NOT CHANGE
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        """
        REQUIRED SIGNATURE - DO NOT CHANGE PARAMETER NAMES OR ORDER
        
        Args:
            n_features: Number of input features (always 13)
            window_size: Window size (always 50)
            dropout: Dropout probability (default 0.3)
        """
        super(YourModelName, self).__init__()
        
        # Your model architecture here
        # ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        REQUIRED SIGNATURE - DO NOT CHANGE
        
        Args:
            x: Input tensor of shape [batch_size, window_size, n_features]
               = [batch_size, 50, 13]
        
        Returns:
            Output tensor of shape [batch_size, 1]
            Values MUST be in range [0, 1] (use sigmoid)
        """
        # Your forward pass here
        # ...
        
        return output  # shape: [batch_size, 1], range: [0, 1]
```

---

## MANDATORY MODEL SPECIFICATIONS

### INPUT/OUTPUT CONTRACT

**Input Specification (IMMUTABLE):**
```python
# Input tensor
shape: [batch_size, 50, 13]
dtype: torch.float32

# Feature order (DO NOT REORDER):
features = [
    'position_x',              # 0
    'position_y',              # 1
    'velocity_x',              # 2
    'velocity_y',              # 3
    'linear_acceleration_x',   # 4
    'linear_acceleration_y',   # 5
    'delta_t',                 # 6
    'residual_pv_x',          # 7
    'residual_pv_y',          # 8
    'residual_pv_norm',       # 9
    'residual_va_x',          # 10
    'residual_va_y',          # 11
    'residual_va_norm'        # 12
]

# Features are NORMALIZED (mean=0, std=1)
```

**Output Specification (IMMUTABLE):**
```python
# Output tensor
shape: [batch_size, 1]
dtype: torch.float32
range: [0.0, 1.0]  # MUST use sigmoid activation

# Interpretation:
# output[i, 0] = probability that the window ending at sample i is under attack
```

---

## REQUIRED MODEL IMPLEMENTATIONS

### 1. BASELINE: 1D-CNN (Already Implemented - Copy from step1)

```python
class GPSSpoofingDetector(nn.Module):
    # COPY EXACTLY from step1_benchmark/model.py
    # Lines 34-139
    pass
```

### 2. LSTM

```python
class LSTMDetector(nn.Module):
    """
    LSTM-based detector.
    
    Architecture:
    Input: [batch, 50, 13]
      ↓
    LSTM(hidden_size=128, num_layers=2, dropout=dropout)
      ↓
    Take last hidden state: [batch, 128]
      ↓
    FC: 128 → 64 → 32 → 1
      ↓
    Sigmoid → [batch, 1]
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        super(LSTMDetector, self).__init__()
        
        self.hidden_size = 128
        self.num_layers = 2
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 50, 13]
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch, 50, 128]
        # h_n: [num_layers, batch, 128]
        
        # Take last hidden state from last layer
        last_hidden = h_n[-1]  # [batch, 128]
        
        # FC layers
        x = self.fc1(last_hidden)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x
```

### 3. BiLSTM

```python
class BiLSTMDetector(nn.Module):
    """
    Bidirectional LSTM detector.
    
    Architecture:
    Input: [batch, 50, 13]
      ↓
    BiLSTM(hidden_size=64, num_layers=2)
      ↓
    Concat forward & backward: [batch, 128]
      ↓
    FC: 128 → 64 → 32 → 1
      ↓
    Sigmoid → [batch, 1]
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        super(BiLSTMDetector, self).__init__()
        
        self.hidden_size = 64  # Per direction
        self.num_layers = 2
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0,
            bidirectional=True  # KEY: bidirectional
        )
        
        # FC layers (input is hidden_size * 2 due to bidirectional)
        self.fc1 = nn.Linear(self.hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 50, 13]
        
        # BiLSTM forward
        lstm_out, (h_n, c_n) = self.bilstm(x)
        # h_n: [num_layers * 2, batch, 64]
        
        # Concatenate forward and backward from last layer
        # h_n[-2]: forward direction last layer
        # h_n[-1]: backward direction last layer
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, 128]
        
        # FC layers
        x = self.fc1(last_hidden)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x
```

### 4. GRU

```python
class GRUDetector(nn.Module):
    """
    GRU-based detector.
    
    Similar to LSTM but more efficient.
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        super(GRUDetector, self).__init__()
        
        self.hidden_size = 128
        self.num_layers = 2
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0
        )
        
        # FC layers
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 50, 13]
        
        # GRU forward
        gru_out, h_n = self.gru(x)
        # h_n: [num_layers, batch, 128]
        
        # Take last hidden state
        last_hidden = h_n[-1]  # [batch, 128]
        
        # FC layers
        x = self.fc1(last_hidden)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x
```

### 5. CNN-LSTM Hybrid

```python
class CNNLSTMDetector(nn.Module):
    """
    Hybrid CNN-LSTM detector.
    
    Architecture:
    Input: [batch, 50, 13]
      ↓
    1D Conv layers (extract local features)
      ↓  [batch, seq_len', 128]
    LSTM (model temporal dependencies)
      ↓  [batch, 128]
    FC → Sigmoid
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        super(CNNLSTMDetector, self).__init__()
        
        # CNN feature extractor
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # After 2 pooling layers: seq_len = 50 / 4 = 12
        # LSTM on extracted features
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0
        )
        
        # FC layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.bn_fc = nn.BatchNorm1d(64)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 50, 13]
        
        # Transpose for Conv1d: [batch, 13, 50]
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # [batch, 64, 25]
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # [batch, 128, 12]
        
        # Transpose back for LSTM: [batch, 12, 128]
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # [batch, 128]
        
        # FC layers
        x = self.fc1(last_hidden)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x
```

### 6. TCN (Already Implemented - Copy from step1)

```python
class TemporalConvNet(nn.Module):
    # COPY EXACTLY from step1_benchmark/model.py
    # Lines 142-201
    pass
```

### 7. Transformer

```python
class TransformerDetector(nn.Module):
    """
    Transformer-based detector.
    
    Architecture:
    Input: [batch, 50, 13]
      ↓
    Linear projection to d_model
      ↓
    Positional encoding
      ↓
    Transformer encoder (4 heads, 3 layers)
      ↓
    Global average pooling
      ↓
    FC → Sigmoid
    """
    
    def __init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3):
        super(TransformerDetector, self).__init__()
        
        self.d_model = 64
        self.nhead = 4
        self.num_layers = 3
        self.dim_feedforward = 128
        
        # Input projection
        self.input_projection = nn.Linear(n_features, self.d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(window_size, self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(self.d_model, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(32)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 50, 13]
        
        # Project to d_model
        x = self.input_projection(x)  # [batch, 50, 64]
        
        # Add positional encoding
        batch_size = x.size(0)
        pos_enc = self.pos_encoding.to(x.device)
        x = x + pos_enc  # Broadcasting
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch, 50, 64]
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch, 64]
        
        # Output layers
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x
```

### 8. REQUIRED: create_model() Factory Function

```python
def create_model(model_type: str, 
                n_features: int, 
                window_size: int = 50, 
                dropout: float = 0.3) -> nn.Module:
    """
    Factory function to create model instances.
    
    REQUIRED SIGNATURE - DO NOT CHANGE
    
    Args:
        model_type: One of ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
        n_features: Number of input features (always 13)
        window_size: Window size (always 50)
        dropout: Dropout probability
    
    Returns:
        Model instance conforming to required interface
    """
    models = {
        'cnn': GPSSpoofingDetector,
        'lstm': LSTMDetector,
        'bilstm': BiLSTMDetector,
        'gru': GRUDetector,
        'cnn_lstm': CNNLSTMDetector,
        'tcn': TemporalConvNet,
        'transformer': TransformerDetector
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Must be one of {list(models.keys())}")
    
    return models[model_type](n_features, window_size, dropout)
```

---

## IMPLEMENTATION CHECKLIST

### Pre-Implementation Verification

- [ ] Have you copied `step1_benchmark/data_loader.py` → `step3_multiModel/`?
- [ ] Have you copied `step1_benchmark/labeling.py` → `step3_multiModel/`?
- [ ] Have you copied `step1_benchmark/feature_engineering.py` → `step3_multiModel/`?
- [ ] Have you copied `step1_benchmark/window_creation.py` → `step3_multiModel/`?
- [ ] Have you copied `step1_benchmark/training.py` → `step3_multiModel/`?
- [ ] Have you copied `step1_benchmark/evaluation.py` → `step3_multiModel/`?
- [ ] Have you copied or symlinked `step2_multiAttack/multi_attack_injector.py` → `step3_multiModel/`?
- [ ] Have you copied `step2_multiAttack/config_step2.py` → `step3_multiModel/config_step3.py`?

### Model Implementation Verification

For EACH model class:

- [ ] Constructor signature exactly matches: `__init__(self, n_features: int, window_size: int = 50, dropout: float = 0.3)`
- [ ] Forward signature exactly matches: `forward(self, x: torch.Tensor) -> torch.Tensor`
- [ ] Input shape verified: `[batch_size, 50, 13]`
- [ ] Output shape verified: `[batch_size, 1]`
- [ ] Output range verified: `[0.0, 1.0]` (using sigmoid)
- [ ] No hardcoded values (use parameters)
- [ ] Proper use of batch normalization and dropout

### Integration Verification

- [ ] `create_model()` function includes all 7 models
- [ ] All model types tested with dummy input: `torch.randn(4, 50, 13)`
- [ ] No import errors
- [ ] No shape mismatches

---

## MAIN EXECUTION SCRIPT STRUCTURE

**File: `step3_multiModel/main.py`**

```python
"""
Multi-Model Multi-Attack GPS Spoofing Detection

Tests all model architectures on all attack types.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import pandas as pd
from itertools import product

# Add step1_benchmark to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step1_benchmark'))

# Import from step1_benchmark (UNCHANGED)
from data_loader import load_flights_data, split_flights, get_flight_subset, compute_delta_t, convert_to_local_coordinates
from labeling import generate_point_labels
from feature_engineering import compute_all_features, get_feature_columns, normalize_features
from window_creation import create_windows_from_dataset, balance_windows, WindowDataset
from training import train_model
from evaluation import evaluate_model

# Import step3 specific
import config_step3 as config
from multi_attack_injector import MultiAttackInjector
from model import create_model  # NEW: supports all models


# Define experiment matrix
ATTACK_TYPES = [
    'step',
    'drift_ramp',
    'drift_sigmoid',
    'delay',
    'replay_same_hard',
    'replay_other_soft',
    'takeover_step',
    'takeover_ramp'
]

MODEL_TYPES = [
    'cnn',
    'lstm',
    'bilstm',
    'gru',
    'cnn_lstm',
    'tcn',
    'transformer'
]

# Attack-specific parameters
ATTACK_PARAMS = {
    'step': {},
    'drift_ramp': {'profile': 'ramp'},
    'drift_sigmoid': {'profile': 'sigmoid'},
    'delay': {},
    'replay_same_hard': {'donor_source': 'same_flight_earlier', 'stitching': 'hard'},
    'replay_other_soft': {'donor_source': 'same_route_other_flight', 'stitching': 'soft'},
    'takeover_step': {'offset_profile': 'step'},
    'takeover_ramp': {'offset_profile': 'ramp'}
}

# Map attack names to injection types
ATTACK_TYPE_MAP = {
    'step': 'step',
    'drift_ramp': 'drift',
    'drift_sigmoid': 'drift',
    'delay': 'delay',
    'replay_same_hard': 'replay',
    'replay_other_soft': 'replay',
    'takeover_step': 'takeover',
    'takeover_ramp': 'takeover'
}


def run_single_experiment(attack_name, model_type, base_df, train_flights, val_flights, test_flights, output_dir):
    """
    Run a single attack-model combination.
    
    CRITICAL: This function MUST NOT modify any data processing, training, or evaluation logic.
    ONLY the model creation is different.
    """
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {attack_name} + {model_type.upper()}")
    print(f"{'='*80}")
    
    # Create output directory
    exp_output_dir = os.path.join(output_dir, attack_name, model_type)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Check if experiment already done
    model_path = os.path.join(exp_output_dir, 'best_model.pth')
    if config.SKIP_EXISTING and os.path.exists(model_path):
        print(f"Skipping - already completed")
        metrics_path = os.path.join(exp_output_dir, 'test_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return {'attack': attack_name, 'model': model_type, 'metrics': metrics, 'skipped': True}
    
    # Initialize injector
    injector = MultiAttackInjector(random_seed=config.RANDOM_SEED)
    
    # Inject attacks
    print(f"Injecting {attack_name} attacks...")
    attack_type = ATTACK_TYPE_MAP[attack_name]
    attack_params = ATTACK_PARAMS[attack_name]
    
    train_df = get_flight_subset(base_df, train_flights)
    train_df, train_attack_info = injector.inject_attacks_to_dataset(
        train_df, attack_type=attack_type, attack_ratio=0.5, **attack_params
    )
    train_attack_info.to_csv(os.path.join(exp_output_dir, 'train_attack_info.csv'), index=False)
    
    val_df = get_flight_subset(base_df, val_flights)
    val_df, val_attack_info = injector.inject_attacks_to_dataset(
        val_df, attack_type=attack_type, attack_ratio=0.1, **attack_params
    )
    val_attack_info.to_csv(os.path.join(exp_output_dir, 'val_attack_info.csv'), index=False)
    
    test_df = get_flight_subset(base_df, test_flights)
    test_df, test_attack_info = injector.inject_attacks_to_dataset(
        test_df, attack_type=attack_type, attack_ratio=0.1, **attack_params
    )
    test_attack_info.to_csv(os.path.join(exp_output_dir, 'test_attack_info.csv'), index=False)
    
    # Generate labels
    print("Generating labels...")
    train_df = generate_point_labels(train_df, train_attack_info)
    val_df = generate_point_labels(val_df, val_attack_info)
    test_df = generate_point_labels(test_df, test_attack_info)
    
    # Feature engineering
    print("Computing features...")
    train_df = compute_all_features(train_df)
    val_df = compute_all_features(val_df)
    test_df = compute_all_features(test_df)
    
    feature_cols = get_feature_columns()
    train_df, val_df, test_df, normalization_stats = normalize_features(
        train_df, val_df, test_df, feature_cols
    )
    
    with open(os.path.join(exp_output_dir, 'normalization_stats.json'), 'w') as f:
        json.dump(normalization_stats, f, indent=2)
    
    # Create windows
    print("Creating windows...")
    train_windows, train_labels, train_flight_ids = create_windows_from_dataset(train_df, feature_cols)
    val_windows, val_labels, val_flight_ids = create_windows_from_dataset(val_df, feature_cols)
    test_windows, test_labels, test_flight_ids = create_windows_from_dataset(test_df, feature_cols)
    
    train_windows, train_labels, train_flight_ids = balance_windows(
        train_windows, train_labels, train_flight_ids,
        target_pos_ratio=0.35, random_seed=config.RANDOM_SEED
    )
    val_windows, val_labels, val_flight_ids = balance_windows(
        val_windows, val_labels, val_flight_ids,
        target_pos_ratio=0.03, random_seed=config.RANDOM_SEED
    )
    test_windows, test_labels, test_flight_ids = balance_windows(
        test_windows, test_labels, test_flight_ids,
        target_pos_ratio=0.03, random_seed=config.RANDOM_SEED
    )
    
    # Create dataloaders
    train_dataset = WindowDataset(train_windows, train_labels)
    val_dataset = WindowDataset(val_windows, val_labels)
    test_dataset = WindowDataset(test_windows, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Train model - ONLY DIFFERENCE IS MODEL TYPE
    print(f"Training {model_type} model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = create_model(
        model_type=model_type,  # <-- ONLY THIS CHANGES
        n_features=len(feature_cols),
        window_size=config.WINDOW_SIZE,
        dropout=0.3
    )
    
    pos_weight = train_dataset.get_positive_weight()
    
    history = train_model(
        model, train_loader, val_loader,
        pos_weight=pos_weight,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        patience=config.EARLY_STOPPING_PATIENCE,
        device=device,
        save_path=model_path
    )
    
    with open(os.path.join(exp_output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Evaluate
    print("Evaluating...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metrics, predictions, targets = evaluate_model(model, test_loader, test_flight_ids, device=device)
    
    with open(os.path.join(exp_output_dir, 'test_metrics.json'), 'w') as f:
        metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in metrics.items()}
        json.dump(metrics_serializable, f, indent=2)
    
    np.savez(os.path.join(exp_output_dir, 'test_predictions.npz'),
             predictions=predictions, targets=targets, flight_ids=test_flight_ids)
    
    print(f"Results: AUC-ROC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
    
    return {'attack': attack_name, 'model': model_type, 'metrics': metrics, 'skipped': False}


def main():
    """Run full experiment matrix."""
    
    print("="*80)
    print("Multi-Model Multi-Attack GPS Spoofing Detection")
    print("="*80)
    
    # Setup
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and split data (ONCE, shared across all experiments)
    print("\nLoading and preprocessing data...")
    df = load_flights_data(data_path=config.DATA_PATH)
    train_flights, val_flights, test_flights = split_flights(df)
    
    with open(os.path.join(output_dir, 'flight_splits.json'), 'w') as f:
        json.dump({
            'train': train_flights,
            'val': val_flights,
            'test': test_flights,
            'random_seed': config.RANDOM_SEED
        }, f, indent=2)
    
    df = compute_delta_t(df)
    df = convert_to_local_coordinates(df)
    
    # Run experiment matrix
    all_results = []
    total_experiments = len(ATTACK_TYPES) * len(MODEL_TYPES)
    current_experiment = 0
    
    for attack_name, model_type in product(ATTACK_TYPES, MODEL_TYPES):
        current_experiment += 1
        print(f"\n\n{'#'*80}")
        print(f"# EXPERIMENT {current_experiment}/{total_experiments}: {attack_name} + {model_type}")
        print(f"{'#'*80}")
        
        result = run_single_experiment(
            attack_name, model_type,
            df, train_flights, val_flights, test_flights,
            output_dir
        )
        all_results.append(result)
    
    # Generate comparison tables
    print("\n\nGenerating comparison tables...")
    
    # Overall comparison
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'attack_type': result['attack'],
            'model_type': result['model'],
            'auc_roc': result['metrics']['roc_auc'],
            'auc_pr': result['metrics']['pr_auc'],
            'f1_score': result['metrics']['f1'],
            'precision': result['metrics']['precision'],
            'recall': result['metrics']['recall']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, 'overall_comparison.csv'), index=False)
    
    print("\n\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Overall comparison: {os.path.join(output_dir, 'overall_comparison.csv')}")


if __name__ == '__main__':
    main()
```

---

## VALIDATION TESTS

Before running experiments, run these validation tests:

```python
# test_models.py

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step1_benchmark'))

from model import create_model

def test_model_interface():
    """Test that all models conform to interface."""
    
    model_types = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
    n_features = 13
    window_size = 50
    batch_size = 4
    
    for model_type in model_types:
        print(f"Testing {model_type}...")
        
        # Create model
        model = create_model(model_type, n_features, window_size, dropout=0.3)
        model.eval()
        
        # Test input
        x = torch.randn(batch_size, window_size, n_features)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Validate output shape
        assert output.shape == (batch_size, 1), \
            f"{model_type}: Expected shape ({batch_size}, 1), got {output.shape}"
        
        # Validate output range
        assert (output >= 0).all() and (output <= 1).all(), \
            f"{model_type}: Output not in [0, 1] range"
        
        print(f"  ✓ {model_type} passed")
    
    print("\n✓ All models passed interface validation")

if __name__ == '__main__':
    test_model_interface()
```

---

## CRITICAL SUCCESS CRITERIA

Your implementation MUST pass all these checks:

### 1. Interface Compliance
- [ ] All models accept `(batch, 50, 13)` input
- [ ] All models return `(batch, 1)` output in range `[0, 1]`
- [ ] All models have identical constructor signature

### 2. Data Integrity
- [ ] Same flight splits used across all experiments
- [ ] Same normalization statistics per attack type
- [ ] Same window creation logic
- [ ] Same balancing ratios

### 3. Training Consistency
- [ ] Same loss function (BCELoss with pos_weight)
- [ ] Same optimizer (Adam)
- [ ] Same learning rate schedule
- [ ] Same early stopping criteria

### 4. Evaluation Consistency
- [ ] Same metrics computed (AUC-ROC, AUC-PR, F1, etc.)
- [ ] Same per-flight aggregation
- [ ] Same threshold selection (max F1)

### 5. Output Structure
```
output/
├── {attack_type}/
│   ├── {model_type}/
│   │   ├── best_model.pth
│   │   ├── test_metrics.json
│   │   ├── training_history.json
│   │   ├── test_predictions.npz
│   │   ├── train_attack_info.csv
│   │   ├── val_attack_info.csv
│   │   └── test_attack_info.csv
├── flight_splits.json
└── overall_comparison.csv
```

---

## EXPECTED TIMELINE

- Model implementation: 2-3 hours
- Validation testing: 30 minutes
- Full experiment matrix (8 attacks × 7 models = 56 experiments): 8-12 hours (GPU)
- Analysis and visualization: 2 hours

**Total: ~1-2 days**

---

## FINAL CHECKLIST

Before declaring completion:

- [ ] All 7 models implemented and tested
- [ ] Validation test passes for all models
- [ ] Main script runs without errors
- [ ] All 56 experiments complete
- [ ] Output directory structure correct
- [ ] All metric files present
- [ ] No modifications to data processing pipeline
- [ ] No modifications to training/evaluation logic
- [ ] Comparison CSV generated successfully

---

## CONTACT FOR ISSUES

If you encounter ANY of the following, STOP and report:

1. Shape mismatch errors
2. Data processing differences between experiments
3. Missing imports from step1_benchmark
4. Inconsistent metrics across same attack-model combinations
5. Memory errors (models too large)

**DO NOT** attempt to fix by modifying:
- Data loading
- Feature engineering
- Window creation
- Training loop
- Evaluation metrics

These are IMMUTABLE components.

---

END OF SPECIFICATION
