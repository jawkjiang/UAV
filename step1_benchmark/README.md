# GPS Spoofing Detection for UAV Systems

## Overview

This project implements a **GPS spoofing detection model** for UAV systems using deep learning. The model detects position-level GPS spoofing attacks through analysis of sensor consistency patterns in flight data.

## Key Features

✅ **Position-level step-type GPS spoofing injection**
- Step position offset with exponentially decaying velocity transient
- Derived acceleration transient based on real time deltas
- No modification to orientation fields

✅ **Strict data handling**
- Flight-based train/val/test split (70/15/15)
- No time interpolation (uses real irregular sampling)
- Windows cannot cross flight boundaries
- Controlled positive sample ratios (train: 30-40%, val/test: 1-5%)

✅ **Advanced feature engineering**
- Position-velocity consistency residuals
- Velocity-acceleration consistency residuals
- All dynamics computed with real Δt

✅ **1D-CNN model**
- Temporal convolutional architecture
- Weighted binary cross entropy loss
- Early stopping with validation monitoring

✅ **Comprehensive evaluation**
- PR-AUC and ROC-AUC
- Recall @ fixed FPR (1%, 5%, 10%)
- Average false positives per flight

## Project Structure

```
UAV/
├── data/
│   └── src/
│       └── flights.csv          # Input flight data
├── output/                       # Generated outputs
│   ├── flight_splits.json
│   ├── train_attack_info.csv
│   ├── test_metrics.json
│   └── evaluation_curves.png
├── models/                       # Saved models
│   └── best_model.pth
├── config.py                     # Configuration parameters
├── data_loader.py                # Data loading and splitting
├── attack_injector.py            # GPS spoofing injection
├── labeling.py                   # Label generation
├── feature_engineering.py        # Feature computation
├── window_creation.py            # Window slicing and dataset
├── model.py                      # 1D-CNN architecture
├── training.py                   # Training pipeline
├── evaluation.py                 # Evaluation metrics
├── main.py                       # Main execution script
└── requirements.txt              # Dependencies
```

## Installation

### 1. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline:

```bash
python main.py
```

This will:
1. Load and split flight data by flight ID
2. Inject GPS spoofing attacks with controlled ratios
3. Compute consistency residual features
4. Create sliding windows (50 samples, step=5)
5. Train 1D-CNN model with weighted loss
6. Evaluate on test set and save results

### Configuration

Edit [config.py](config.py) to customize:

- **Data paths**: `DATA_PATH`, `OUTPUT_DIR`, `MODEL_DIR`
- **Split ratios**: `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`
- **Window parameters**: `WINDOW_SIZE`, `STEP_SIZE`
- **Attack parameters**: `ATTACK_MAGNITUDES`, `ATTACK_DURATION`
- **Model hyperparameters**: `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`
- **Random seed**: `RANDOM_SEED` for reproducibility

## Attack Injection Details

### Position Offset

For all samples at or after attack start time `t_s`:

```
p̃(t) = p(t) + Δp
```

Where:
- `||Δp||` ∈ {5m, 15m, 30m} (random)
- Direction: random 2D unit vector

### Velocity Transient

Within `[t_s, t_s + 3.0s]`:

```
Δv(t) = v_max · d̂ · exp(-(t-t_s)/1.0)
```

Where:
- `v_max` ∈ [0.5, 2.0] m/s (random)
- `d̂`: offset direction unit vector
- Uses real `Δt` from data

### Acceleration Transient

Within `[t_s, t_s + 3.0s]`:

```
Δa(t) = (Δv(t) - Δv(t-Δt)) / Δt
```

Added to `linear_acceleration_x/y`.

## Feature Engineering

### Basic Features (7)
- `position_x`, `position_y`
- `velocity_x`, `velocity_y`
- `linear_acceleration_x`, `linear_acceleration_y`
- `delta_t` (real time delta)

### Consistency Residual Features (6)

**Position-Velocity Residuals:**
```
r_pv = |Δp - v·Δt|
```
Computes: `residual_pv_x`, `residual_pv_y`, `residual_pv_norm`

**Velocity-Acceleration Residuals:**
```
r_va = |(v_t - v_{t-1})/Δt - a_t|
```
Computes: `residual_va_x`, `residual_va_y`, `residual_va_norm`

**Total: 13 features**

## Model Architecture

### 1D-CNN (Default)

```
Input: [batch, 50, 13]
  ↓
Conv1D(13→64, k=7) + BN + ReLU + Pool
  ↓
Conv1D(64→128, k=5) + BN + ReLU + Pool
  ↓
Conv1D(128→256, k=3) + BN + ReLU
  ↓
Conv1D(256→128, k=3) + BN + ReLU
  ↓
Global Avg/Max Pooling
  ↓
FC(256→64) + BN + ReLU + Dropout
  ↓
FC(64→32) + BN + ReLU + Dropout
  ↓
FC(32→1) + Sigmoid
  ↓
Output: [batch, 1] (spoofing probability)
```

### Alternative: TCN

Temporal Convolutional Network with dilated convolutions. Enable in [main.py](main.py):

```python
model = create_model(model_type='tcn', ...)
```

## Training

- **Loss**: Weighted Binary Cross Entropy
  - Positive weight = n_negative / n_positive
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: ReduceLROnPlateau (patience=5)
- **Early Stopping**: patience=10 epochs on validation PR-AUC
- **Batch Size**: 64

## Evaluation Metrics

### 1. PR-AUC (Primary)
Area under Precision-Recall curve

### 2. Recall @ Fixed FPR
Recall at FPR = 1%, 5%, 10%

### 3. Average False Positives per Flight
Mean number of false positive detections per flight

### 4. Standard Classification Metrics
Precision, Recall, F1-Score at threshold=0.5

## Output Files

After running `main.py`, the following files are generated:

### `output/`
- `flight_splits.json`: Train/val/test flight IDs
- `train_attack_info.csv`: Attack injection details for training flights
- `val_attack_info.csv`: Attack injection details for validation flights
- `test_attack_info.csv`: Attack injection details for test flights
- `normalization_stats.json`: Feature normalization parameters
- `training_history.json`: Loss and PR-AUC per epoch
- `test_metrics.json`: Complete evaluation metrics
- `test_predictions.npz`: Predictions and targets on test set
- `evaluation_curves.png`: PR and ROC curves

### `models/`
- `best_model.pth`: Best model checkpoint (highest val PR-AUC)

## Reproducibility

All random operations use fixed seeds:

- NumPy: `np.random.seed(RANDOM_SEED)`
- PyTorch: `torch.manual_seed(RANDOM_SEED)`
- PyTorch CUDA: `torch.cuda.manual_seed_all(RANDOM_SEED)`

Set `RANDOM_SEED` in [config.py](config.py) to ensure reproducible results.

## Data Requirements

Input data (`flights.csv`) must contain:

**Required columns:**
- `flight`: Flight identifier (integer)
- `time`: Timestamp (float, irregular sampling)
- `position_x`, `position_y`: Position in meters
- `velocity_x`, `velocity_y`: Velocity in m/s
- `linear_acceleration_x`, `linear_acceleration_y`: Acceleration in m/s²

**Optional columns:**
- `orientation_x/y/z/w`: Quaternion (not modified by attacks)
- Other sensor data (not used as features)

**Forbidden as input feature:**
- `route`: Flight route identifier

## Performance Tips

1. **GPU Acceleration**: Install PyTorch with CUDA support for faster training
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Batch Size**: Increase `BATCH_SIZE` if GPU memory allows

3. **Parallel Data Loading**: Modify DataLoader:
   ```python
   DataLoader(..., num_workers=4, pin_memory=True)
   ```

4. **Mixed Precision Training**: Use `torch.cuda.amp` for faster training

## Troubleshooting

### Issue: "Flight too short" warnings
**Solution**: Some flights may be shorter than `ATTACK_START_BUFFER + ATTACK_END_BUFFER`. These are automatically skipped.

### Issue: Low positive sample ratio
**Solution**: Adjust `attack_ratio` in attack injection or modify `balance_windows()` target ratios.

### Issue: Model underfitting
**Solution**: 
- Increase model capacity (more channels)
- Decrease dropout
- Increase training epochs

### Issue: Model overfitting
**Solution**:
- Increase dropout
- Add more regularization
- Reduce model complexity

## Citation

If you use this code, please cite:

```
GPS Spoofing Detection for UAV Systems using 1D-CNN
Position-level Attack Injection with Consistency Residual Features
2024
```

## License

MIT License

## Contact

For questions or issues, please open an issue on the repository.
