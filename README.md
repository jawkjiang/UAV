# UAV GPS Spoofing Detection

A research pipeline for detecting GPS spoofing attacks on UAV systems using deep learning.
The project spans 8 experimental steps, progressing from a baseline 1D-CNN to a multi-model,
time-aware evaluation framework with TimeGAN data augmentation.

## Project Structure

```
UAV/
├── data/src/                    # Raw input data (tracked)
│   ├── flights.csv              # UAV flight records
│   └── parameters.csv           # Flight parameters
├── step1_benchmark/             # Baseline 1D-CNN detector
├── step2_multiAttack/           # Multi-attack-type injection
├── step3_multiModel/            # Multi-model comparison (mixed attacks)
├── step3b_multiModelGeneral/    # Generalised multi-model training
├── step4_multiMetrics/          # Power-grid-inspired risk metrics
├── step4b_newMetrics/           # Time-aware metrics (MTBFA, detection delay)
├── step5_timegan/               # TimeGAN data augmentation
├── step6_newMetrcisWithTimegan/ # Time-aware metrics on TimeGAN-augmented data
├── step7_performance_analysis/  # Inference profiling & deployment analysis
├── step8_painter/               # Paper figure generation
├── styles/                      # Matplotlib style sheets
└── uav_comm_demo/               # MATLAB simulation demo
```

## Prerequisites

- Python >= 3.11
- PyTorch >= 1.10
- CUDA-capable GPU recommended for steps 3–6

Install dependencies with [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Each step is self-contained. Run steps in order because later steps consume outputs
from earlier ones. All outputs are written to the `output/` subdirectory of each step
(not tracked by git; regenerate by running the step).

### Step 1 — Baseline benchmark

```bash
cd step1_benchmark
python main.py
```

Trains a 1D-CNN on position-level GPS spoofing with a flight-based 70/15/15 split.
Outputs: model weights, evaluation metrics, PR/ROC curves.

### Step 2 — Multiple attack types

```bash
cd step2_multiAttack
python main.py
```

Extends step 1 to six GPS spoofing attack variants and compares detection performance.

### Step 3 — Multi-model comparison (mixed attacks)

```bash
cd step3_multiModel
python main.py
```

Trains LSTM, GRU, BiLSTM, 1D-CNN, CNN-LSTM, TCN, and Transformer on the mixed-attack
dataset and produces a side-by-side comparison.

### Step 3b — Generalised multi-model

```bash
cd step3b_multiModelGeneral
python main.py
```

Re-runs the seven models with a generalised mixed-attack injector for improved
cross-attack robustness.

### Step 4 — Power-grid risk metrics

```bash
cd step4_multiMetrics
python main.py
```

Adds power-grid-inspired risk scoring (criticality, exposure time) on top of step 3b outputs.

### Step 4b — Time-aware metrics

```bash
cd step4b_newMetrics
python main.py
```

Computes MTBFA (Mean Time Between False Alarms) and detection delay from step 3b
prediction outputs.

### Step 5 — TimeGAN data augmentation

Run the full pipeline (recommended):

```bash
cd step5_timegan
python run_all.py
```

Or run individual stages:

```bash
python prepare_normal_data.py      # extract normal flights from raw data
python train_timegan.py            # train the TimeGAN generator
python generate_synthetic_flights.py  # synthesise 5× normal flights
python merge_and_split.py          # merge & split train/val/test
python inject_attacks.py           # inject attacks at low ratio (15/10/5 %)
python main.py                     # feature engineering + model training
```

To train a specific model type:

```bash
python main.py --model lstm        # lstm | gru | bilstm | cnn | cnn_lstm | tcn | transformer
```

### Step 6 — Time-aware metrics on TimeGAN data

```bash
cd step6_newMetrcisWithTimegan
python main.py
```

Evaluates MTBFA and detection delay on the TimeGAN-augmented test set.

### Step 7 — Performance analysis

```bash
cd step7_performance_analysis
python main.py
```

Profiles inference latency, model complexity, and produces deployment recommendations.

### Step 8 — Paper figure generation

Generate all figures at once:

```bash
cd step8_painter
python main.py
```

Generate a specific section:

```bash
python main.py --section b   # b | c | d | e | f | i
```

## Data

The raw data (`data/src/flights.csv`, `data/src/parameters.csv`) is committed to the
repository. All derived datasets (split CSVs, windowed arrays, synthetic flights) are
intermediate products and are not tracked by git — they are recreated by running the
pipeline from step 1 onward.

## Configuration

Each step has a `config.py` that controls hyperparameters (window size, attack ratio,
model architecture, etc.). Edit this file before running `main.py` to customise an
experiment.
