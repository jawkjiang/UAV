# Step 4b: Time-Aware Evaluation Framework

A comprehensive evaluation framework for GPS spoofing detection models using **time-aware metrics** that reflect real-world deployment performance.

## 📋 Overview

Traditional ML metrics (ROC-AUC, F1) don't capture critical aspects of real-time detection systems:
- **Detection Delay**: How quickly attacks are detected
- **False Alarm Rate**: Real operational impact of false positives

This framework introduces:
- **DR@Δt**: Detection Rate within Δt seconds
- **ADD**: Average Detection Delay
- **MTBFA**: Mean Time Between False Alarms

## 🎯 Key Features

### Time-Aware Metrics

1. **Detection Rate @ Δt (DR@Δt)**
   - Percentage of attacks detected within Δt seconds of attack start
   - Evaluated at Δt = {1, 2, 5, 10, 15, 30} seconds
   - Critical for safety-critical applications

2. **Average Detection Delay (ADD)**
   - Mean time from attack start to first detection
   - Lower is better (faster response)
   - Measured in seconds

3. **Mean Time Between False Alarms (MTBFA)**
   - Average time between false alarm incidents
   - Higher is better (fewer disruptions)
   - Measured in hours
   - Consecutive false positives within 1s count as single alarm

### Scenario Analysis

Evaluates models under three deployment scenarios:

1. **Safety-Critical**: Fast response, tolerates some false alarms
   - DR@5s ≥ 0.95
   - ADD ≤ 3s
   - MTBFA ≥ 5h

2. **Long-term Monitoring**: Low false alarm rate, slower detection OK
   - DR@10s ≥ 0.90
   - MTBFA ≥ 24h
   - ADD ≤ 10s

3. **Balanced**: Equilibrium between speed and false alarms
   - DR@5s ≥ 0.90
   - MTBFA ≥ 12h
   - ADD ≤ 5s

## 🚀 Quick Start

### Prerequisites

Requires completed Step 3b training with model outputs in:
```
step3b_multiModelGeneral/output/
├── cnn/test_predictions.npz
├── lstm/test_predictions.npz
├── bilstm/test_predictions.npz
├── gru/test_predictions.npz
├── cnn_lstm/test_predictions.npz
├── tcn/test_predictions.npz
└── transformer/test_predictions.npz
```

### Run Evaluation

```bash
# Navigate to step4b directory
cd step4b_newMetrics

# Run complete evaluation pipeline
python main.py
```

### Test Individual Modules

```bash
# Test time-aware metrics computation
python time_aware_metrics.py

# Test data loading
python data_loader.py

# Test scenario analysis
python scenario_analysis.py
```

## 📊 Outputs

### CSV Files

**`output/time_aware_metrics/overall_metrics.csv`**
```csv
model,n_attacks,ADD,MTBFA,DR@1s,DR@2s,DR@5s,DR@10s,DR@15s,DR@30s
cnn,210,3.2,18.5,0.65,0.78,0.92,0.96,0.98,0.99
lstm,210,2.8,12.3,0.72,0.85,0.95,0.98,0.99,1.00
...
```

**`output/time_aware_metrics/{model}_per_attack_metrics.csv`**
```csv
model,attack_type,n_instances,ADD,DR@5s,DR@10s
cnn,step,35,1.5,0.95,0.98
cnn,drift_ramp,42,4.2,0.87,0.93
...
```

**`output/time_aware_metrics/detailed_delays.csv`**
- Individual delay for each attack instance
- Useful for deep-dive analysis

### Visualizations

1. **`dr_vs_delay.png`**: DR@Δt curves for all models
2. **`dr_vs_mtbfa.png`**: Trade-off scatter plot
3. **`delay_distribution.png`**: Box plots of detection delays
4. **`per_attack_heatmap.png`**: DR@5s by model and attack type
5. **`mtbfa_comparison.png`**: Horizontal bar chart of MTBFA

### Reports

**`scenario_analysis.txt`**: Detailed scenario evaluation
- Requirements for each scenario
- Model rankings with scores
- Pass/fail status and reasons
- Recommendations

**`evaluation_summary.md`**: Markdown summary
- Overall performance table
- Scenario recommendations
- Key findings

## 📁 Project Structure

```
step4b_newMetrics/
├── config.py                  # Configuration parameters
├── data_loader.py             # Load step3b outputs
├── time_aware_metrics.py      # Core metrics computation
├── evaluation.py              # Evaluation pipeline
├── visualizations.py          # Plot generation
├── scenario_analysis.py       # Scenario evaluation
├── main.py                    # Main execution script
├── README.md                  # This file
└── output/
    └── time_aware_metrics/    # Generated outputs
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Models to evaluate
MODELS = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']

# Time thresholds
DELTA_T_VALUES = [1, 2, 5, 10, 15, 30]

# False alarm gap threshold
FALSE_ALARM_GAP_THRESHOLD = 1.0  # seconds

# Scenario constraints (edit as needed)
SCENARIO_SAFETY_CRITICAL = {...}
```

## 📈 Interpretation Guide

### Good Performance Indicators

- **DR@5s > 0.90**: Detects 90%+ attacks within 5 seconds
- **ADD < 3s**: Average delay under 3 seconds
- **MTBFA > 10h**: Less than 1 false alarm per 10 hours

### Model Selection

Choose based on deployment scenario:

```
Safety-Critical → Prioritize DR@5s and ADD
Monitoring      → Prioritize MTBFA
Balanced        → Consider weighted score
```

## 🔍 Key Implementation Details

### Attack Segment Identification

- Scans ground truth labels to find attack periods
- Each continuous sequence of `y_true=1` is one attack
- Does not cross flight boundaries

### Detection Delay Calculation

- Measures time from **attack start** to **first detection**
- Only considers detections within the attack period
- Undetected attacks have `delay=None`

### MTBFA Calculation

- Counts **false alarm instances** (not individual false positives)
- Consecutive FPs within 1s counted as single instance
- Divides total normal flight time by number of instances

## ⚙️ Dependencies

```python
numpy
pandas
matplotlib
seaborn
```

Install via:
```bash
pip install numpy pandas matplotlib seaborn
```

## 📝 Notes

### Data Mapping

- Step3b outputs are **window-level** predictions
- Metadata (timestamps, flight IDs) reconstructed from attack info
- Window center time used as reference point

### Assumptions

- Sampling rate: 100 Hz (10ms per sample)
- Window size: 50 samples
- Step size: 5 samples
- Sequential window ordering by flight

### Limitations

- Window-level predictions vs point-level ideal
- Time resolution limited by window step size
- Assumes attack info CSV contains accurate timing

## 🎓 Research Context

Part of UAV GPS Spoofing Detection project evaluating:
- 7 deep learning architectures
- 6 attack types
- Time-aware vs traditional metrics

**Key Contribution**: Demonstrates that traditional metrics (ROC-AUC) don't reflect real-world detection performance. Time-aware metrics reveal significant differences in operational characteristics.

## 📧 Support

For issues or questions, refer to:
- `TIME_AWARE_EVALUATION_CONTEXT.md` - Detailed specification
- Code comments in individual modules
- Test outputs from `python {module}.py`

---

**Last Updated**: January 2026
**Version**: 1.0
