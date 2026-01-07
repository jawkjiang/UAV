# Multi-Model Multi-Attack GPS Spoofing Detection

## Experiment Overview

- Total experiments: 56
- Attack types tested: 8
- Model architectures tested: 7

## Best Overall Configuration

- **Attack Type:** drift_ramp
- **Model Type:** bilstm
- **AUC-ROC:** 1.0000
- **F1 Score:** 1.0000

## Model Rankings

| Rank | Model | Avg AUC-ROC | Avg F1 Score |
|------|-------|-------------|--------------|
| 1 | tcn | 0.9438 | 0.7847 |
| 2 | cnn | 0.9311 | 0.7651 |
| 3 | cnn_lstm | 0.9266 | 0.7398 |
| 4 | gru | 0.9219 | 0.5773 |
| 5 | bilstm | 0.9218 | 0.7153 |
| 6 | lstm | 0.9132 | 0.7638 |
| 7 | transformer | 0.9089 | 0.5429 |

## Attack Detection Difficulty

(Sorted by detection difficulty - lower AUC-ROC = harder to detect)

| Rank | Attack Type | Avg AUC-ROC | Avg F1 Score |
|------|-------------|-------------|--------------|
| 1 | replay_same_hard | 0.6033 | 0.0685 |
| 2 | replay_other_soft | 0.7964 | 0.1223 |
| 3 | delay | 0.9927 | 0.8071 |
| 4 | takeover_ramp | 0.9992 | 0.9723 |
| 5 | takeover_step | 0.9997 | 0.9037 |
| 6 | step | 0.9998 | 0.9639 |
| 7 | drift_sigmoid | 0.9998 | 0.9287 |
| 8 | drift_ramp | 1.0000 | 0.8208 |

## Best Model for Each Attack Type

| Attack Type | Best Model | AUC-ROC | F1 Score |
|-------------|------------|---------|----------|
| delay | tcn | 0.9998 | 0.9976 |
| drift_ramp | bilstm | 1.0000 | 1.0000 |
| drift_sigmoid | bilstm | 1.0000 | 0.9509 |
| replay_other_soft | tcn | 0.8809 | 0.2164 |
| replay_same_hard | tcn | 0.6702 | 0.0858 |
| step | bilstm | 1.0000 | 1.0000 |
| takeover_ramp | gru | 1.0000 | 1.0000 |
| takeover_step | cnn | 1.0000 | 1.0000 |

## Visualizations

See the `visualizations/` directory for detailed plots:
- `heatmap_auc_roc.png` - AUC-ROC heatmap
- `heatmap_f1_score.png` - F1 Score heatmap
- `boxplot_models.png` - Model performance distributions
- `boxplot_attacks.png` - Attack difficulty distributions
- `bar_model_performance.png` - Average model performance
- `scatter_precision_recall.png` - Precision-Recall tradeoffs