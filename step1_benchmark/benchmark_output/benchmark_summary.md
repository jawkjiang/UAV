# Multi-Model Benchmark Results

## Summary Table

| model         |   num_params |   pr_auc_mean |   pr_auc_std |   roc_auc_mean |   roc_auc_std |   recall_at_fpr0.01_mean |   recall_at_fpr0.01_std |   actual_fpr_at_thr_mean |   actual_fpr_at_thr_std |   threshold_at_fpr0.01_mean |   threshold_at_fpr0.01_std |   avg_fp_per_flight_mean |   avg_fp_per_flight_std |   delay_median_mean |   delay_median_std |   delay_p90_mean |   delay_p90_std |   delay_mean_mean |   delay_mean_std |
|:--------------|-------------:|--------------:|-------------:|---------------:|--------------:|-------------------------:|------------------------:|-------------------------:|------------------------:|----------------------------:|---------------------------:|-------------------------:|------------------------:|--------------------:|-------------------:|-----------------:|----------------:|------------------:|-----------------:|
| A_Rule        |            0 |      0.211773 |            0 |       0.504943 |             0 |                 0.105932 |                       0 |               0.00290698 |                       0 |                 4.22764     |                          0 |                   0.625  |                       0 |                0.2  |                  0 |            0.519 |               0 |          -10.921  |                0 |
| B_XGBoost     |            0 |      1        |            0 |       1        |             0 |                 1        |                       0 |               0.090843   |                       0 |                 0.000168882 |                          0 |                  19.5312 |                       0 |                0.15 |                  0 |            0.5   |               0 |          -40.201  |                0 |
| C_CNN         |       353857 |      1        |            0 |       1        |             0 |                 0        |                       0 |               0          |                       0 |               inf           |                        nan |                   0      |                       0 |              nan    |                nan |          nan     |             nan |          nan      |              nan |
| D_TCN         |       326425 |      0.871473 |            0 |       0.976331 |             0 |                 0        |                       0 |               0          |                       0 |               inf           |                        nan |                   0      |                       0 |              nan    |                nan |          nan     |             nan |          nan      |              nan |
| E_GRU         |       301313 |      0.999989 |            0 |       0.999998 |             0 |                 0        |                       0 |               0          |                       0 |               inf           |                        nan |                   0      |                       0 |              nan    |                nan |          nan     |             nan |          nan      |              nan |
| F_Transformer |       312241 |      0.555906 |            0 |       0.599146 |             0 |                 0.486229 |                       0 |               0.0563953  |                       0 |                 0.535491    |                          0 |                  12.125  |                       0 |                0.35 |                  0 |            1.32  |               0 |          -25.2012 |                0 |

## Parameter Count Alignment

| model         |   num_params | param_range_check   |
|:--------------|-------------:|:--------------------|
| A_Rule        |            0 | ✓                   |
| B_XGBoost     |            0 | ✓                   |
| C_CNN         |       353857 | ✓                   |
| D_TCN         |       326425 | ✓                   |
| E_GRU         |       301313 | ✓                   |
| F_Transformer |       312241 | ✓                   |

## Training Configuration

- Learning rate: 0.001
- Batch size: 256
- Max epochs: 40
- Early stopping patience: 6
- Dropout: 0.1
- Random seeds: []
