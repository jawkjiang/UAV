# Time-Aware Evaluation Summary

## Overview

- **Models Evaluated**: 7
- **Attack Types**: step, drift_ramp, drift_sigmoid, delay, takeover_step, takeover_ramp
- **Time Thresholds**: [1, 2, 5, 10, 15, 30]

## Overall Performance

| Model | ADD (s) | MTBFA (h) | DR@1s | DR@5s | DR@10s | False Alarms |
|-------|---------|-----------|-------|-------|--------|--------------|
| BILSTM | 0.41 | 1.0 | 0.833 | 1.000 | 1.000 | 7 |
| CNN | 0.42 | 0.7 | 0.833 | 1.000 | 1.000 | 10 |
| CNN_LSTM | 0.41 | 0.7 | 0.833 | 1.000 | 1.000 | 10 |
| GRU | 0.41 | 1.1 | 0.833 | 1.000 | 1.000 | 6 |
| LSTM | 0.41 | 1.1 | 0.833 | 1.000 | 1.000 | 6 |
| TCN | 0.41 | 1.4 | 0.833 | 1.000 | 1.000 | 5 |
| TRANSFORMER | 0.41 | 0.5 | 0.833 | 1.000 | 1.000 | 13 |

## Scenario Recommendations

### Safety-Critical
**Recommended Model**: TCN
- Quick response required, tolerates some false alarms

### Long-term Monitoring
**Recommended Model**: TCN
- Low false alarm rate, slower detection acceptable

### Balanced
**Recommended Model**: TCN
- Balance between detection speed and false alarm rate

## Key Findings

- **Fastest Detection**: LSTM (ADD=0.41s)
- **Lowest False Alarm Rate**: TCN (MTBFA=1.4h)
- **Highest DR@5s**: CNN (DR@5s=1.000)
