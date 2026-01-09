# Time-Aware Evaluation Summary

## Overview

- **Models Evaluated**: 7
- **Attack Types**: step, drift_ramp, drift_sigmoid, delay, takeover_step, takeover_ramp
- **Time Thresholds**: [1, 2, 5, 10, 15, 30]

## Overall Performance

| Model | ADD (s) | MTBFA (h) | DR@1s | DR@5s | DR@10s | False Alarms |
|-------|---------|-----------|-------|-------|--------|--------------|
| BILSTM | N/A | inf | 0.000 | 0.000 | 0.000 | 0 |
| CNN | 0.02 | inf | 0.015 | 0.015 | 0.015 | 0 |
| CNN_LSTM | N/A | inf | 0.000 | 0.000 | 0.000 | 0 |
| GRU | N/A | inf | 0.000 | 0.000 | 0.000 | 0 |
| LSTM | N/A | inf | 0.000 | 0.000 | 0.000 | 0 |
| TCN | 0.00 | inf | 0.490 | 0.490 | 0.490 | 0 |
| TRANSFORMER | N/A | inf | 0.000 | 0.000 | 0.000 | 0 |

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

- **Fastest Detection**: TCN (ADD=0.00s)
- **Lowest False Alarm Rate**: CNN (MTBFA=inf)
- **Highest DR@5s**: TCN (DR@5s=0.490)
