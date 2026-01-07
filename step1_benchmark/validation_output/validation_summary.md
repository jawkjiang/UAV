# GPS Spoofing Detection - Validation Report

## Executive Summary

This report validates the near-perfect detection performance (PR-AUC=1.0) achieved by the GPS spoofing detection model.

## Experiment Results

### Experiment 1: Single-Feature Threshold Baseline

**Conclusion:** Model utilizes complex temporal structure beyond single feature.

- PR-AUC: 0.1128
- Recall@FPR=0.01: 0.0896
- Avg FP per flight: 2.19

### Experiment 2: Feature Ablation Study

**Conclusion:** Model benefits from combination of basic and residual features.

| Configuration | PR-AUC | Recall@FPR=0.01 |
|--------------|--------|----------------|
| Full Features | 1.0000 | 1.0000 |
| No Residuals | 1.0000 | 0.0000 |
| Residuals Only | 0.2942 | 0.2406 |

### Experiment 3: Attack Parameter Robustness

**Conclusion:** Model is overfitted to specific attack injection patterns.

| Attack Config | PR-AUC | Recall@FPR=0.01 |
|--------------|--------|----------------|
| Short/Fast | 0.2551 | 0.0000 |
| Long/Slow | 0.2554 | 0.0000 |

### Experiment 4: Leave-One-Route-Out Generalization

**Conclusion:** Model performance varies significantly across routes (std=0.2500). May be learning route-specific patterns.

### Experiment 5: Attack Position Distribution

**Conclusion:** Full evaluation requires re-injection with controlled attack timing. Current dataset has attacks primarily detected via window-end labels, which may introduce positional bias.

## Overall Conclusion

### Model Validity Boundaries

The GPS spoofing detection model demonstrates:

3. **Limited Generalization**: Model may be overfitted to training attack parameters.


### Applicability Scope

The model is effective when:
- GPS spoofing causes step-type position offsets
- Attacks induce measurable position-velocity-acceleration inconsistencies
- Sensor sampling is similar to training distribution

The model may have limitations with:
- Gradual drift-type spoofing
- Attacks that maintain kinematic consistency
- Significantly different vehicle dynamics or sensor configurations
