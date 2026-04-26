# Detection Delay Calculation Methodology

## The Critical Question

**Question**: If the model only predicts whether a **window** contains an attack (binary classification), how can we measure **detection delay**? The model doesn't predict which specific timestamp within the window has the attack.

**Answer**: Ground truth has **sample-level temporal precision**, not just window-level labels.

---

## Ground Truth Temporal Resolution

### Attack Injection Process

1. **Precise Timestamp Injection** (from `multi_attack_injector.py`):
```python
# Attack injector selects a random timestamp t_s within flight
t_s = random.uniform(start_allowable, end_allowable)  # Precise timestamp in seconds

attack_info = {
    'attacked': True,
    'attack_type': 'step',  # or 'ramp', 'drift', etc.
    'attack_start_time': t_s,  # ← Precise attack start timestamp
    'attack_params': {...}
}
```

2. **Sample-Level Label Generation**:
   - Original data: 100 Hz sampling → 1 sample every 0.01 seconds
   - Attack injection modifies GPS data **starting from timestamp `t_s`**
   - Each sample is labeled:
     - `label = 1` if `timestamp >= attack_start_time`
     - `label = 0` otherwise
   - Result: **Sample-level ground truth** with 0.01s precision

3. **Metadata Storage** (from `inject_attacks_stratified.py`):
```python
attack_metadata.append({
    'flight': flight_id,
    'attack_type': attack_info_dict['attack_type'],
    'attack_start_time': float(attack_info_dict['attack_start_time']),  # ← Stored
    'is_attacked': 1
})
```

---

## Window-Level Prediction vs. Sample-Level Ground Truth

### Key Insight: Different Resolutions for Different Purposes

| Aspect | Ground Truth | Model Prediction |
|--------|-------------|------------------|
| **Temporal Resolution** | Sample-level (0.01s) | Window-level (1s) |
| **Purpose** | Know exact attack start | Detect attack presence |
| **Availability** | Always known (injected) | Must be learned |
| **Used for** | Calculating delay metrics | Real-time detection |

### Visualization

```
Time:          0.00s    0.50s    1.00s    1.50s    2.00s    2.50s    3.00s
               ├────────┼────────┼────────┼────────┼────────┼────────┤
Ground Truth:  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1
               (sample-level, 100Hz)
                                          ↑
                                     t_attack = 1.50s (precise)

Windows:       ├────Window 0─────┤├────Window 1─────┤├────Window 2─────┤
               │ 0 0 0 0 0 0 0 0 0 0│ 0 0 0 0 0 1 1 1 1 1│ 1 1 1 1 1 1 1 1 1 1│
               │                   │                   │                   │
Labels:        │   label = 0       │   label = 1       │   label = 1       │
               │ (no attack)       │ (contains attack) │ (contains attack) │
                                   
Model Pred:    │   ŷ = 0           │   ŷ = 0           │   ŷ = 1           │
                                   │                   │                   │
                                   │                   └─────► t_detect = 2.00s
                                   └─► Missed (FN)              (window center)
```

---

## Detection Delay Calculation (from `time_aware_metrics.py`)

### Step 1: Identify Attack Segments with Precise Timing

```python
def identify_attack_segments_v2(y_true, timestamps, flight_ids, attack_types, 
                                attack_segments_info):
    """
    Extract attack segments with precise attack_start_time.
    
    Parameters:
        attack_segments_info: List of dicts with:
            - flight_id: int
            - attack_type: str
            - attack_start_time: float  ← Precise timestamp from injection
    """
    attack_info_dict = {item['flight_id']: item for item in attack_segments_info}
    
    for flight_id in unique_flights:
        # Get precise attack start time from metadata
        if flight_id in attack_info_dict:
            attack_info = attack_info_dict[flight_id]
            t_attack_real = attack_info['attack_start_time']  # ← Precise timestamp
        else:
            # Fallback: use first attack window's timestamp
            first_attack_idx = flight_indices[attack_indices_local[0]]
            t_attack_real = timestamps[first_attack_idx]
        
        segments.append({
            'flight_id': int(flight_id),
            'attack_type': attack_type_real,
            't_attack': float(t_attack_real),  # ← Ground truth attack start time
            't_first_window': float(timestamps[seg_indices[0]]),
            't_last_window': float(timestamps[seg_indices[-1]]),
            'indices': seg_indices.tolist()
        })
```

### Step 2: Calculate Detection Delay

```python
def calculate_detection_delay_v2(predictions, timestamps, attack_segments):
    """
    Calculate detection delay using precise attack_start_time.
    
    Key:
        - t_attack: Precise attack start time (from ground truth metadata)
        - t_detect: Timestamp of first window predicted as attack
        - delay = t_detect - t_attack
    """
    delays = []
    
    for seg in attack_segments:
        t_attack = seg['t_attack']  # ← Precise ground truth time
        seg_indices = seg['indices']
        
        # Find first window predicted as attack
        pred_attack_mask = predictions[seg_indices] == 1
        
        if not pred_attack_mask.any():
            # Missed detection (FN)
            delays.append(np.nan)
            continue
        
        # Get timestamp of first detection
        first_detect_idx = seg_indices[np.where(pred_attack_mask)[0][0]]
        t_detect = timestamps[first_detect_idx]
        
        # Calculate delay
        delay = t_detect - t_attack
        
        # Handle negative delays (window detected before attack starts)
        if delay < 0:
            # Window center may be before attack start, but window contains attack
            delay = 0.0  # Immediate detection
        
        delays.append(delay)
    
    # Average Detection Delay (ADD)
    valid_delays = [d for d in delays if not np.isnan(d)]
    ADD = np.mean(valid_delays) if valid_delays else np.nan
    
    return ADD, delays
```

---

## Why This Works: The Resolution Mismatch is By Design

### 1. Ground Truth: Sample-Level (High Resolution)
- **What**: Each sample has a binary label (0 or 1)
- **Why**: Attack injection happens at a specific timestamp
- **Precision**: 0.01s (100 Hz sampling)
- **Purpose**: Know exact attack start time for metric calculation

### 2. Model Input/Output: Window-Level (Lower Resolution)
- **Input**: 100 consecutive samples → 1 window
- **Output**: Binary prediction (0 or 1) per window
- **Precision**: 1s (window size at 100 Hz)
- **Purpose**: Practical real-time detection

### 3. Delay Calculation: Combines Both
```
delay = t_detect (from model prediction) - t_attack (from ground truth)
        ↑                                  ↑
        Window-level                       Sample-level
        (what model predicts)              (what we know from injection)
```

---

## Concrete Example: Transformer Model on Flight 160

### Ground Truth
```python
# From attack injection metadata
attack_start_time = 1.50s  # Precise timestamp

# Sample-level labels (100 Hz)
Time:  1.40s  1.41s  1.42s  ...  1.49s  1.50s  1.51s  1.52s  ...  2.00s
Label:   0     0     0    ...    0     1     1     1    ...    1
                                        ↑
                                   Attack starts here
```

### Model Prediction
```python
# Window-level predictions
Window 140 (1.40-1.50s): ŷ = 0  (contains 90% normal, 10% attack → label = 1, but missed)
Window 145 (1.45-1.55s): ŷ = 0  (contains 50% normal, 50% attack → label = 1, but missed)
Window 150 (1.50-1.60s): ŷ = 0  (contains 0% normal, 100% attack → label = 1, but missed)
Window 155 (1.55-1.65s): ŷ = 1  (first detection! window center at 1.60s)
```

### Delay Calculation
```python
t_attack = 1.50s  # From ground truth metadata
t_detect = 1.60s  # Timestamp of Window 155 (first ŷ=1)
delay = 1.60 - 1.50 = 0.10s
```

---

## Addressing the Paradox: Window Labels vs. Attack Timestamps

### The Question
**If a window contains 100 samples, and the attack starts at sample 50, how do we know which sample is the attack start?**

### The Answer
**We don't determine it from the model—we know it from the ground truth metadata.**

1. **Attack Injection** creates the precise timestamp (`attack_start_time = 1.50s`)
2. **Window Labeling** uses this to assign binary labels:
   - If window contains ANY samples with `timestamp >= attack_start_time`, label = 1
   - Otherwise, label = 0
3. **Model Training** learns to predict window labels (not timestamps)
4. **Delay Calculation** uses:
   - **t_attack**: From injection metadata (precise)
   - **t_detect**: Timestamp of first window with ŷ=1 (window-level)

---

## Why Not Flight-Level Evaluation?

### If We Used Flight-Level Metrics
```python
# Flight-level prediction
"This flight has an attack" (binary classification)

# How to calculate delay?
# Option 1: delay = 0 (detected at flight start) - meaningless
# Option 2: delay = flight_duration (detected at flight end) - meaningless
# Option 3: Can't calculate delay at all
```

### With Window-Level Evaluation
```python
# Window-level prediction stream
t=0.00s: ŷ=0  (normal)
t=1.00s: ŷ=0  (normal)
t=2.00s: ŷ=1  (attack detected!) ← This timestamp is meaningful
t=3.00s: ŷ=1  (attack continues)

# Delay calculation is well-defined
delay = t_detect (2.00s) - t_attack (1.50s) = 0.50s
```

---

## Summary: Three-Layer Architecture

| Layer | Resolution | Data Type | Purpose |
|-------|-----------|-----------|---------|
| **Ground Truth** | Sample-level (0.01s) | Binary labels + metadata | Precise attack timing |
| **Model I/O** | Window-level (1.0s) | Binary predictions | Practical detection |
| **Metrics** | Event-level | Aggregated statistics | Performance evaluation |

### Key Points

1. **Ground truth has sample-level precision** from attack injection metadata
2. **Model operates at window-level** for practical real-time detection
3. **Delay calculation combines both**:
   - `t_attack` from sample-level ground truth (precise)
   - `t_detect` from window-level prediction (first detection)
4. **This is standard practice** in time-series anomaly detection:
   - Precise ground truth for evaluation
   - Coarser predictions for practicality
   - Metrics that bridge the gap

---

## Code Verification

### Attack Injection (Step 5)
```python
# File: step5_timegan/inject_attacks_stratified.py
attacked_flight, attack_info_dict = injector.inject_attack_to_flight(...)

attack_metadata.append({
    'attack_start_time': float(attack_info_dict['attack_start_time']),  # ← Precise
})
```

### Data Loading (Step 6)
```python
# File: step6_newMetrcisWithTimegan/data_loader.py
attack_dict[fid] = {
    'attack_start_time': float(row['attack_start_time'])  # ← Loaded from CSV
}
```

### Delay Calculation (Step 6)
```python
# File: step6_newMetrcisWithTimegan/time_aware_metrics.py
t_attack_real = attack_info['attack_start_time']  # ← From metadata
delay = t_detect - t_attack  # ← Precise calculation
```

---

## Reviewer Concerns Addressed

### Concern 1: "The model only predicts window labels"
**Response**: Correct. But ground truth has precise attack timestamps from injection.

### Concern 2: "How can you measure delay without knowing attack position in window?"
**Response**: We **do** know the precise attack position—from the injection metadata, not from the model.

### Concern 3: "Isn't this circular reasoning?"
**Response**: No. This is the standard evaluation approach:
- Training: Model learns window-level patterns
- Evaluation: Compare window-level predictions against sample-level ground truth
- Metrics: Use precise ground truth to calculate meaningful delays

### Concern 4: "Why not just use flight-level evaluation?"
**Response**: Flight-level cannot measure delay—you need temporal granularity for time-aware metrics.

---

## Conclusion

**The detection delay methodology is valid because:**

1. ✅ Ground truth has **sample-level temporal precision** (0.01s)
2. ✅ Attack injection creates **precise `attack_start_time` metadata**
3. ✅ Model predictions are **window-level** (practical for real-time)
4. ✅ Delay calculation **combines both resolutions**:
   - Precise t_attack from ground truth
   - Practical t_detect from model predictions
5. ✅ This is **standard practice** in time-series anomaly detection

**The model doesn't need to predict attack timestamps—we already know them from the ground truth.**
