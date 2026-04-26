"""
Load MATLAB-generated per-flight CSV files into a unified DataFrame.

Each CSV has columns: time, x, y, z, vx, vy, vz, ax, ay, az,
                      label, attack_type, onset_time
This loader assigns a unique integer flight_id and concatenates all flights.
"""
import os
import glob
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import config


def load_split(split: str, data_dir: str = config.DATA_DIR) -> pd.DataFrame:
    """
    Load all flights for one split ('train', 'val', or 'test').

    Returns a DataFrame with columns:
        flight_id, time, x, y, z, vx, vy, vz, ax, ay, az,
        label, attack_type, onset_time
    """
    split_dir = os.path.join(data_dir, split)
    csv_files = sorted(glob.glob(os.path.join(split_dir, 'flight_*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No flight CSVs found in {split_dir}")

    frames = []
    for fid, fpath in enumerate(csv_files):
        df = pd.read_csv(fpath)
        df['flight_id'] = fid
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # onset_time may be NaN for normal flights — keep as float
    out['onset_time'] = pd.to_numeric(out['onset_time'], errors='coerce')
    out['label']      = out['label'].astype(int)

    print(f"[{split}] {len(csv_files)} flights, "
          f"{len(out)} samples, "
          f"{out.groupby('flight_id')['label'].max().sum()} attacked")
    return out


def load_all_splits(data_dir: str = config.DATA_DIR
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test splits."""
    train = load_split('train', data_dir)
    val   = load_split('val',   data_dir)
    test  = load_split('test',  data_dir)
    return train, val, test


def make_windows(df: pd.DataFrame,
                 window_size: int = config.WINDOW_SIZE,
                 step_size:   int = config.STEP_SIZE
                 ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Slice all flights in df into fixed-length windows.

    Returns:
        X        : float32 array  [N_windows, window_size, N_FEATURES]
        y        : int32 array    [N_windows]  — end-point label
        metadata : list of dicts  — {flight_id, onset_time, window_end_time, attack_type}
    """
    feat_cols = config.FEATURE_COLS
    X_list, y_list, meta_list = [], [], []

    for fid, grp in df.groupby('flight_id', sort=True):
        grp = grp.sort_values('time').reset_index(drop=True)
        n   = len(grp)
        onset  = grp['onset_time'].iloc[0]
        a_type = grp['attack_type'].iloc[0]

        feats  = grp[feat_cols].values.astype(np.float32)
        labels = grp['label'].values.astype(np.int32)
        times  = grp['time'].values

        for start in range(0, n - window_size + 1, step_size):
            end = start + window_size
            X_list.append(feats[start:end])

            # End-point labelling (causally consistent)
            y_list.append(int(labels[end - 1]))

            meta_list.append({
                'flight_id':       fid,
                'onset_time':      onset,
                'window_end_time': float(times[end - 1]),
                'attack_type':     a_type,
            })

    X = np.stack(X_list, axis=0)          # [N, L, F]
    y = np.array(y_list, dtype=np.int32)  # [N]
    return X, y, meta_list


if __name__ == '__main__':
    train_df, val_df, test_df = load_all_splits()
    X_tr, y_tr, m_tr = make_windows(train_df)
    X_va, y_va, m_va = make_windows(val_df)
    X_te, y_te, m_te = make_windows(test_df)
    print(f"Windows — train:{X_tr.shape} val:{X_va.shape} test:{X_te.shape}")
    print(f"Label balance — train:{y_tr.mean():.2%} val:{y_va.mean():.2%} test:{y_te.mean():.2%}")
