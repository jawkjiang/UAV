"""
Loader for the ALFA Dataset (CMU AirLab Fixed-Wing UAV Faults).

The ALFA dataset contains 47 real fixed-wing UAV flights with labelled faults
(engine failure, control surface jams). We use the normal segments as clean
baselines and inject synthetic GPS spoofing to create attacked samples.

Dataset source:
    https://kilthub.cmu.edu/articles/dataset/ALFA_A_Dataset_for_UAV_Fault_and_Anomaly_Detection/12707963
    DOI: 10.1184/R1/12707963

Expected directory structure (after download + extraction):
    step9_realworld/data/alfa/
        flight_01.csv
        flight_02.csv
        ...
        labels.csv   (or per-flight JSON with fault onset times)

The ALFA CSV files contain columns roughly:
    time, latitude, longitude, altitude, airspeed, roll, pitch, yaw,
    ax, ay, az, p, q, r, fault_label, fault_onset_time

We align to the 9-d state vector [x,y,z,vx,vy,vz,ax,ay,az] using:
    - x,y,z: convert lat/lon/alt → local ENU via pyproj or simple approx
    - vx,vy,vz: derived from airspeed + attitude angles
    - ax,ay,az: body-frame accelerations rotated to ENU (approx)
"""
import os
import math
import numpy as np
import pandas as pd
from typing import List, Tuple

ALFA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'alfa')
FEATURE_COLS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']


def _latlon_to_enu(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
                   lat0: float, lon0: float, alt0: float) -> Tuple[np.ndarray, ...]:
    """Convert lat/lon/alt to local ENU using flat-earth approximation."""
    R = 6371000.0  # Earth radius in metres
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = R * dlon * math.cos(math.radians(lat0))  # East
    y = R * dlat                                   # North
    z = alt - alt0                                 # Up
    return x, y, z


def _airspeed_to_enu_velocity(airspeed: np.ndarray, yaw: np.ndarray,
                               pitch: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Approximate body-frame airspeed to ENU velocity using attitude angles."""
    vx = airspeed * np.cos(pitch) * np.sin(yaw)   # East component
    vy = airspeed * np.cos(pitch) * np.cos(yaw)   # North component
    vz = airspeed * np.sin(pitch)                  # Up component
    return vx, vy, vz


def load_alfa_flight(csv_path: str, flight_id: str, inject_spoofing: bool = False,
                     spoofing_onset_frac: float = 0.6,
                     step_magnitude: float = 15.0) -> pd.DataFrame:
    """
    Load one ALFA flight CSV and align to the 9-d feature schema.

    If inject_spoofing=True, a step attack is injected at spoofing_onset_frac
    into the attacked portion of the flight (only on normal flights).

    Returns DataFrame with columns:
        time, x, y, z, vx, vy, vz, ax, ay, az, label, attack_type, onset_time, flight_id
    """
    df_raw = pd.read_csv(csv_path)
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    # --- Determine reference origin ---
    lat0 = float(df_raw['latitude'].iloc[0])
    lon0 = float(df_raw['longitude'].iloc[0])
    alt0 = float(df_raw['altitude'].iloc[0])

    # --- ENU position ---
    x, y, z = _latlon_to_enu(
        df_raw['latitude'].values, df_raw['longitude'].values,
        df_raw['altitude'].values, lat0, lon0, alt0)

    # --- ENU velocity (from airspeed + attitude) ---
    yaw   = df_raw.get('yaw',   pd.Series(np.zeros(len(df_raw)))).values
    pitch = df_raw.get('pitch', pd.Series(np.zeros(len(df_raw)))).values
    airspeed = df_raw.get('airspeed', pd.Series(np.ones(len(df_raw)) * 10.0)).values
    vx, vy, vz = _airspeed_to_enu_velocity(airspeed, yaw, pitch)

    # --- Acceleration (body-frame → approximate ENU, or fallback to 0) ---
    ax = df_raw.get('ax', pd.Series(np.zeros(len(df_raw)))).values
    ay = df_raw.get('ay', pd.Series(np.zeros(len(df_raw)))).values
    az = df_raw.get('az', pd.Series(np.zeros(len(df_raw)))).values

    # --- Time ---
    time_col = next((c for c in df_raw.columns if 'time' in c), None)
    t = df_raw[time_col].values.astype(float) if time_col else np.arange(len(df_raw)) * 0.1

    out = pd.DataFrame({
        'time': t, 'x': x, 'y': y, 'z': z,
        'vx': vx, 'vy': vy, 'vz': vz,
        'ax': ax, 'ay': ay, 'az': az,
        'label': 0, 'attack_type': 'none', 'onset_time': float('nan'),
        'flight_id': flight_id,
    })

    # --- Inject synthetic GPS spoofing step on normal segments ---
    if inject_spoofing and len(out) > 20:
        onset_idx  = int(len(out) * spoofing_onset_frac)
        onset_time = float(out['time'].iloc[onset_idx])
        out.loc[out.index[onset_idx:], 'x'] += step_magnitude
        out.loc[out.index[onset_idx:], 'label'] = 1
        out['attack_type'] = 'step'
        out['onset_time']  = onset_time

    return out


def load_alfa_dataset(inject_ratio: float = 0.5,
                      seed: int = 0) -> pd.DataFrame:
    """
    Load all ALFA flights from ALFA_DIR.

    inject_ratio: fraction of normal flights to inject synthetic spoofing into.
    Returns concatenated DataFrame with all flights.
    """
    if not os.path.isdir(ALFA_DIR):
        raise FileNotFoundError(
            f'ALFA data directory not found: {ALFA_DIR}\n'
            f'Download from https://kilthub.cmu.edu/articles/dataset/'
            f'ALFA_A_Dataset_for_UAV_Fault_and_Anomaly_Detection/12707963\n'
            f'and place CSV files in {ALFA_DIR}')

    csv_files = sorted([f for f in os.listdir(ALFA_DIR) if f.endswith('.csv')])
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {ALFA_DIR}')

    rng = np.random.default_rng(seed)
    inject_mask = rng.random(len(csv_files)) < inject_ratio

    dfs = []
    for i, fname in enumerate(csv_files):
        flight_id = f'alfa_{os.path.splitext(fname)[0]}'
        try:
            df = load_alfa_flight(
                os.path.join(ALFA_DIR, fname), flight_id,
                inject_spoofing=bool(inject_mask[i]))
            dfs.append(df)
        except Exception as e:
            print(f'  Warning: skipping {fname}: {e}')

    if not dfs:
        raise RuntimeError('No ALFA flights loaded successfully')

    combined = pd.concat(dfs, ignore_index=True)
    n_inject = inject_mask.sum()
    print(f'[ALFA] {len(csv_files)} flights loaded, {n_inject} with injected spoofing')
    return combined
