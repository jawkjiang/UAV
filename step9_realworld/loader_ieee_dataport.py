"""
Loader for the IEEE DataPort UAV Attack Dataset.

Contains real UAV flights (Holybro S500 + Pixhawk 4) with hardware GPS spoofing
attacks performed using a HackRF One SDR.

Dataset source:
    https://ieee-dataport.org/open-access/uav-attack-dataset
    DOI: 10.21227/00dg-0d12

Expected directory structure (after download):
    step9_realworld/data/ieee_dataport/
        normal/
            flight_001.csv  ...
        spoofing/
            flight_001.csv  ...

The CSVs are PX4 ULOG exports containing columns approximately:
    timestamp, gps_lat, gps_lon, gps_alt, gps_vel_n, gps_vel_e, gps_vel_d,
    accel_x, accel_y, accel_z, attack_active (0/1), attack_onset_us

We align to the 9-d state vector [x,y,z,vx,vy,vz,ax,ay,az] using:
    - x,y,z: lat/lon/alt → local ENU
    - vx,vy,vz: NED velocity → ENU (swap N↔E, negate D)
    - ax,ay,az: IMU accelerations (body → ENU approximation)
"""
import os
import math
import numpy as np
import pandas as pd
from typing import Optional

DATAPORT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'ieee_dataport')
FEATURE_COLS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']


def _latlon_to_enu(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
                   lat0: float, lon0: float, alt0: float):
    R = 6371000.0
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = R * dlon * math.cos(math.radians(lat0))
    y = R * dlat
    z = alt - alt0
    return x, y, z


def _ned_to_enu(vn, ve, vd):
    """Convert NED velocity to ENU."""
    return ve, vn, -vd


def _find_col(df: pd.DataFrame, candidates: list, default=None) -> Optional[str]:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return default


def load_dataport_flight(csv_path: str, flight_id: str,
                          is_attacked: bool = False) -> pd.DataFrame:
    """
    Load one IEEE DataPort flight CSV and align to 9-d feature schema.

    Returns DataFrame with columns:
        time, x, y, z, vx, vy, vz, ax, ay, az, label, attack_type, onset_time, flight_id
    """
    df_raw = pd.read_csv(csv_path)
    df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(' ', '_')

    # --- Time (microseconds → seconds, zero-referenced) ---
    t_col = _find_col(df_raw, ['timestamp', 'time_us', 'time', 'ts'])
    if t_col is None:
        t = np.arange(len(df_raw)) * 0.1
    else:
        t = df_raw[t_col].values.astype(float)
        if t.max() > 1e9:      # microseconds
            t = t / 1e6
        t = t - t[0]

    # --- GPS position → ENU ---
    lat = df_raw[_find_col(df_raw, ['gps_lat', 'lat', 'latitude', 'pos_lat']) or 'lat'].values
    lon = df_raw[_find_col(df_raw, ['gps_lon', 'lon', 'longitude', 'pos_lon']) or 'lon'].values
    alt = df_raw[_find_col(df_raw, ['gps_alt', 'alt', 'altitude', 'pos_alt']) or 'alt'].values
    x, y, z = _latlon_to_enu(lat, lon, alt, lat[0], lon[0], alt[0])

    # --- Velocity: NED → ENU ---
    vn_col = _find_col(df_raw, ['vel_n', 'gps_vel_n', 'vn', 'velocity_north'])
    ve_col = _find_col(df_raw, ['vel_e', 'gps_vel_e', 've', 'velocity_east'])
    vd_col = _find_col(df_raw, ['vel_d', 'gps_vel_d', 'vd', 'velocity_down'])
    if vn_col and ve_col and vd_col:
        vx, vy, vz = _ned_to_enu(df_raw[vn_col].values,
                                   df_raw[ve_col].values,
                                   df_raw[vd_col].values)
    else:
        # Derive velocity via finite difference of ENU position
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        vz = np.gradient(z, t)

    # --- Acceleration ---
    ax_col = _find_col(df_raw, ['accel_x', 'acc_x', 'imu_ax', 'ax'])
    ay_col = _find_col(df_raw, ['accel_y', 'acc_y', 'imu_ay', 'ay'])
    az_col = _find_col(df_raw, ['accel_z', 'acc_z', 'imu_az', 'az'])
    ax = df_raw[ax_col].values if ax_col else np.gradient(vx, t)
    ay = df_raw[ay_col].values if ay_col else np.gradient(vy, t)
    az = df_raw[az_col].values if az_col else np.gradient(vz, t)

    # --- Attack labels ---
    label = np.zeros(len(df_raw), dtype=int)
    onset_time = float('nan')
    attack_type = 'none'
    if is_attacked:
        atk_col = _find_col(df_raw, ['attack_active', 'attack', 'spoofing', 'label'])
        onset_col = _find_col(df_raw, ['attack_onset_us', 'onset_time', 'attack_start'])
        if atk_col:
            label = (df_raw[atk_col].values != 0).astype(int)
            first_attack = np.where(label == 1)[0]
            onset_time = float(t[first_attack[0]]) if len(first_attack) > 0 else float('nan')
        elif onset_col:
            onset_s = float(df_raw[onset_col].iloc[0])
            if onset_s > 1e6:
                onset_s /= 1e6
            onset_time = onset_s - t[0]
            label = (t >= onset_time).astype(int)
        else:
            # Assume full flight is attacked
            onset_time = float(t[len(t)//2])
            label[len(t)//2:] = 1
        attack_type = 'spoofing'

    return pd.DataFrame({
        'time': t, 'x': x, 'y': y, 'z': z,
        'vx': vx, 'vy': vy, 'vz': vz,
        'ax': ax, 'ay': ay, 'az': az,
        'label': label, 'attack_type': attack_type,
        'onset_time': onset_time, 'flight_id': flight_id,
    })


def load_dataport_dataset() -> pd.DataFrame:
    """
    Load all IEEE DataPort flights.
    Expects subdirectories: normal/ and spoofing/ under DATAPORT_DIR.
    """
    if not os.path.isdir(DATAPORT_DIR):
        raise FileNotFoundError(
            f'IEEE DataPort data directory not found: {DATAPORT_DIR}\n'
            f'Download from https://ieee-dataport.org/open-access/uav-attack-dataset\n'
            f'and place CSV files in {DATAPORT_DIR}/normal/ and {DATAPORT_DIR}/spoofing/')

    dfs = []
    for split, is_attacked in [('normal', False), ('spoofing', True)]:
        split_dir = os.path.join(DATAPORT_DIR, split)
        if not os.path.isdir(split_dir):
            print(f'  Warning: {split_dir} not found, skipping')
            continue
        csv_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.csv')])
        for fname in csv_files:
            fid = f'dataport_{split}_{os.path.splitext(fname)[0]}'
            try:
                df = load_dataport_flight(
                    os.path.join(split_dir, fname), fid, is_attacked=is_attacked)
                dfs.append(df)
            except Exception as e:
                print(f'  Warning: skipping {fname}: {e}')

    if not dfs:
        raise RuntimeError('No IEEE DataPort flights loaded')

    combined = pd.concat(dfs, ignore_index=True)
    n_attacked = combined[combined['attack_type'] != 'none']['flight_id'].nunique()
    print(f'[DataPort] {combined["flight_id"].nunique()} flights loaded, '
          f'{n_attacked} with GPS spoofing attacks')
    return combined
