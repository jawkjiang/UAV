"""
Instructions for downloading real-world datasets for cross-domain evaluation.

Run this script to check which datasets are already available.
It will print instructions for any missing datasets.
"""
import os, sys

STEP9_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(STEP9_DIR, 'data')

def check_dataset(name, path, url, instructions):
    exists = os.path.isdir(path) and any(
        f.endswith('.csv') for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    ) if os.path.isdir(path) else False

    status = "OK" if exists else "MISSING"
    print(f"\n[{status}] {name}")
    if not exists:
        print(f"  URL: {url}")
        print(f"  Target: {path}")
        print(f"  Steps:")
        for line in instructions:
            print(f"    {line}")
    else:
        n = sum(1 for f in os.listdir(path) if f.endswith('.csv'))
        print(f"  Found {n} CSV files in {path}")

check_dataset(
    name="ALFA Dataset (CMU AirLab)",
    path=os.path.join(DATA_DIR, 'alfa'),
    url="https://kilthub.cmu.edu/articles/dataset/ALFA_A_Dataset_for_UAV_Fault_and_Anomaly_Detection/12707963",
    instructions=[
        "1. Go to the URL above, click 'Download all'",
        "2. Extract the ZIP",
        "3. Place all *_flight.csv files in:",
        f"   {os.path.join(DATA_DIR, 'alfa')}/",
        "Note: ALFA has 47 fixed-wing UAV flights with fault labels.",
        "We use the normal segments and inject synthetic GPS step attacks."
    ]
)

check_dataset(
    name="IEEE DataPort UAV Attack Dataset",
    path=os.path.join(DATA_DIR, 'ieee_dataport', 'spoofing'),
    url="https://ieee-dataport.org/open-access/uav-attack-dataset",
    instructions=[
        "1. Go to the URL above (free IEEE account required), download all files",
        "2. Extract and convert ULOG → CSV using: pip install pyulog && ulog2csv <file.ulg>",
        "3. Place normal flight CSVs in:",
        f"   {os.path.join(DATA_DIR, 'ieee_dataport', 'normal')}/",
        "   Place spoofing flight CSVs in:",
        f"   {os.path.join(DATA_DIR, 'ieee_dataport', 'spoofing')}/",
        "Note: Dataset contains Holybro S500 + Pixhawk 4 + HackRF spoofing attacks."
    ]
)

print("\nAfter downloading, run:")
print("  python step9_realworld/cross_domain_eval.py --dataset both")
