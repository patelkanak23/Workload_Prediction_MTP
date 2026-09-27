"""
Alibaba Cluster Trace 2018 → Preprocessed CSV for MSCNet

This script downloads and preprocesses the Alibaba Cluster Trace 2018
machine_usage data into the CSV format expected by MSCNet's Dataset_Custom.

Steps:
  1. Download machine_usage.tar.gz (~1.7 GB) from Alibaba OSS
  2. Extract the CSV file(s)
  3. Sample 100 machines randomly (seed=2022 for reproducibility)
  4. Extract CPU utilization column
  5. Pivot to wide format: date × 100 machines
  6. Handle missing values (forward-fill, then zero-fill)
  7. Output alibaba.csv with columns: date, m_0, m_1, ..., m_99, OT

Usage:
  python alibaba_preprocess.py                     # Full pipeline (download + preprocess)
  python alibaba_preprocess.py --skip-download     # Preprocess only (if already downloaded)
  python alibaba_preprocess.py --raw-path /path/to/machine_usage.csv  # Use existing file

Output:
  ../dataset/alibaba.csv
"""

import os
import sys
import argparse
import tarfile
import urllib.request
import shutil
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================
DOWNLOAD_URL = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz"
SEED = 2022
N_MACHINES = 100

# machine_usage.csv has NO header. Columns from the official schema:
COLUMNS = [
    "machine_id",     # string: uid of machine
    "time_stamp",     # double: time stamp, in second
    "cpu_util_percent",  # bigint: [0, 100] cpu utilization percentage
    "mem_util_percent",  # bigint: [0, 100] memory utilization percentage
    "mem_gps",        # double: memory bandwidth (GB/s)
    "mpki",           # bigint: cache miss per kilo instructions
    "net_in",         # double: network in bandwidth
    "net_out",        # double: network out bandwidth
    "disk_io_percent" # double: disk io percentage
]

# We only need machine_id, time_stamp, and cpu_util_percent
USE_COLS = [0, 1, 2]  # machine_id, time_stamp, cpu_util_percent


def download_data(download_dir: str) -> str:
    """Download machine_usage.tar.gz from Alibaba OSS."""
    os.makedirs(download_dir, exist_ok=True)
    tar_path = os.path.join(download_dir, "machine_usage.tar.gz")

    if os.path.exists(tar_path):
        print(f"[INFO] File already exists: {tar_path}")
        return tar_path

    print(f"[INFO] Downloading machine_usage.tar.gz (~1.7 GB)...")
    print(f"[INFO] URL: {DOWNLOAD_URL}")
    print(f"[INFO] This may take 10-30 minutes depending on your connection.")

    def report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {pct:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
        else:
            mb_down = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  Downloaded: {mb_down:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, tar_path, reporthook=report_hook)
        print(f"\n[INFO] Download complete: {tar_path}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print(f"[INFO] You can manually download from: {DOWNLOAD_URL}")
        print(f"[INFO] Place the file at: {tar_path}")
        sys.exit(1)

    return tar_path


def extract_data(tar_path: str, extract_dir: str) -> str:
    """Extract machine_usage CSV from tar.gz."""
    # Check if already extracted
    csv_candidates = list(Path(extract_dir).glob("**/machine_usage*.csv"))
    if csv_candidates:
        print(f"[INFO] Already extracted: {csv_candidates[0]}")
        return str(csv_candidates[0])

    print(f"[INFO] Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    csv_candidates = list(Path(extract_dir).glob("**/machine_usage*.csv"))
    if not csv_candidates:
        # The tar might contain multiple numbered CSV files
        csv_candidates = list(Path(extract_dir).glob("**/*.csv"))

    if not csv_candidates:
        print(f"[ERROR] No CSV files found after extraction in {extract_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(csv_candidates)} CSV file(s)")
    return str(csv_candidates[0]) if len(csv_candidates) == 1 else extract_dir


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw machine_usage data, handling both single and multi-file cases."""
    data_path = Path(data_path)

    if data_path.is_file():
        print(f"[INFO] Loading single file: {data_path}")
        df = pd.read_csv(
            data_path,
            header=None,
            names=COLUMNS,
            usecols=USE_COLS,
            dtype={"machine_id": str, "time_stamp": float, "cpu_util_percent": float},
        )
    elif data_path.is_dir():
        # Multiple CSV files in directory
        csv_files = sorted(data_path.glob("*.csv"))
        print(f"[INFO] Loading {len(csv_files)} CSV files from {data_path}")
        dfs = []
        for i, f in enumerate(csv_files):
            print(f"  Loading {f.name} ({i+1}/{len(csv_files)})...")
            chunk = pd.read_csv(
                f,
                header=None,
                names=COLUMNS,
                usecols=USE_COLS,
                dtype={"machine_id": str, "time_stamp": float, "cpu_util_percent": float},
            )
            dfs.append(chunk)
        df = pd.concat(dfs, ignore_index=True)
    else:
        print(f"[ERROR] Path not found: {data_path}")
        sys.exit(1)

    print(f"[INFO] Loaded {len(df):,} rows, {df['machine_id'].nunique()} unique machines")
    return df


def preprocess(df: pd.DataFrame, output_path: str):
    """
    Preprocess raw machine_usage into MSCNet-compatible CSV.

    Steps:
    1. Drop rows with missing CPU utilization
    2. Sample 100 machines (seed=2022)
    3. Convert timestamps to 5-minute bins
    4. Pivot to wide format (date × machines)
    5. Forward-fill + zero-fill missing values
    6. Add 'OT' column (mean CPU across machines) as the target
    7. Save to CSV
    """

    # 1. Clean: drop NaN CPU values
    initial_len = len(df)
    df = df.dropna(subset=["cpu_util_percent"])
    print(f"[INFO] Dropped {initial_len - len(df):,} rows with missing CPU values")

    # Clip CPU to [0, 100]
    df["cpu_util_percent"] = df["cpu_util_percent"].clip(0, 100)

    # 2. Sample 100 machines
    all_machines = sorted(df["machine_id"].unique())
    print(f"[INFO] Total unique machines: {len(all_machines)}")

    rng = np.random.RandomState(SEED)
    if len(all_machines) < N_MACHINES:
        print(f"[WARNING] Only {len(all_machines)} machines available, using all")
        sampled_machines = all_machines
    else:
        sampled_machines = rng.choice(all_machines, size=N_MACHINES, replace=False)
        sampled_machines = sorted(sampled_machines)

    df = df[df["machine_id"].isin(sampled_machines)].copy()
    print(f"[INFO] After sampling {len(sampled_machines)} machines: {len(df):,} rows")

    # 3. Convert timestamps to datetime and bin into 5-minute intervals
    # The Alibaba trace timestamps are seconds since start of trace
    # We create synthetic dates starting from 2018-01-01
    base_time = pd.Timestamp("2018-01-01 00:00:00")
    df["datetime"] = base_time + pd.to_timedelta(df["time_stamp"], unit="s")

    # Bin into 5-minute intervals (300 seconds)
    df["datetime"] = df["datetime"].dt.floor("5min")

    # Average CPU within each 5-min bin for each machine
    df = df.groupby(["machine_id", "datetime"])["cpu_util_percent"].mean().reset_index()
    print(f"[INFO] After 5-min aggregation: {len(df):,} rows")

    # 4. Pivot to wide format: rows=datetime, columns=machines
    pivot = df.pivot(index="datetime", columns="machine_id", values="cpu_util_percent")

    # Rename columns to m_0, m_1, ..., m_99
    machine_map = {m: f"m_{i}" for i, m in enumerate(pivot.columns)}
    pivot = pivot.rename(columns=machine_map)

    # Sort by datetime
    pivot = pivot.sort_index()
    print(f"[INFO] Pivot shape: {pivot.shape} (timesteps × machines)")

    # 5. Fill missing values
    missing_pct = pivot.isna().mean().mean() * 100
    print(f"[INFO] Missing values before fill: {missing_pct:.2f}%")

    pivot = pivot.ffill().bfill().fillna(0)

    # 6. Add OT column (mean CPU across all machines — used as default target)
    pivot["OT"] = pivot.mean(axis=1)

    # 7. Add date column and save
    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={"datetime": "date"})
    pivot["date"] = pivot["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pivot.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Saved preprocessed data to: {output_path}")
    print(f"  Shape: {pivot.shape}")
    print(f"  Columns: date + {N_MACHINES} machine columns + OT")
    print(f"  Time range: {pivot['date'].iloc[0]} to {pivot['date'].iloc[-1]}")
    print(f"  Timesteps: {len(pivot)}")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

    # Print split info for MSCNet (7:1:2)
    n = len(pivot)
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test
    print(f"\n  MSCNet splits (7:1:2):")
    print(f"    Train: {n_train} samples")
    print(f"    Val:   {n_val} samples")
    print(f"    Test:  {n_test} samples")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Alibaba Cluster Trace 2018 for MSCNet")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing files")
    parser.add_argument("--raw-path", type=str, default=None,
                        help="Path to existing machine_usage.csv or directory of CSVs")
    parser.add_argument("--download-dir", type=str, default=None,
                        help="Directory to store downloaded/extracted data")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    if args.download_dir is None:
        args.download_dir = os.path.join(project_dir, "raw_data")
    if args.output is None:
        args.output = os.path.join(project_dir, "dataset", "alibaba.csv")

    print("=" * 60)
    print("Alibaba Cluster Trace 2018 → MSCNet Preprocessing")
    print("=" * 60)
    print(f"  Seed: {SEED}")
    print(f"  Machines to sample: {N_MACHINES}")
    print(f"  Output: {args.output}")
    print()

    if args.raw_path:
        # User provided existing data
        raw_path = args.raw_path
    elif args.skip_download:
        # Look for already-downloaded data
        extract_dir = os.path.join(args.download_dir, "extracted")
        csv_candidates = list(Path(extract_dir).glob("**/*.csv"))
        if not csv_candidates:
            print("[ERROR] --skip-download specified but no CSVs found in", extract_dir)
            sys.exit(1)
        raw_path = str(csv_candidates[0]) if len(csv_candidates) == 1 else extract_dir
    else:
        # Full pipeline: download + extract
        tar_path = download_data(args.download_dir)
        extract_dir = os.path.join(args.download_dir, "extracted")
        raw_path = extract_data(tar_path, extract_dir)

    # Load and preprocess
    df = load_raw_data(raw_path)
    preprocess(df, args.output)


if __name__ == "__main__":
    main()
