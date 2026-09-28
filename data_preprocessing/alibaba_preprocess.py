"""
Alibaba Cluster Trace 2018 -> Preprocessed CSV for MSCNet
(Memory-efficient chunked version)

Handles the full 247M-row machine_usage.csv (~8.6 GB) by processing
in chunks, so it works on machines with limited RAM.

Usage:
  python alibaba_preprocess.py                     # Full pipeline (download + preprocess)
  python alibaba_preprocess.py --skip-download     # Preprocess only (if already downloaded)
  python alibaba_preprocess.py --raw-path /path/to/machine_usage.csv

Output:
  ../dataset/alibaba.csv
"""

import os
import sys
import argparse
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================
DOWNLOAD_URL = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz"
SEED = 2022
N_MACHINES = 100
CHUNK_SIZE = 5_000_000  # Read 5M rows at a time

# machine_usage.csv has NO header. Columns from the official schema:
COLUMN_NAMES = [
    "machine_id",        # string: uid of machine
    "time_stamp",        # double: seconds since trace start
    "cpu_util_percent",  # int: [0, 100]
    "mem_util_percent",  # int: [0, 100]
    "mem_gps",           # double
    "mpki",              # int
    "net_in",            # double
    "net_out",           # double
    "disk_io_percent",   # double
]

# We only need these 3 columns
USE_COLS = [0, 1, 2]
USE_NAMES = ["machine_id", "time_stamp", "cpu_util_percent"]
DTYPES = {"machine_id": str, "time_stamp": float, "cpu_util_percent": float}


def download_data(download_dir):
    """Download machine_usage.tar.gz from Alibaba OSS."""
    os.makedirs(download_dir, exist_ok=True)
    tar_path = os.path.join(download_dir, "machine_usage.tar.gz")

    if os.path.exists(tar_path):
        print(f"[INFO] Already downloaded: {tar_path}")
        return tar_path

    print(f"[INFO] Downloading machine_usage.tar.gz (~1.7 GB)...")
    print(f"[INFO] URL: {DOWNLOAD_URL}")

    def report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {pct:.1f}% ({mb:.0f}/{total_mb:.0f} MB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, tar_path, reporthook=report_hook)
        print(f"\n[INFO] Download complete: {tar_path}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print(f"[INFO] Manually download from: {DOWNLOAD_URL}")
        sys.exit(1)

    return tar_path


def extract_data(tar_path, extract_dir):
    """Extract machine_usage CSV from tar.gz."""
    csv_candidates = list(Path(extract_dir).glob("**/machine_usage*.csv"))
    if csv_candidates:
        print(f"[INFO] Already extracted: {csv_candidates[0]}")
        return str(csv_candidates[0])

    print(f"[INFO] Extracting {tar_path}...")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    csv_candidates = list(Path(extract_dir).glob("**/*.csv"))
    if not csv_candidates:
        print(f"[ERROR] No CSV files found after extraction")
        sys.exit(1)

    print(f"[INFO] Found {len(csv_candidates)} CSV file(s)")
    return str(csv_candidates[0])


def pass1_sample_machines(csv_path):
    """
    Pass 1: Scan the CSV in chunks to discover all unique machine IDs,
    then randomly sample N_MACHINES of them.
    """
    print(f"\n[PASS 1] Scanning for unique machine IDs...")
    all_machine_ids = set()
    total_rows = 0

    reader = pd.read_csv(
        csv_path, header=None, names=COLUMN_NAMES, usecols=[0],
        dtype={"machine_id": str}, chunksize=CHUNK_SIZE,
    )

    for i, chunk in enumerate(reader):
        all_machine_ids.update(chunk["machine_id"].unique())
        total_rows += len(chunk)
        print(f"  Chunk {i+1}: {total_rows:,} rows scanned, {len(all_machine_ids)} unique machines so far", end="\r")

    print(f"\n[PASS 1] Done: {total_rows:,} total rows, {len(all_machine_ids)} unique machines")

    # Sample 100 machines
    all_machine_ids = sorted(all_machine_ids)
    rng = np.random.RandomState(SEED)
    sampled = rng.choice(all_machine_ids, size=min(N_MACHINES, len(all_machine_ids)), replace=False)
    sampled = set(sampled)
    print(f"[PASS 1] Sampled {len(sampled)} machines (seed={SEED})")

    return sampled, total_rows


def pass2_filter_and_aggregate(csv_path, sampled_machines):
    """
    Pass 2: Read the CSV in chunks, filter to only sampled machines,
    aggregate CPU utilization into 5-minute bins.
    """
    print(f"\n[PASS 2] Filtering to {len(sampled_machines)} machines and aggregating...")
    filtered_chunks = []
    total_rows = 0
    kept_rows = 0

    reader = pd.read_csv(
        csv_path, header=None, names=COLUMN_NAMES, usecols=USE_COLS,
        dtype=DTYPES, chunksize=CHUNK_SIZE,
    )

    for i, chunk in enumerate(reader):
        total_rows += len(chunk)

        # Filter to sampled machines
        mask = chunk["machine_id"].isin(sampled_machines)
        filtered = chunk[mask].copy()

        if len(filtered) > 0:
            # Drop NaN CPU values
            filtered = filtered.dropna(subset=["cpu_util_percent"])
            # Clip to [0, 100]
            filtered["cpu_util_percent"] = filtered["cpu_util_percent"].clip(0, 100)

            if len(filtered) > 0:
                # Convert timestamp to 5-minute bins (300 seconds)
                filtered["time_bin"] = (filtered["time_stamp"] // 300).astype(int)
                # Average CPU within each bin for each machine
                agg = filtered.groupby(["machine_id", "time_bin"])["cpu_util_percent"].mean().reset_index()
                filtered_chunks.append(agg)
                kept_rows += len(agg)

        print(f"  Chunk {i+1}: {total_rows:,} rows processed, {kept_rows:,} aggregated rows kept", end="\r")

    print(f"\n[PASS 2] Done: {kept_rows:,} aggregated rows from {total_rows:,} total")

    # Combine all filtered chunks
    df = pd.concat(filtered_chunks, ignore_index=True)

    # Some time_bins might have been split across chunks, re-aggregate
    df = df.groupby(["machine_id", "time_bin"])["cpu_util_percent"].mean().reset_index()
    print(f"[PASS 2] After final aggregation: {len(df):,} rows")

    return df


def build_csv(df, output_path):
    """
    Convert the filtered+aggregated data into MSCNet-compatible CSV format.
    """
    print(f"\n[BUILD] Creating MSCNet-compatible CSV...")

    # Convert time_bin back to datetime
    base_time = pd.Timestamp("2018-01-01 00:00:00")
    df["datetime"] = base_time + pd.to_timedelta(df["time_bin"] * 300, unit="s")

    # Pivot: rows=datetime, columns=machines
    pivot = df.pivot_table(
        index="datetime", columns="machine_id",
        values="cpu_util_percent", aggfunc="mean"
    )

    # Rename columns to m_0, m_1, ..., m_99
    machine_map = {m: f"m_{i}" for i, m in enumerate(sorted(pivot.columns))}
    pivot = pivot.rename(columns=machine_map)
    pivot = pivot[sorted(pivot.columns, key=lambda x: int(x.split("_")[1]))]

    # Sort by time
    pivot = pivot.sort_index()
    print(f"[BUILD] Pivot shape: {pivot.shape} (timesteps x machines)")

    # Fill missing values
    missing_pct = pivot.isna().mean().mean() * 100
    print(f"[BUILD] Missing values: {missing_pct:.2f}%")
    pivot = pivot.ffill().bfill().fillna(0)

    # Add OT column (mean CPU across all machines - default target for --features S)
    pivot["OT"] = pivot.mean(axis=1)

    # Format date column
    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={"datetime": "date"})
    pivot["date"] = pivot["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pivot.to_csv(output_path, index=False, float_format="%.4f")

    # Print summary
    n = len(pivot)
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test

    print(f"\n{'='*60}")
    print(f"[SUCCESS] Saved: {output_path}")
    print(f"  Shape: {pivot.shape}")
    print(f"  Columns: date + {len(pivot.columns) - 2} machines + OT")
    print(f"  Time range: {pivot['date'].iloc[0]} to {pivot['date'].iloc[-1]}")
    print(f"  Timesteps: {n}")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print(f"  MSCNet splits (7:1:2):")
    print(f"    Train: {n_train}")
    print(f"    Val:   {n_val}")
    print(f"    Test:  {n_test}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Alibaba Cluster Trace 2018 for MSCNet")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--raw-path", type=str, default=None,
                        help="Path to existing machine_usage.csv")
    parser.add_argument("--download-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    if args.download_dir is None:
        args.download_dir = os.path.join(project_dir, "raw_data")
    if args.output is None:
        args.output = os.path.join(project_dir, "dataset", "alibaba.csv")

    print("=" * 60)
    print("Alibaba Cluster Trace 2018 -> MSCNet Preprocessing")
    print("(Memory-efficient chunked version)")
    print("=" * 60)
    print(f"  Seed: {SEED}")
    print(f"  Machines to sample: {N_MACHINES}")
    print(f"  Chunk size: {CHUNK_SIZE:,} rows")
    print(f"  Output: {args.output}")
    print()

    # Resolve raw data path
    if args.raw_path:
        csv_path = args.raw_path
    elif args.skip_download:
        candidates = list(Path(args.download_dir).glob("**/machine_usage*.csv"))
        if not candidates:
            print("[ERROR] No CSV found. Remove --skip-download to download.")
            sys.exit(1)
        csv_path = str(candidates[0])
    else:
        tar_path = download_data(args.download_dir)
        extract_dir = os.path.join(args.download_dir, "extracted")
        csv_path = extract_data(tar_path, extract_dir)

    print(f"[INFO] Raw data: {csv_path}")

    # Two-pass processing
    sampled_machines, total_rows = pass1_sample_machines(csv_path)
    df = pass2_filter_and_aggregate(csv_path, sampled_machines)
    build_csv(df, args.output)


if __name__ == "__main__":
    main()
