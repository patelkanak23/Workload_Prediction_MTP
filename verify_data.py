"""
Step 3: Verify alibaba.csv loads correctly for MSCNet.

Tests the exact same logic as Dataset_Custom in data_loader.py
to confirm our preprocessed data is compatible before sending to Colab.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# Configuration (matching alibaba.sh exactly)
# ============================================================
ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
DATA_PATH = "alibaba.csv"
SEQ_LEN = 96
LABEL_LEN = 96     # from alibaba.sh: --label_len $seq_len
PRED_LENS = [24, 48, 72, 96]
FEATURES = "M"      # from alibaba.sh: --features M
TARGET = "OT"        # default target column
ENC_IN = 100         # from alibaba.sh: --enc_in 100

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} -- {detail}")


def main():
    global PASSED, FAILED

    csv_path = os.path.join(ROOT_PATH, DATA_PATH)
    print("=" * 60)
    print("Step 3: Data Loading Verification")
    print("=" * 60)
    print(f"  CSV: {csv_path}")
    print(f"  seq_len={SEQ_LEN}, label_len={LABEL_LEN}")
    print(f"  features={FEATURES}, enc_in={ENC_IN}")
    print()

    # --------------------------------------------------------
    # 1. Load CSV (mirrors Dataset_Custom.__read_data__)
    # --------------------------------------------------------
    print("[TEST 1] Loading CSV...")
    df_raw = pd.read_csv(csv_path)

    check("CSV loads without error", True)
    check("Has 'date' column", "date" in df_raw.columns, f"Columns: {list(df_raw.columns[:5])}")
    check("Has target 'OT' column", TARGET in df_raw.columns)

    n_features = len(df_raw.columns) - 1  # exclude 'date'
    check(f"Feature count = {ENC_IN + 1} (100 machines + OT)", n_features == ENC_IN + 1,
          f"Got {n_features}")

    print(f"  Shape: {df_raw.shape}")
    print(f"  Columns: {list(df_raw.columns[:5])} ... {list(df_raw.columns[-3:])}")
    print()

    # --------------------------------------------------------
    # 2. Column reordering (mirrors Dataset_Custom)
    # --------------------------------------------------------
    print("[TEST 2] Column reordering (target last)...")
    cols = list(df_raw.columns)
    cols.remove(TARGET)
    cols.remove("date")
    df_raw = df_raw[["date"] + cols + [TARGET]]

    check("Target 'OT' is last column", df_raw.columns[-1] == TARGET)
    check("'date' is first column", df_raw.columns[0] == "date")
    print()

    # --------------------------------------------------------
    # 3. Train/Val/Test split (7:1:2, mirrors Dataset_Custom)
    # --------------------------------------------------------
    print("[TEST 3] Train/Val/Test splits...")
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test

    print(f"  Total samples: {len(df_raw)}")
    print(f"  Train: {num_train}, Val: {num_vali}, Test: {num_test}")

    check("Train > 0", num_train > 0)
    check("Val > 0", num_vali > 0)
    check("Test > 0", num_test > 0)
    check("Splits sum to total", num_train + num_vali + num_test == len(df_raw))
    print()

    # --------------------------------------------------------
    # 4. Borders (exact Dataset_Custom logic)
    # --------------------------------------------------------
    print("[TEST 4] Split borders...")
    for pred_len in PRED_LENS:
        border1s = [0, num_train - SEQ_LEN, len(df_raw) - num_test - SEQ_LEN]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        for flag, flag_name in enumerate(["train", "val", "test"]):
            b1 = border1s[flag]
            b2 = border2s[flag]
            n_samples = b2 - b1 - SEQ_LEN - pred_len + 1

            check(f"H={pred_len} {flag_name}: border valid (b1={b1} < b2={b2})", b1 < b2)
            check(f"H={pred_len} {flag_name}: has samples ({n_samples})", n_samples > 0,
                  f"b1={b1}, b2={b2}, seq={SEQ_LEN}, pred={pred_len}")
    print()

    # --------------------------------------------------------
    # 5. Feature selection (features='M')
    # --------------------------------------------------------
    print("[TEST 5] Feature selection (--features M)...")
    if FEATURES == "M" or FEATURES == "MS":
        cols_data = df_raw.columns[1:]  # all except 'date'
        df_data = df_raw[cols_data]
    elif FEATURES == "S":
        df_data = df_raw[[TARGET]]

    check(f"Feature matrix shape: ({len(df_data)}, {len(df_data.columns)})",
          df_data.shape[1] == ENC_IN + 1,  # 100 machines + OT
          f"Expected {ENC_IN + 1}, got {df_data.shape[1]}")
    print()

    # --------------------------------------------------------
    # 6. StandardScaler (fit on train only)
    # --------------------------------------------------------
    print("[TEST 6] StandardScaler normalization...")
    scaler = StandardScaler()
    train_data = df_data[border1s[0]:border2s[0]]
    scaler.fit(train_data.values)
    data = scaler.transform(df_data.values)

    check("No NaN after scaling", not np.isnan(data).any())
    check("No Inf after scaling", not np.isinf(data).any())
    check("Train mean ~ 0", abs(data[:num_train].mean()) < 0.1,
          f"Mean = {data[:num_train].mean():.4f}")
    print()

    # --------------------------------------------------------
    # 7. Sliding window shapes (simulates __getitem__)
    # --------------------------------------------------------
    print("[TEST 7] Sliding window shapes...")
    for pred_len in PRED_LENS:
        b1 = border1s[0]  # train
        b2 = border2s[0]
        data_x = data[b1:b2]
        data_y = data[b1:b2]

        # First sample
        idx = 0
        s_begin = idx
        s_end = s_begin + SEQ_LEN
        r_begin = s_end - LABEL_LEN
        r_end = r_begin + LABEL_LEN + pred_len

        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]

        expected_x_shape = (SEQ_LEN, ENC_IN + 1)
        expected_y_shape = (LABEL_LEN + pred_len, ENC_IN + 1)

        check(f"H={pred_len} seq_x shape: {seq_x.shape} == {expected_x_shape}",
              seq_x.shape == expected_x_shape)
        check(f"H={pred_len} seq_y shape: {seq_y.shape} == {expected_y_shape}",
              seq_y.shape == expected_y_shape)
    print()

    # --------------------------------------------------------
    # 8. Date parsing (time features)
    # --------------------------------------------------------
    print("[TEST 8] Date parsing...")
    try:
        dates = pd.to_datetime(df_raw["date"])
        check("Dates parse correctly", True)
        check(f"Date range: {dates.iloc[0]} to {dates.iloc[-1]}", True)

        # Check 5-minute frequency
        diffs = dates.diff().dropna()
        mode_diff = diffs.mode().iloc[0]
        check(f"Modal time interval: {mode_diff}", mode_diff == pd.Timedelta("5min"),
              f"Got {mode_diff}")
    except Exception as e:
        check("Dates parse correctly", False, str(e))
    print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("=" * 60)
    total = PASSED + FAILED
    if FAILED == 0:
        print(f"ALL {total} CHECKS PASSED!")
        print("Data is ready for MSCNet training on Colab.")
    else:
        print(f"{PASSED}/{total} passed, {FAILED} FAILED")
        print("Fix the failures before proceeding to training.")
    print("=" * 60)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
