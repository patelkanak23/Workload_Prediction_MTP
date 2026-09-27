# Data Preprocessing

Scripts to convert raw cloud workload traces into the CSV format expected by MSCNet's `Dataset_Custom` loader.

## Expected output format

```csv
date,machine_0,machine_1,...,machine_99
2018-01-01 00:00:00,45.2,32.1,...,67.8
2018-01-01 00:05:00,47.8,33.5,...,65.2
```

- **100 columns** (one per sampled machine's CPU utilization)
- **`date` column** with ISO format timestamps
- **5-minute aggregation** from raw 10-second sampling
- Target column for `--features S` mode: `OT` (last column)
