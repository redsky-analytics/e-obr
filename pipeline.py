"""
Orderbook windowing pipeline (per-asset, in-memory).

Source data is the Hive-partitioned Parquet export written by export_example.sql:
    gs://<SOURCE_BUCKET>/<SOURCE_PREFIX>/trade_date=YYYY-MM-DD/asset=<A>/part-*.parquet

For every row i in an asset's concatenated stream (sorted by nanoseconds_start,
seq_nbr) where the previous PREV_N spreads and the next POST_N spreads contain
no nulls, emit:
    prev_30 = spreads[i - PREV_N + 1 .. i]   (chronological, includes self)
    post_50 = spreads[i .. i + POST_N - 1]   (chronological, includes self)

One asset at a time, fully in memory (largest asset is ~500MB Parquet).

Commands:
    run       <asset>   read GCS → numba kernel → write features back to GCS
    all                 run for every asset, sequentially
    external            CREATE OR REPLACE EXTERNAL TABLE over the features Parquets
    list                print all distinct assets present in the GCS source
    inspect   <asset>   dump the Parquet schema of every partition file for an
                        asset and flag any cross-file schema differences (debug)

The features Parquets are written Hive-partitioned by asset:
    gs://<GCS_BUCKET>/<GCS_PREFIX>/asset=<A>/data.parquet
The `asset` column is omitted from the file payload — BigQuery's hive partition
discovery re-attaches it from the path, which avoids the partition/column name
collision an external table would otherwise reject.
"""

import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import gcsfs
from numba import njit
from google.cloud import bigquery

# ============================================================================
# CONFIG  — env-driven
# ============================================================================
BQ_DATASET       = os.environ["BQ_DATASET"]                                # "project.dataset"
PROJECT          = BQ_DATASET.split(".", 1)[0]
LOCATION         = os.environ.get("BQ_LOCATION", "US")
OUTPUT_TABLE     = f"{BQ_DATASET}.Orderbook_Windows"

# Input: Hive-partitioned Parquet written by export_example.sql
SOURCE_BUCKET    = "bkt-pr-usc1-use5-2664-cdrdsml-dflt"
SOURCE_PREFIX    = os.environ.get("SOURCE_PREFIX", "downsampled_orderbook")

# Output: features Parquet (queried via the external table)
GCS_BUCKET       = os.environ.get("GCS_BUCKET", SOURCE_BUCKET)
GCS_PREFIX       = os.environ.get("GCS_PREFIX", "orderbook_windows")

# Algorithm params — fixed, not deployment config
PREV_N = 300
POST_N = 500

# ============================================================================
# Numba kernel
# ============================================================================
@njit(cache=True, boundscheck=False)
def build_windows(spread, prev_n, post_n):
    """
    Single-asset rolling window extraction.

    Row i qualifies iff there is no NaN in spread[i - prev_n + 1 .. i + post_n - 1].
    For each qualifying row, return (row_idx, prev_window, post_window).

    The original SQL used a looser filter (only the boundary value had to be
    non-null). This matches the stricter "no null anywhere in window" semantic
    from the reference snippet.
    """
    n = spread.shape[0]
    if n < prev_n + post_n - 1:
        return (np.empty(0, np.int64),
                np.empty((0, prev_n), np.float64),
                np.empty((0, post_n), np.float64))

    # nan_after[j] = smallest k >= j where spread[k] is NaN, else n.
    # A row i is valid iff nan_after[i - prev_n + 1] >= i + post_n.
    nan_after = np.empty(n, np.int64)
    last = n
    for j in range(n - 1, -1, -1):
        if np.isnan(spread[j]):
            last = j
        nan_after[j] = last

    # Pass 1: count valid rows
    n_out = 0
    for i in range(prev_n - 1, n - post_n + 1):
        if nan_after[i - prev_n + 1] >= i + post_n:
            n_out += 1

    prev_out = np.empty((n_out, prev_n), np.float64)
    post_out = np.empty((n_out, post_n), np.float64)
    row_idx  = np.empty(n_out, np.int64)

    # Pass 2: fill outputs
    k = 0
    for i in range(prev_n - 1, n - post_n + 1):
        if nan_after[i - prev_n + 1] >= i + post_n:
            for j in range(prev_n):
                prev_out[k, j] = spread[i - prev_n + 1 + j]
            for j in range(post_n):
                post_out[k, j] = spread[i + j]
            row_idx[k] = i
            k += 1

    return row_idx, prev_out, post_out

# ============================================================================
# Clients (lazy)
# ============================================================================
_bq_client = None
_gcs_fs = None

def bq():
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=PROJECT, location=LOCATION)
    return _bq_client

def gcs():
    global _gcs_fs
    if _gcs_fs is None:
        _gcs_fs = gcsfs.GCSFileSystem(project=PROJECT)
    return _gcs_fs

# ============================================================================
# Asset listing
# ============================================================================
def list_assets():
    fs = gcs()
    paths = fs.glob(f"{SOURCE_BUCKET}/{SOURCE_PREFIX}/trade_date=*/asset=*")
    return sorted({p.split("asset=", 1)[1].rstrip("/") for p in paths if "asset=" in p})

# ============================================================================
# Per-asset pipeline:  GCS Parquet → numba kernel → GCS features Parquet
# ============================================================================
def _normalize_trade_date(t):
    """BigQuery's Parquet export sometimes writes trade_date as dictionary<string>
    instead of date32[day], which makes concat_tables across partitions fail. Cast
    everything to date32 before merging."""
    if "trade_date" not in t.column_names:
        return t
    col = t.column("trade_date")
    if col.type == pa.date32():
        return t
    return t.set_column(
        t.schema.get_field_index("trade_date"),
        "trade_date",
        col.cast(pa.date32()),
    )

def run_one(asset):
    fs = gcs()

    t0 = time.time()
    files = fs.glob(f"{SOURCE_BUCKET}/{SOURCE_PREFIX}/trade_date=*/asset={asset}/*.parquet")
    if not files:
        print(f"[{asset}] no orderbook data")
        return
    # ParquetFile bypasses pq.read_table's default partitioning="hive" — the
    # `trade_date=…/asset=…` path would otherwise be auto-discovered, adding a
    # dictionary<string> trade_date column that collides with the date32 column
    # already in the file.
    tables = [_normalize_trade_date(pq.ParquetFile(f, filesystem=fs).read()) for f in files]
    tbl = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    tbl = tbl.sort_by([("nanoseconds_start", "ascending"), ("seq_nbr", "ascending")])
    t_dl = time.time() - t0

    t1 = time.time()
    spread = tbl.column("spread").to_numpy(zero_copy_only=False)
    if spread.dtype != np.float64:
        spread = spread.astype(np.float64)
    row_idx, prev_w, post_w = build_windows(spread, PREV_N, POST_N)
    if row_idx.size == 0:
        print(f"[{asset}] dl {t_dl:.1f}s, no valid windows")
        return
    sliced = tbl.take(pa.array(row_idx))
    if "asset" in sliced.column_names:
        sliced = sliced.drop(["asset"])
    out = sliced.append_column(
        "prev_30",
        pa.FixedSizeListArray.from_arrays(pa.array(prev_w.ravel(), pa.float64()), PREV_N),
    )
    out = out.append_column(
        "post_50",
        pa.FixedSizeListArray.from_arrays(pa.array(post_w.ravel(), pa.float64()), POST_N),
    )
    t_proc = time.time() - t1

    t2 = time.time()
    remote = f"{GCS_BUCKET}/{GCS_PREFIX}/asset={asset}/data.parquet"
    with fs.open(remote, "wb") as f:
        pq.write_table(out, f, compression="zstd")
    t_up = time.time() - t2

    print(f"[{asset}] dl {t_dl:.1f}s  proc {t_proc:.1f}s  up {t_up:.1f}s  → gs://{remote}")

# ============================================================================
# Inspect:  dump per-file Parquet schemas, surface cross-file mismatches
# ============================================================================
def inspect_asset(asset):
    fs = gcs()
    files = fs.glob(f"{SOURCE_BUCKET}/{SOURCE_PREFIX}/trade_date=*/asset={asset}/*.parquet")
    if not files:
        print(f"[{asset}] no files")
        return
    print(f"[{asset}] {len(files)} file(s)")

    schemas = {}   # str(schema) -> list[path]
    for f in files:
        pf = pq.ParquetFile(f, filesystem=fs)
        s = pf.schema_arrow
        nrows = pf.metadata.num_rows
        key = "\n".join(f"  {fld.name}: {fld.type}" for fld in s)
        schemas.setdefault(key, []).append((f, nrows))

    for i, (key, members) in enumerate(schemas.items(), 1):
        total_rows = sum(n for _, n in members)
        print(f"\n--- schema #{i}  ({len(members)} files, {total_rows:,} rows) ---")
        print(key)
        for f, n in members[:3]:
            print(f"  e.g. {f} ({n:,} rows)")
        if len(members) > 3:
            print(f"  ... and {len(members) - 3} more")

    if len(schemas) > 1:
        print(f"\nWARNING: {len(schemas)} distinct schemas across files for {asset}")
    else:
        print(f"\nOK: all files share one schema")

# ============================================================================
# External table:  CREATE OR REPLACE EXTERNAL TABLE over the GCS features
# ============================================================================
def create_external_table():
    sql = f"""
    CREATE OR REPLACE EXTERNAL TABLE `{OUTPUT_TABLE}`
    WITH PARTITION COLUMNS
    OPTIONS (
      format = 'PARQUET',
      uris = ['gs://{GCS_BUCKET}/{GCS_PREFIX}/*'],
      hive_partition_uri_prefix = 'gs://{GCS_BUCKET}/{GCS_PREFIX}/',
      require_hive_partition_filter = false
    )
    """
    bq().query(sql).result()
    return OUTPUT_TABLE

# ============================================================================
# Driver
# ============================================================================
def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return
    cmd = argv[1]
    if cmd == "list":
        for a in list_assets():
            print(a)
    elif cmd == "all":
        for a in list_assets():
            try:
                run_one(a)
            except Exception as e:
                print(f"FAILED {a}: {e!r}")
    elif cmd == "external":
        name = create_external_table()
        print(f"defined external table {name}")
    elif cmd == "run":
        if len(argv) < 3:
            print(f"usage: {argv[0]} run <asset>")
            return
        run_one(argv[2])
    elif cmd == "inspect":
        if len(argv) < 3:
            print(f"usage: {argv[0]} inspect <asset>")
            return
        inspect_asset(argv[2])
    else:
        print(f"unknown command: {cmd}")
        print(__doc__)

if __name__ == "__main__":
    main(sys.argv)
