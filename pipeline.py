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

Flags:
    --local             write features to LOCAL_OUTPUT_DIR/asset=<A>/data.parquet
                        instead of GCS (works for `run` and `all`)

The features Parquets are written Hive-partitioned by asset:
    gs://<GCS_BUCKET>/<GCS_PREFIX>/asset=<A>/data.parquet
The `asset` column is omitted from the file payload — BigQuery's hive partition
discovery re-attaches it from the path, which avoids the partition/column name
collision an external table would otherwise reject.
"""

import os
import sys
import time
from pathlib import Path

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

# Optional local output destination (used when --local is passed)
LOCAL_OUTPUT_DIR = Path(os.environ.get("LOCAL_OUTPUT_DIR", "./data/features"))

# Algorithm params — fixed, not deployment config
PREV_N = 300
POST_N = 500
# Streaming write: rows of output per row group (and per kernel allocation)
# 250k × (300+500) × 4 bytes ≈ 800 MB peak per chunk.
CHUNK_SIZE = 250_000

# ============================================================================
# Numba kernels
#
# Split into two passes so that the (huge) prev/post arrays are only ever
# allocated for one chunk at a time:
#   1. find_valid_indices  → returns int64 array of qualifying input row indices.
#   2. fill_windows_chunk  → fills float32 prev/post for valid_idx[start:stop].
#
# Row i qualifies iff there is no NaN in spread[i - prev_n + 1 .. i + post_n - 1]
# (stricter than the original SQL's boundary-only check).
# ============================================================================
@njit(cache=True, boundscheck=False)
def find_valid_indices(spread, prev_n, post_n):
    n = spread.shape[0]
    if n < prev_n + post_n - 1:
        return np.empty(0, np.int64)

    # nan_after[j] = smallest k >= j where spread[k] is NaN, else n.
    nan_after = np.empty(n, np.int64)
    last = n
    for j in range(n - 1, -1, -1):
        if np.isnan(spread[j]):
            last = j
        nan_after[j] = last

    n_out = 0
    for i in range(prev_n - 1, n - post_n + 1):
        if nan_after[i - prev_n + 1] >= i + post_n:
            n_out += 1

    out = np.empty(n_out, np.int64)
    k = 0
    for i in range(prev_n - 1, n - post_n + 1):
        if nan_after[i - prev_n + 1] >= i + post_n:
            out[k] = i
            k += 1
    return out

@njit(cache=True, boundscheck=False)
def fill_windows_chunk(spread, valid_idx, start, stop, prev_n, post_n):
    n_out = stop - start
    prev_out = np.empty((n_out, prev_n), np.float32)
    post_out = np.empty((n_out, post_n), np.float32)
    for k in range(n_out):
        i = valid_idx[start + k]
        for j in range(prev_n):
            prev_out[k, j] = np.float32(spread[i - prev_n + 1 + j])
        for j in range(post_n):
            post_out[k, j] = np.float32(spread[i + j])
    return prev_out, post_out

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

def _log(asset, msg):
    print(f"[{asset}] {msg}", flush=True)

def run_one(asset, local=False):
    fs = gcs()

    _log(asset, "glob source partitions…")
    files = fs.glob(f"{SOURCE_BUCKET}/{SOURCE_PREFIX}/trade_date=*/asset={asset}/*.parquet")
    if not files:
        _log(asset, "no orderbook data")
        return
    _log(asset, f"{len(files)} partition file(s)")

    t0 = time.time()
    # ParquetFile bypasses pq.read_table's default partitioning="hive" — the
    # `trade_date=…/asset=…` path would otherwise be auto-discovered, adding a
    # dictionary<string> trade_date column that collides with the date32 column
    # already in the file.
    tables = []
    for i, f in enumerate(files, 1):
        t = _normalize_trade_date(pq.ParquetFile(f, filesystem=fs).read())
        tables.append(t)
        _log(asset, f"  read {i}/{len(files)}  ({t.num_rows:,} rows)")
    total_rows = sum(t.num_rows for t in tables)
    _log(asset, f"concat + sort {total_rows:,} rows…")
    tbl = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    del tables
    tbl = tbl.sort_by([("nanoseconds_start", "ascending"), ("seq_nbr", "ascending")])
    t_dl = time.time() - t0
    _log(asset, f"dl + sort {t_dl:.1f}s")

    t1 = time.time()
    _log(asset, "scanning for valid windows…")
    spread = tbl.column("spread").to_numpy(zero_copy_only=False)
    if spread.dtype != np.float64:
        spread = spread.astype(np.float64)
    valid_idx = find_valid_indices(spread, PREV_N, POST_N)
    n_valid = valid_idx.size
    if n_valid == 0:
        _log(asset, f"dl {t_dl:.1f}s, no valid windows")
        return
    _log(asset, f"  {n_valid:,} windows kept of {tbl.num_rows:,} input rows")
    sliced = tbl.take(pa.array(valid_idx))
    del tbl
    if "asset" in sliced.column_names:
        sliced = sliced.drop(["asset"])
    t_proc = time.time() - t1
    _log(asset, f"valid-scan + filter {t_proc:.1f}s")

    # Set up the output sink (local tmp file or GCS handle)
    if local:
        dest_path = LOCAL_OUTPUT_DIR / f"asset={asset}" / "data.parquet"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(".parquet.tmp")
        sink = str(tmp_path)
        dest_uri = str(dest_path)
    else:
        remote = f"{GCS_BUCKET}/{GCS_PREFIX}/asset={asset}/data.parquet"
        sink = fs.open(remote, "wb")
        dest_uri = f"gs://{remote}"

    approx_mb_total = n_valid * (PREV_N + POST_N) * 4 / 1e6   # float32
    n_chunks = (n_valid + CHUNK_SIZE - 1) // CHUNK_SIZE
    _log(asset, f"writing {n_valid:,} rows in {n_chunks} chunk(s) of {CHUNK_SIZE:,} (~{approx_mb_total:.0f} MB uncompressed total) → {dest_uri}")

    t2 = time.time()
    writer = None
    try:
        for chunk_start in range(0, n_valid, CHUNK_SIZE):
            chunk_stop = min(chunk_start + CHUNK_SIZE, n_valid)
            n_chunk = chunk_stop - chunk_start
            prev_chunk, post_chunk = fill_windows_chunk(
                spread, valid_idx, chunk_start, chunk_stop, PREV_N, POST_N
            )
            chunk_tbl = (
                sliced.slice(chunk_start, n_chunk)
                .append_column(
                    "prev_30",
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(prev_chunk.ravel(), pa.float32()), PREV_N
                    ),
                )
                .append_column(
                    "post_50",
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(post_chunk.ravel(), pa.float32()), POST_N
                    ),
                )
            )
            if writer is None:
                writer = pq.ParquetWriter(sink, chunk_tbl.schema, compression="zstd")
            writer.write_table(chunk_tbl)
            _log(asset, f"  wrote {chunk_stop:,}/{n_valid:,} rows")
    finally:
        if writer is not None:
            writer.close()
        if not local:
            sink.close()

    if local:
        tmp_path.rename(dest_path)

    t_up = time.time() - t2
    _log(asset, f"up {t_up:.1f}s  ({approx_mb_total / max(t_up, 0.1):.0f} MB/s) → {dest_uri}")
    _log(asset, f"DONE  dl {t_dl:.1f}s  proc {t_proc:.1f}s  up {t_up:.1f}s")

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
    local = "--local" in argv
    argv = [a for a in argv if a != "--local"]

    if len(argv) < 2:
        print(__doc__)
        return
    cmd = argv[1]
    if cmd == "list":
        for a in list_assets():
            print(a)
    elif cmd == "all":
        assets = list_assets()
        dest = "local disk" if local else "GCS"
        print(f"[all] {len(assets)} assets ({dest}): {', '.join(assets)}", flush=True)
        for n, a in enumerate(assets, 1):
            print(f"\n[all] === {n}/{len(assets)}: {a} ===", flush=True)
            try:
                run_one(a, local=local)
            except Exception as e:
                print(f"FAILED {a}: {e!r}", flush=True)
    elif cmd == "external":
        name = create_external_table()
        print(f"defined external table {name}")
    elif cmd == "run":
        if len(argv) < 3:
            print(f"usage: {argv[0]} run <asset> [--local]")
            return
        run_one(argv[2], local=local)
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
