"""
Orderbook windowing pipeline (per-asset).

For every row i in Downsampled_Orderbook (sorted by nanoseconds_start, seq_nbr
within each asset) where the previous PREV_N spreads and the next POST_N
spreads contain no nulls, emit:
    prev_30 = spreads[i - PREV_N + 1 .. i]   (chronological, includes self)
    post_50 = spreads[i .. i + POST_N - 1]   (chronological, includes self)

Stages (run independently or end-to-end):
    download <asset>   pull the asset's slice from BigQuery → local Parquet
    process  <asset>   read local Parquet → numba kernel → features Parquet
    upload   <asset>   push features Parquet to GCS
    run      <asset>   download + process + upload + cleanup local files
    all                run for every asset, sequentially
    load               LOAD all GCS Parquets into the final BigQuery table
    list               print all distinct assets
"""

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import gcsfs
from numba import njit
from google.cloud import bigquery, bigquery_storage_v1
from google.cloud.bigquery_storage_v1 import types

# ============================================================================
# CONFIG  — edit these
# ============================================================================
PROJECT          = "your_project"
LOCATION         = "US"
ORDERBOOK_TABLE  = "your_project.your_dataset.Downsampled_Orderbook"
OUTPUT_TABLE     = "your_project.your_dataset.Orderbook_Windows"
GCS_BUCKET       = "your_bucket"
GCS_PREFIX       = "orderbook_windows"

LOCAL_INPUT_DIR  = Path("./data/orderbook")
LOCAL_OUTPUT_DIR = Path("./data/features")

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
# BigQuery clients (lazy)
# ============================================================================
_bq_client = None
_bqs_client = None

def bq():
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=PROJECT, location=LOCATION)
    return _bq_client

def bqs():
    global _bqs_client
    if _bqs_client is None:
        _bqs_client = bigquery_storage_v1.BigQueryReadClient()
    return _bqs_client

# ============================================================================
# Download:  BQ → local Parquet (sorted by nanoseconds_start, seq_nbr)
# ============================================================================
def list_assets():
    rows = bq().query(
        f"SELECT DISTINCT asset FROM `{ORDERBOOK_TABLE}` ORDER BY asset"
    ).result()
    return [r.asset for r in rows]

def download_asset(asset):
    out_path = LOCAL_INPUT_DIR / f"asset={asset}" / "data.parquet"
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sql = f"""
        SELECT asset, trade_date, nanoseconds_start, seq_nbr, spread
        FROM `{ORDERBOOK_TABLE}`
        WHERE asset = @asset
        ORDER BY nanoseconds_start, seq_nbr
    """
    job = bq().query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("asset", "STRING", asset)]
        ),
    )
    job.result()
    dest = job.destination

    session = bqs().create_read_session(
        parent=f"projects/{PROJECT}",
        read_session=types.ReadSession(
            table=f"projects/{dest.project}/datasets/{dest.dataset_id}/tables/{dest.table_id}",
            data_format=types.DataFormat.ARROW,
        ),
        max_stream_count=1,   # single stream preserves ORDER BY
    )
    if not session.streams:
        return None

    reader = bqs().read_rows(session.streams[0].name)
    writer = None
    tmp_path = out_path.with_suffix(".parquet.tmp")
    for batch in reader.rows(session).to_arrow_iterable():
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, batch.schema, compression="zstd")
        writer.write_batch(batch)
    if writer is not None:
        writer.close()
        tmp_path.rename(out_path)
    return out_path

# ============================================================================
# Process:  local Parquet → numba kernel → features Parquet
# ============================================================================
def process_asset(asset):
    in_path  = LOCAL_INPUT_DIR  / f"asset={asset}" / "data.parquet"
    out_path = LOCAL_OUTPUT_DIR / f"asset={asset}" / "data.parquet"
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tbl = pq.read_table(in_path)
    if tbl.num_rows == 0:
        return None

    spread = tbl.column("spread").to_numpy(zero_copy_only=False)
    if spread.dtype != np.float64:
        spread = spread.astype(np.float64)

    row_idx, prev_w, post_w = build_windows(spread, PREV_N, POST_N)
    if row_idx.size == 0:
        return None

    sliced = tbl.take(pa.array(row_idx))
    prev_arr = pa.FixedSizeListArray.from_arrays(
        pa.array(prev_w.ravel(), pa.float64()), PREV_N
    )
    post_arr = pa.FixedSizeListArray.from_arrays(
        pa.array(post_w.ravel(), pa.float64()), POST_N
    )
    out = sliced.append_column("prev_30", prev_arr)
    out = out.append_column("post_50", post_arr)

    tmp_path = out_path.with_suffix(".parquet.tmp")
    pq.write_table(out, tmp_path, compression="zstd")
    tmp_path.rename(out_path)
    return out_path

# ============================================================================
# Upload:  local Parquet → GCS
# ============================================================================
def upload_asset(asset):
    fs = gcsfs.GCSFileSystem(project=PROJECT)
    local  = LOCAL_OUTPUT_DIR / f"asset={asset}" / "data.parquet"
    remote = f"{GCS_BUCKET}/{GCS_PREFIX}/asset={asset}/data.parquet"
    fs.put_file(str(local), remote)
    return f"gs://{remote}"

# ============================================================================
# Final load:  GCS → BigQuery (one big partitioned table)
# ============================================================================
def load_to_bq():
    job = bq().load_table_from_uri(
        f"gs://{GCS_BUCKET}/{GCS_PREFIX}/asset=*/data.parquet",
        OUTPUT_TABLE,
        job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition="WRITE_TRUNCATE",
            hive_partitioning=bigquery.HivePartitioningOptions.from_api_repr({
                "mode": "AUTO",
                "sourceUriPrefix": f"gs://{GCS_BUCKET}/{GCS_PREFIX}/",
            }),
        ),
    )
    job.result()
    return job.output_rows

# ============================================================================
# Driver
# ============================================================================
def run_one(asset, cleanup=True):
    t0 = time.time()
    p_in = download_asset(asset)
    if p_in is None:
        print(f"[{asset}] no orderbook data")
        return
    t_dl = time.time() - t0

    t1 = time.time()
    p_out = process_asset(asset)
    t_proc = time.time() - t1
    if p_out is None:
        print(f"[{asset}] dl {t_dl:.1f}s, no valid windows")
        if cleanup:
            p_in.unlink(missing_ok=True)
        return

    t2 = time.time()
    gs_uri = upload_asset(asset)
    t_up = time.time() - t2

    if cleanup:
        p_in.unlink(missing_ok=True)
        p_out.unlink(missing_ok=True)

    print(f"[{asset}] dl {t_dl:.1f}s  proc {t_proc:.1f}s  up {t_up:.1f}s  → {gs_uri}")

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
    elif cmd == "load":
        n = load_to_bq()
        print(f"loaded {n:,} rows into {OUTPUT_TABLE}")
    elif cmd in ("download", "process", "upload", "run"):
        if len(argv) < 3:
            print(f"usage: {argv[0]} {cmd} <asset>")
            return
        asset = argv[2]
        if   cmd == "download": print(download_asset(asset))
        elif cmd == "process":  print(process_asset(asset))
        elif cmd == "upload":   print(upload_asset(asset))
        elif cmd == "run":      run_one(asset, cleanup=False)
    else:
        print(f"unknown command: {cmd}")
        print(__doc__)

if __name__ == "__main__":
    main(sys.argv)
