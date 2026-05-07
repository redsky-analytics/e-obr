# orderbook windowing

Two-step pipeline that turns the BigQuery `Downsampled_Orderbook` table into
per-asset Parquet files where each row carries the previous 300 and next 500
spread values as fixed-size lists.

## what's here

- `export_example.sql` — BigQuery script. Loops over `(trade_date, asset)` and
  shards the orderbook to `gs://<bucket>/downsampled_orderbook/trade_date=…/asset=…/`
  via `EXPORT DATA`.
- `pipeline.py` — reads those shards, runs a numba kernel to extract the
  prev/post windows, and writes one Parquet per asset (Hive-partitioned by
  `asset=`).

## setup

```
pip install -r requirements.txt
gcloud auth application-default login
export BQ_DATASET=myproject.mydataset
```

`SOURCE_BUCKET` is hardcoded; everything else has a default.

## running

Run the BigQuery export once (paste `export_example.sql` into the BQ console),
then:

```
python pipeline.py list                  # show all assets
python pipeline.py run 6A                # one asset → GCS
python pipeline.py run 6A --local        # one asset → ./data/features/
python pipeline.py all                   # everything
python pipeline.py external              # CREATE OR REPLACE EXTERNAL TABLE
```

Useful extras:

```
python pipeline.py inspect 6A            # dump per-partition Parquet schemas
python pipeline.py view 6A --local       # interactive row pager
python pipeline.py run 6A --whole        # don't chunk — single big row group
```

## notes

- Output windows are float32 — single-precision is fine for spreads and
  halves both memory and disk.
- The chunked write keeps peak memory ~2-3 GB regardless of asset size; a
  single-shot `--whole` run for the largest asset peaks around 30 GB.
- `trade_date` in the source is sometimes written as `date32` and sometimes as
  `dictionary<string>`. The reader normalizes it to `date32` before concat.
- `asset` is intentionally dropped from the output payload — the path
  partition column re-attaches it cleanly via the external table.
