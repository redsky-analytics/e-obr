EXPORT DATA OPTIONS (
  uri = 'gs://your-bucket/downsampled_orderbook/*/*.parquet',
  format = 'PARQUET',
  overwrite = true
) AS

SELECT *
FROM (
  SELECT
    CONCAT(
      'trade_date=',
      CAST(trade_date AS STRING),
      '/asset=',
      asset
    ) AS _PARTITION_PATH,

    asset,
    trade_date,
    formatted_time,
    day,
    hour,
    minute,
    second,
    nanoseconds_start,
    seq_nbr,
    spread

  FROM Downsampled_Orderbook
)
ORDER BY
  trade_date,
  asset,
  nanoseconds_start,
  seq_nbr;
