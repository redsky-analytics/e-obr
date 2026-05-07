DECLARE export_sql STRING;

FOR r IN (
  SELECT DISTINCT trade_date, asset
  FROM Downsampled_Orderbook
)
DO
  SET export_sql = FORMAT("""
    EXPORT DATA OPTIONS (
      uri = 'gs://bkt-pr-usc1-use5-2664-cdrdsml-dflt/downsampled_orderbook/trade_date=%s/asset=%s/part-*.parquet',
      format = 'PARQUET',
      overwrite = true
    ) AS
    SELECT
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
    WHERE trade_date = DATE '%s'
      AND asset = '%s'
    ORDER BY nanoseconds_start, seq_nbr
  """,
  CAST(r.trade_date AS STRING),
  r.asset,
  CAST(r.trade_date AS STRING),
  r.asset
  );

  EXECUTE IMMEDIATE export_sql;
END FOR;