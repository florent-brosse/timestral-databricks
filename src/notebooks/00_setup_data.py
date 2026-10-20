# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Load the Rossmann dataset into Unity Catalog
# MAGIC
# MAGIC [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales): daily sales of 1,115 German drugstores,
# MAGIC 2013-01-01 → 2015-07-31, with the store calendar (open, promo, school and public holidays).
# MAGIC The Kaggle `test.csv` gives the same calendar for the **next 48 days** (2015-08-01 → 2015-09-17) without the sales:
# MAGIC exactly what a forecasting model with *known-future covariates* needs in production.
# MAGIC
# MAGIC This notebook copies the CSV files into the `raw` volume and creates three tables:
# MAGIC
# MAGIC | table | content |
# MAGIC |---|---|
# MAGIC | `sales_history` | one row per store and day: sales, customers and the 4 covariates |
# MAGIC | `future_covariates` | the 4 covariates for the 48 days to forecast (no sales) |
# MAGIC | `stores` | store attributes (type, assortment, competition, Promo2) |
# MAGIC
# MAGIC The files come from a public Hugging Face copy of the Kaggle competition data
# MAGIC ([`AiiN-aini/rossmann-store-sales`](https://huggingface.co/datasets/AiiN-aini/rossmann-store-sales)), so no Kaggle
# MAGIC login is needed. Check the [competition rules](https://www.kaggle.com/c/rossmann-store-sales/rules) before reusing the data.

# COMMAND ----------

import urllib.request

from pyspark.sql import functions as F

import common

cfg = common.get_config(dbutils, spark)
SOURCE = "https://huggingface.co/datasets/AiiN-aini/rossmann-store-sales/resolve/main"

for name in ["train.csv", "test.csv", "store.csv"]:
    dest = f"{cfg.raw_volume}/{name}"
    urllib.request.urlretrieve(f"{SOURCE}/{name}", dest)
    print(f"{dest}: {dbutils.fs.ls(dest)[0].size / 1e6:.1f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tables
# MAGIC `StateHoliday` is `0` (none), `a` (public holiday), `b` (Easter) or `c` (Christmas); it becomes a 0/1 `public_holiday`.

# COMMAND ----------

def read_csv(name):
    return spark.read.option("header", True).option("inferSchema", True).csv(f"{cfg.raw_volume}/{name}")

def covariates(df):
    return df.select(
        F.col("Store").cast("int").alias("store"),
        F.to_date("Date").alias("date"),
        *([F.col("Sales").cast("double").alias("sales"), F.col("Customers").cast("double").alias("customers")]
          if "Sales" in df.columns else []),
        F.col("Open").cast("double").alias("open"),  # NULL in test.csv when unknown
        F.col("Promo").cast("double").alias("promo"),
        F.col("SchoolHoliday").cast("double").alias("school_holiday"),
        (F.col("StateHoliday").cast("string") != "0").cast("double").alias("public_holiday"),
        F.col("StateHoliday").cast("string").alias("state_holiday"),
    )

comments = {
    "sales_history": "Rossmann daily sales per store (Kaggle train.csv) with the 4 calendar covariates used for forecasting.",
    "future_covariates": "Known-future covariates for the 48 days after the history (Kaggle test.csv). No sales: this is what we forecast.",
    "stores": "Rossmann store attributes (Kaggle store.csv).",
}
frames = {
    "sales_history": covariates(read_csv("train.csv")),
    "future_covariates": covariates(read_csv("test.csv")),
    "stores": read_csv("store.csv").withColumnRenamed("Store", "store"),
}
for name, df in frames.items():
    df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(cfg.table(name))
    spark.sql(f"COMMENT ON TABLE {cfg.table(name)} IS '{comments[name]}'")

spark.sql(f"ALTER TABLE {cfg.table('sales_history')} ALTER COLUMN open COMMENT '1 if the store is open that day'")
spark.sql(f"ALTER TABLE {cfg.table('sales_history')} ALTER COLUMN promo COMMENT '1 if the store runs a promotion that day'")

# COMMAND ----------

summary = spark.sql(f"""
  SELECT 'sales_history' AS table, count(*) AS rows, count(DISTINCT store) AS stores, min(date) AS first_day, max(date) AS last_day
  FROM {cfg.table('sales_history')}
  UNION ALL
  SELECT 'future_covariates', count(*), count(DISTINCT store), min(date), max(date) FROM {cfg.table('future_covariates')}
""")
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC 180 stores have no rows for 2014-07-01 → 2014-12-31 (closed for refurbishment in the Kaggle data). The models get
# MAGIC those days as missing values: Timestral takes NaN as "not observed", Prophet drops the rows.

# COMMAND ----------

display(spark.sql(f"""
  SELECT n_days, count(*) AS stores FROM (SELECT store, count(*) AS n_days FROM {cfg.table('sales_history')} GROUP BY store)
  GROUP BY n_days ORDER BY n_days
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecast table
# MAGIC Created here so the Timestral and Prophet notebooks can run in parallel: it is partitioned by `model` and each
# MAGIC notebook only overwrites its own partition, so the two writes never conflict.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {cfg.table('forecasts')} (
  model STRING COMMENT 'timestral (main path), prophet, or a compute variant: timestral_classic_gpu, ...',
  model_version STRING COMMENT 'Unity Catalog model version (catalog.schema.timestral/N) or the prophet version',
  scenario STRING COMMENT 'full_cov | full_nocov | short_cov (see README)',
  forecast_type STRING COMMENT 'backtest (actuals known) or future (production forecast)',
  cut_date DATE COMMENT 'first forecast day; the model only saw sales before this date',
  store INT,
  date DATE,
  horizon_day INT COMMENT '1 = cut_date',
  q10 DOUBLE COMMENT '10% quantile',
  q50 DOUBLE COMMENT 'median: the point forecast',
  q90 DOUBLE COMMENT '90% quantile',
  quantiles ARRAY<DOUBLE> COMMENT 'Timestral only: the 21 quantile levels 0.01, 0.05, 0.1 ... 0.9, 0.95, 0.99',
  mlflow_run_id STRING COMMENT 'MLflow run that produced the forecast',
  created_at TIMESTAMP
)
PARTITIONED BY (model)
COMMENT 'Sales forecasts of every store by Timestral and Prophet, backtest and future.'
""")
