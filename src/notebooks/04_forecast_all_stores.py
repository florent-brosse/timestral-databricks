# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Forecast every store from the registered model (`@champion`)
# MAGIC
# MAGIC One pretrained model, no training. `common.resolve_champion` looks up the version behind
# MAGIC `models:/<catalog>.<schema>.timestral@champion` in Unity Catalog, downloads it once to the models volume
# MAGIC (`registered/timestral-v<N>`), and the notebook loads that copy with `mlflow.pyfunc.load_model`, on the GPU when
# MAGIC there is one. The stores are forecast in batches of 64: per batch, one `forecast` call for each forecast origin
# MAGIC (3 scenarios × 3 cut dates + the future forecast = 10 calls). Spark reads the inputs and writes the results; the
# MAGIC forecasting itself is plain Python on one node, no Spark UDF.
# MAGIC
# MAGIC The main job runs it on **serverless GPU** (`model = 'timestral'`). The compute benchmark job runs the same notebook
# MAGIC on a classic GPU node (`timestral_classic_gpu`) and a classic CPU node (`timestral_classic_cpu`), and notebook 04b
# MAGIC on a CPU cluster (`timestral_cpu_cluster`).
# MAGIC
# MAGIC Why not Spark `applyInPandas` like Prophet? Prophet fits a small model per store, which spreads well across
# MAGIC Spark tasks. Timestral is one large model (837 MB): it is used best by loading it once and pushing many series
# MAGIC through it in large batches on a GPU.
# MAGIC
# MAGIC For every store we compute:
# MAGIC
# MAGIC * a **backtest**: forecasts from `n_cuts` past dates (rolling origin, 48 days each), compared later with the real sales,
# MAGIC   in 3 scenarios:
# MAGIC   * `full_cov`: all history since 2013 + the 4 known-future covariates;
# MAGIC   * `full_nocov`: all history, no covariates;
# MAGIC   * `short_cov`: only the last 90 days + covariates (a new store, or a new product);
# MAGIC * the **future** forecast (2015-08-01 → 2015-09-17) from the full history and the future covariates of `future_covariates`.
# MAGIC
# MAGIC The rows go to the `forecasts` table with the model version, and the run is tracked in MLflow.

# COMMAND ----------

import json
import os

import torch

import common

cfg = common.resolve_champion(common.with_calendar(spark, common.get_config(dbutils, spark)))
STORES_PER_CALL = 64
MODEL = common.timestral_variant(spark)
print(f"{cfg.model_name}@champion = version {cfg.model_version} -> forecasts labelled '{MODEL}'")
print("backtest cuts:", [str(c.date()) for c in cfg.cut_dates()], "| future from", cfg.future_start)

result = common.forecast_and_save(
    spark, cfg, MODEL, f"{cfg.model_name}/{cfg.model_version}", common.timestral_forecast_group,
    distributed=False, stores_per_call=STORES_PER_CALL,
    params={"checkpoint": cfg.model_id, "stores_per_call": STORES_PER_CALL,
            "device": "cuda" if torch.cuda.is_available() else "cpu", "driver_cpus": os.cpu_count(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"},
)

# COMMAND ----------

display(spark.sql(f"""
  SELECT scenario, forecast_type, cut_date, count(DISTINCT store) AS stores, count(*) AS rows
  FROM {cfg.table('forecasts')} WHERE model = '{MODEL}' GROUP BY ALL ORDER BY forecast_type, scenario, cut_date
"""))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({k: v for k, v in result.items() if k != "runs"}))
