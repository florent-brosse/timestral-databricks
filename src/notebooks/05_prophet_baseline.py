# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Prophet baseline (one model fitted per store)
# MAGIC
# MAGIC [Prophet](https://facebook.github.io/prophet/) is the usual baseline for business time series: trend + weekly and
# MAGIC yearly seasonality + extra regressors. It is fitted to each store separately, which Spark parallelises with
# MAGIC `applyInPandas` (one store per group).
# MAGIC
# MAGIC Same inputs as Timestral, so the comparison is fair:
# MAGIC
# MAGIC * same backtest cuts, horizon and scenarios; the 4 covariates are added as regressors (`add_regressor`);
# MAGIC * `short_cov` turns off the yearly seasonality (90 days cannot estimate it);
# MAGIC * days without data are dropped (Prophet handles gaps), a missing future `open` is assumed open;
# MAGIC * forecasts are clipped at 0: Prophet adds a negative "closed" effect to the baseline and can go below zero;
# MAGIC * `q10` / `q90` are Prophet's 80% interval (`interval_width=0.8`).

# COMMAND ----------

# MAGIC %pip install -q prophet==1.5.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json

import common

cfg = common.with_calendar(spark, common.get_config(dbutils, spark))

result = common.forecast_and_save(
    spark, cfg, "prophet", f"prophet {__import__('prophet').__version__}", common.prophet_forecast_group,
    distributed=True, stores_per_call=1, params={"interval_width": 0.8},
)

# COMMAND ----------

display(spark.sql(f"""
  SELECT scenario, forecast_type, cut_date, count(DISTINCT store) AS stores, count(*) AS rows
  FROM {cfg.table('forecasts')} WHERE model = 'prophet' GROUP BY ALL ORDER BY forecast_type, scenario, cut_date
"""))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({k: v for k, v in result.items() if k != "runs"}))
