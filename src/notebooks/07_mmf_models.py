# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · The Many Model Forecasting (MMF) models on the same backtest
# MAGIC
# MAGIC [Many Model Forecasting](https://github.com/databricks-industry-solutions/many-model-forecasting) (MMF) is the
# MAGIC Databricks solution accelerator for forecasting at scale: local, global and foundation models behind one
# MAGIC `run_forecast` call, with backtesting, MLflow tracking and Unity Catalog registration built in.
# MAGIC
# MAGIC This notebook runs MMF models on the same stores and the **same backtest** as notebooks 04-05, and saves their
# MAGIC forecasts in the `forecasts` table next to Timestral and Prophet, so notebook 06 scores every model the same way:
# MAGIC
# MAGIC | | notebooks 04-05 | MMF `run_forecast` |
# MAGIC |---|---|---|
# MAGIC | cut dates | 2015-04-19, 2015-05-17, 2015-06-14 | `backtest_length=104`, `stride=28`: the same 3 windows |
# MAGIC | horizon | 48 days | `prediction_length=48` |
# MAGIC | covariates | open, promo, school_holiday, public_holiday | `dynamic_future_numerical`, for the models that take them |
# MAGIC | scenarios | `full_cov`, `full_nocov` | models with covariate support run both, the others `full_nocov` |
# MAGIC
# MAGIC The `mmf_models` parameter lists the models to run, from one family, as MMF recommends: foundation models on a GPU
# MAGIC node, local models (one fit per store, distributed by Spark) on a CPU cluster, global models on a CPU node.
# MAGIC Foundation models use MMF's driver-side prediction (`serverless=True`): the stores go through one GPU in batches,
# MAGIC as Timestral does in notebook 04.
# MAGIC
# MAGIC How MMF backtests: local models are refitted at every cut, global models are trained once on the data before the
# MAGIC first cut, foundation models need no training. MMF returns a point forecast (the median for the foundation models),
# MAGIC so these models have no quantiles and no coverage. Forecasts are clipped at 0, like Prophet's.

# COMMAND ----------

# MMF at a pinned commit, with the dependencies of the model family (local, global or foundation)
import common

MMF_PIP = (f"mmf_sa[{common.mmf_family(common.mmf_models(dbutils))}] @ "
           f"git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@{common.MMF_COMMIT}")

# COMMAND ----------

# MAGIC %pip install -q "{MMF_PIP}"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import uuid

import mlflow
import pandas as pd
from mmf_sa import run_forecast
from pyspark.sql import functions as F

import common

cfg = common.with_calendar(spark, common.get_config(dbutils, spark))
MODELS = common.mmf_models(dbutils)
FAMILY = common.mmf_family(MODELS)
# MMF's own backtest output (appended by run_forecast), one table per family: the job runs the families in parallel.
# Rows of earlier runs of these models are deleted, so the table does not grow with every run.
EVALUATION = f"{cfg.catalog}.{cfg.schema}.mmf_evaluation_{FAMILY}"
if spark.catalog.tableExists(EVALUATION):
    spark.sql(f"DELETE FROM {EVALUATION} WHERE model IN ({', '.join(repr(m) for m in MODELS)})")


def takes_covariates(model: str) -> bool:
    """MMF models with known-future covariate support (see mmf_sa/models/README.md)."""
    return (model in {"StatsForecastAutoArima", "StatsForecastAutoMfles", "MLForecastLGBM", "MLForecastAutoLGBM",
                      "Chronos2", "Chronos2Small", "Chronos2Synth", "TimesFM_2_5_200m"}
            or (model.startswith("NeuralForecast") and model != "NeuralForecastAutoPatchTST"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## MMF input
# MAGIC One row per store and day on the full calendar, with MMF's column names (`unique_id`, `ds`, `y`). MMF's `resample`
# MAGIC option fills missing days with 0, but it cannot be combined with covariates, so the fill happens here: the 180
# MAGIC stores without data for 2014-07 → 2014-12 get sales 0 and `open` = 0 on those days.

# COMMAND ----------

history = spark.table(cfg.table("sales_history"))
stores = history.select("store").distinct()
if cfg.max_stores:
    stores = stores.where(f"store <= {cfg.max_stores}")
calendar = spark.sql(f"SELECT explode(sequence(DATE'{cfg.history_start}', DATE'{cfg.history_end}')) AS date")
mmf_input = (stores.crossJoin(calendar).join(history, ["store", "date"], "left")
             .select(F.col("store").cast("string").alias("unique_id"), F.col("date").cast("timestamp").alias("ds"),
                     F.coalesce("sales", F.lit(0.0)).alias("y"),
                     *[F.coalesce(c, F.lit(0.0)).alias(c) for c in common.COV_KEYS])
             .cache())
n_stores = stores.count()
if FAMILY == "local":
    # MMF's settings for local models: Arrow on, no adaptive execution and enough partitions for one store per task
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    if int(spark.conf.get("spark.sql.shuffle.partitions")) < n_stores:
        spark.conf.set("spark.sql.shuffle.partitions", str(n_stores))

# MMF's backtest windows start at end - backtest_length + 1, then every `stride` days: the cut dates of notebooks 04-05
first_cut = min(cfg.cut_dates())
backtest_length = (pd.Timestamp(cfg.history_end) - first_cut).days + 1
print(f"{n_stores} stores, backtest_length={backtest_length}, stride={cfg.cut_step_days}, cuts={cfg.cut_dates()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run MMF, then save the forecasts next to Timestral and Prophet
# MAGIC `run_forecast` tracks each model in its own MLflow experiment (`<experiment>_mmf`), registers the global and
# MAGIC foundation models in Unity Catalog (`<catalog>.<schema>.<model>_rossmann_<scenario>`) and appends its backtest to
# MAGIC `mmf_evaluation_<family>` (one row per store and window, forecast and actuals as arrays). Each window becomes
# MAGIC 48 rows of the `forecasts` table, under the model name `mmf_<model>`, with an MLflow run in the main experiment
# MAGIC like Timestral and Prophet. MMF logs the errors of a model or a store and carries on, so every scenario is checked
# MAGIC first (every store, cut date and day exactly once): an incomplete model keeps its previous forecasts and the task
# MAGIC fails at the end.

# COMMAND ----------

def run_mmf(model: str, scenario: str) -> tuple:
    """Backtest one MMF model, with or without the covariates; returns its forecasts and the runtime."""
    run_id = str(uuid.uuid4())
    use_cov = scenario == "full_cov"
    t0 = time.perf_counter()
    run_forecast(
        spark=spark,
        train_data=mmf_input if use_cov else mmf_input.drop(*common.COV_KEYS),
        evaluation_output=EVALUATION,
        model_output=f"{cfg.catalog}.{cfg.schema}",
        group_id="unique_id",
        date_col="ds",
        target="y",
        freq="D",
        prediction_length=cfg.horizon,
        backtest_length=backtest_length,
        stride=cfg.cut_step_days,
        metric="mae",
        dynamic_future_numerical=common.COV_KEYS if use_cov else None,
        active_models=[model],
        accelerator="gpu" if common.device() == "cuda" else "cpu",
        serverless=FAMILY == "foundation",  # driver-side prediction on one GPU, like Timestral in notebook 04
        data_quality_check=False,
        experiment_path=f"{cfg.experiment_name}_mmf",
        use_case_name=f"rossmann_{scenario}",  # registered as <model>_rossmann_<scenario>
        run_id=run_id,
    )
    runtime = time.perf_counter() - t0
    forecasts = (
        spark.table(EVALUATION).where(F.col("run_id") == run_id)
        .select(F.col("unique_id").cast("int").alias("store"),
                F.to_date("backtest_window_start_date").alias("cut_date"),
                F.posexplode("forecast").alias("pos", "q50"))
        .select(F.lit(scenario).alias("scenario"), F.lit("backtest").alias("forecast_type"), "cut_date", "store",
                F.date_add("cut_date", F.col("pos")).alias("date"), (F.col("pos") + 1).alias("horizon_day"),
                F.lit(None).cast("double").alias("q10"), F.greatest("q50", F.lit(0.0)).alias("q50"),
                F.lit(None).cast("double").alias("q90"), F.lit(None).cast("array<double>").alias("quantiles"))
    )
    return forecasts, runtime


expected = n_stores * cfg.n_cuts * cfg.horizon
incomplete = []
for model in MODELS:
    label = f"mmf_{model}"
    scenarios = ["full_cov", "full_nocov"] if takes_covariates(model) else ["full_nocov"]
    results = {s: run_mmf(model, s) for s in scenarios}

    # MMF logs a failed model or store and carries on: check every scenario before replacing the saved forecasts
    complete = True
    for scenario, (forecasts, runtime) in results.items():
        c = forecasts.agg(F.count("*").alias("rows"), F.countDistinct("store").alias("stores"),
                          F.countDistinct("store", "cut_date", "date").alias("keys"),
                          F.sort_array(F.collect_set("cut_date")).alias("cuts")).first()
        cuts = [pd.Timestamp(d) for d in c.cuts]
        print(f"{label} · {scenario}: {c.rows:,}/{expected:,} rows, {c.stores}/{n_stores} stores, "
              f"cuts {[str(d.date()) for d in cuts]}, {runtime:.0f}s")
        if not (c.rows == c.keys == expected and cuts == cfg.cut_dates()):
            complete = False
    if not complete:
        incomplete.append(label)
        print(f"  {label}: incomplete MMF output (its errors are in the log above), previous forecasts kept")
        continue

    mlflow.set_experiment(cfg.experiment_name)  # run_forecast switched to the MMF experiment
    params = {"model": label, "model_version": f"mmf_sa@{common.MMF_COMMIT[:7]} {model}", "horizon": cfg.horizon,
              "n_stores": n_stores, "cut_dates": ",".join(str(c.date()) for c in cfg.cut_dates()),
              "execution": "MMF run_forecast", "backtest_length": backtest_length, "stride": cfg.cut_step_days}
    with mlflow.start_run(run_name=label) as parent:
        mlflow.log_params(params)
        mlflow.set_tags({"model": label, "level": "model"})
        parts = []
        for scenario, (forecasts, runtime) in results.items():
            with mlflow.start_run(run_name=f"{label} · {scenario}", nested=True) as child:
                mlflow.log_params({**params, "scenario": scenario, "use_covariates": scenario == "full_cov",
                                   "history": "full"})
                mlflow.set_tags({"model": label, "scenario": scenario, "level": "scenario"})
                mlflow.log_metric("runtime_s", runtime)
            parts.append(forecasts.withColumn("mlflow_run_id", F.lit(child.info.run_id)))

        out = parts[0]
        for part in parts[1:]:
            out = out.unionByName(part)
        out = (out.withColumn("model", F.lit(label)).withColumn("model_version", F.lit(params["model_version"]))
               .withColumn("created_at", F.current_timestamp()).select(*common.FORECAST_COLUMNS))
        out.write.mode("overwrite").option("replaceWhere", f"model = '{label}'").saveAsTable(cfg.table("forecasts"))
        mlflow.log_metrics({"runtime_s": sum(r for _, r in results.values()), "forecast_rows": expected * len(results)})

assert not incomplete, f"MMF returned incomplete forecasts for {incomplete}: see the errors in the log above"
