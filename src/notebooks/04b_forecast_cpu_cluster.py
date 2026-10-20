# Databricks notebook source
# MAGIC %md
# MAGIC # 04b · Timestral on a multi-node CPU cluster (Spark `applyInPandas`)
# MAGIC
# MAGIC Same forecasts as notebook 04, without a GPU: Spark splits the stores into groups of 32, and each task loads the
# MAGIC `@champion` model (the copy that `common.resolve_champion` keeps in the models volume) and forecasts its group on
# MAGIC CPU. The benchmark job runs it on 4 workers × 32 vCPU (128 vCPU) with `spark.task.cpus = 4`: 32 tasks in parallel,
# MAGIC 4 PyTorch threads each (`TIMESTRAL_TORCH_THREADS`), so the 35 groups of the 1,115 stores run in two waves (32 + 3).
# MAGIC
# MAGIC Each task loads its own copy of the 837 MB model + PyTorch, so this pattern is meant for classic clusters with
# MAGIC large workers (here 128 GB / 8 tasks). On serverless, use the GPU path of notebook 04.
# MAGIC
# MAGIC Saved as `model = 'timestral_cpu_cluster'`: its accuracy should match the GPU run (same model, same inputs); the
# MAGIC interesting number is `runtime_s` in MLflow, next to Timestral on GPU and Prophet.

# COMMAND ----------

import json

import common

cfg = common.resolve_champion(common.with_calendar(spark, common.get_config(dbutils, spark)))
STORES_PER_TASK = 32

result = common.forecast_and_save(
    spark, cfg, "timestral_cpu_cluster", f"{cfg.model_name}/{cfg.model_version}", common.timestral_forecast_group,
    distributed=True, stores_per_call=STORES_PER_TASK,
    params={"checkpoint": cfg.model_id, "stores_per_task": STORES_PER_TASK, "device": "cpu",
            "cluster": spark.conf.get("spark.databricks.clusterUsageTags.clusterNodeType", "unknown"),
            "workers": spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers", "unknown")},
)

# COMMAND ----------

dbutils.notebook.exit(json.dumps({k: v for k, v in result.items() if k != "runs"}))
