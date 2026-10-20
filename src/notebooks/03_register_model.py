# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Register Timestral in Unity Catalog (`@champion`)
# MAGIC
# MAGIC Timestral becomes a governed **Unity Catalog model**, `<catalog>.<schema>.timestral`, with versions, lineage,
# MAGIC permissions and a `@champion` alias. The forecasting notebooks load `models:/<catalog>.<schema>.timestral@champion`;
# MAGIC switching checkpoints (e.g. `Timestral-28M` → `Timestral-209M`) means registering a new version and moving the
# MAGIC alias, with no code change.
# MAGIC
# MAGIC Notebook 01 used the checkpoint as plain files. This notebook packages the same files as an MLflow model:
# MAGIC
# MAGIC * [`timestral_model.py`](timestral_model.py): a small `mlflow.pyfunc.PythonModel` (models-from-code) that loads the
# MAGIC   checkpoint and forecasts a long DataFrame;
# MAGIC * `artifacts`: the checkpoint downloaded by notebook 01;
# MAGIC * `code_paths`: the pinned timestral source, so `import timestral` works wherever the model is loaded (Timestral
# MAGIC   is pure Python and not on PyPI: no build step needed).
# MAGIC
# MAGIC Then register it in Unity Catalog and point `@champion` at the new version.
# MAGIC
# MAGIC The signature is given explicitly, so MLflow does not have to run the model: this notebook needs neither a GPU nor
# MAGIC PyTorch. Runs again only if the checkpoint revision or the timestral commit changed.
# MAGIC
# MAGIC | checkpoint | weights (fp32) |
# MAGIC |---|---|
# MAGIC | `mistralai/Timestral-5M` | 21 MB |
# MAGIC | `mistralai/Timestral-28M` | 111 MB |
# MAGIC | `mistralai/Timestral-123M` | 490 MB |
# MAGIC | `mistralai/Timestral-209M` | 837 MB |
# MAGIC
# MAGIC The weights are under a [Modified MIT license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE) that
# MAGIC grants no rights to companies (or their employees) with a global consolidated monthly revenue above $20 million,
# MAGIC unless they get a commercial license from Mistral AI. The model description in Unity Catalog says so too.

# COMMAND ----------

import json
import os

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

import common

cfg = common.get_config(dbutils, spark)
revision = open(f"{cfg.model_dir}/.revision").read()  # written by notebook 01
print(f"checkpoint {cfg.model_id}@{revision} in {cfg.model_dir}\ntimestral source in {cfg.timestral_src}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and register
# MAGIC The model input is a long DataFrame (one row per series and day, `y` = NaN on the days to forecast, known-future
# MAGIC covariates as extra columns), so the registered model also works with `spark_udf` or Model Serving. The notebooks use
# MAGIC its native `TimestralForecaster` for batched calls (`unwrap_python_model().forecaster`).

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(cfg.experiment_name)
client = MlflowClient()
tags = {"hf_model_id": cfg.model_id, "hf_revision": revision, "timestral_commit": common.TIMESTRAL_COMMIT}

try:
    champion = client.get_model_version_by_alias(cfg.model_name, "champion")
except Exception:
    champion = None

if champion is not None and all(champion.tags.get(k) == v for k, v in tags.items()):
    version = champion.version
    print(f"{cfg.model_name}@champion is already version {version} ({cfg.model_id}@{revision[:7]})")
else:
    days = pd.date_range("2015-01-01", periods=127)
    input_example = pd.DataFrame({"series_id": "store_1", "ds": days, "y": np.r_[np.full(120, 5000.0), [np.nan] * 7],
                                  "open": (days.dayofweek != 6).astype(float)})
    output_example = pd.DataFrame({"series_id": ["store_1"], "ds": [days[-1]], "q10": [4000.0], "q50": [5000.0],
                                   "q90": [6000.0]})
    with mlflow.start_run(run_name="register timestral"):
        mlflow.set_tags({"level": "registration", **tags})
        info = mlflow.pyfunc.log_model(
            name="timestral",
            python_model=os.path.join(os.getcwd(), "timestral_model.py"),
            artifacts={"checkpoint": cfg.model_dir},
            code_paths=[f"{cfg.timestral_src}/timestral"],
            signature=infer_signature(input_example, output_example),
            pip_requirements=["torch>=2.4", "safetensors>=0.4", "huggingface_hub>=0.23", "numpy", "pandas"],
            registered_model_name=cfg.model_name,
        )
    version = info.registered_model_version
    for k, v in tags.items():
        client.set_model_version_tag(cfg.model_name, version, k, v)
    client.set_registered_model_alias(cfg.model_name, "champion", version)
    client.update_registered_model(cfg.model_name, description=(
        "Timestral (Mistral AI) time-series foundation model, zero-shot probabilistic forecasting with past-only and "
        "known-future covariates. Input: long DataFrame series_id, ds, y (NaN = forecast), covariate columns. "
        "Code Apache 2.0 (github.com/mistralai/timestral). Weights: Modified MIT license, which grants no rights to "
        "companies (or their employees) with a global consolidated monthly revenue above $20 million without a "
        "commercial license from Mistral AI."))
    print(f"registered {cfg.model_name} version {version}, alias @champion")

dbutils.notebook.exit(json.dumps({"model": cfg.model_name, "champion": version, "revision": revision}))
