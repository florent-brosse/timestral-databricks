# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Quickstart: Timestral in a few lines
# MAGIC
# MAGIC The simplest way to use Timestral, with nothing but the library:
# MAGIC
# MAGIC 1. download the checkpoint into the `models` volume (once, at a pinned revision);
# MAGIC 2. `TimestralForecaster.from_pretrained(<folder>)`;
# MAGIC 3. `forecast(history, prediction_length=48)` → 21 quantiles per day.
# MAGIC
# MAGIC No training, no MLflow, no GPU needed (Spark only reads the store's sales): one store forecasts in seconds on CPU.
# MAGIC
# MAGIC **License:** the weights' [Modified MIT license](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE) grants no rights to companies (or their employees) with a
# MAGIC global consolidated monthly revenue above $20 million, unless they get a commercial license from Mistral AI.
# MAGIC Once this works,
# MAGIC notebook 03 turns the same model into a governed Unity Catalog model for the production pipeline.

# COMMAND ----------

# PyTorch comes with the ML runtimes and the serverless GPU (AI) environment; serverless CPU environments do not
# have it. Install the CPU build only when it is missing, so a GPU runtime keeps its CUDA build.
import importlib.util
import subprocess
import sys

missing = [p for p, mod in [("torch>=2.4", "torch"), ("huggingface_hub>=0.23", "huggingface_hub"), ("safetensors>=0.4", "safetensors")]
           if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing,
                           "--extra-index-url", "https://download.pytorch.org/whl/cpu"])
    dbutils.library.restartPython()

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import HfApi, snapshot_download

import common

cfg = common.with_calendar(spark, common.get_config(dbutils, spark))

# 1. Weights: Hugging Face Hub -> UC volume (plain files; skipped if already there)
revision = HfApi().model_info(cfg.model_id).sha
snapshot_download(cfg.model_id, revision=revision, local_dir=cfg.model_dir,
                  allow_patterns=["config.json", "model.safetensors", "LICENSE", "README.md"])
# Timestral is not on PyPI: its pinned source goes to the same volume and onto sys.path (see common.use_timestral)
common.use_timestral(cfg)
with open(f"{cfg.model_dir}/.revision", "w") as f:  # read by notebook 03 to tag the registered version
    f.write(revision)
print(f"{cfg.model_id}@{revision[:7]} in {cfg.model_dir}")

# COMMAND ----------

from timestral import TimestralForecaster

# 2. Load the model (on the GPU when there is one)
forecaster = TimestralForecaster.from_pretrained(cfg.model_dir, device=common.device())

# 3. Forecast the next 48 days of one store from its sales history
STORE = 470
sales = spark.table(cfg.table("sales_history")).where(f"store = {STORE}").orderBy("date").toPandas()
fc = forecaster.forecast(sales.sales.to_numpy(float), prediction_length=cfg.horizon)

print("quantiles:", tuple(fc.quantiles.shape), "= (series, variates, days, quantile levels)")
print("levels:", fc.quantile_levels)

# COMMAND ----------

days = np.arange(cfg.horizon)
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(np.arange(-90, 0), sales.sales.to_numpy()[-90:], color="black", lw=1, label="last 90 days of sales")
ax.plot(days, common.to_numpy(fc.median[0, 0]), color="tab:orange", lw=1.6, label="forecast (median)")
ax.fill_between(days, common.to_numpy(fc.quantile(0.1)[0, 0]), common.to_numpy(fc.quantile(0.9)[0, 0]),
                color="tab:orange", alpha=0.2, label="10-90% band")
ax.axvline(0, color="red", ls="--", lw=0.8)
ax.set_title(f"Store {STORE}: next {cfg.horizon} days from sales history only")
ax.legend(loc="upper left", fontsize=8)
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC The forecast follows the weekly pattern but cannot know which days the store is closed or runs a promotion.
# MAGIC That is what **known-future covariates** are for: notebook 02.
