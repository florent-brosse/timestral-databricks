# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · How Timestral uses covariates (one store, step by step)
# MAGIC
# MAGIC Timestral forecasts a **target** series from its past. It can also condition on **covariates**, extra series
# MAGIC that explain the target. They are never forecast; they only feed information to the target.
# MAGIC
# MAGIC | argument | shape | known over | example here |
# MAGIC |---|---|---|---|
# MAGIC | `context` (target) | `(B, V, T)` | the past only | daily sales |
# MAGIC | `known_covariates` | `(B, C, T + H)` | the past **and the forecast range** | store open, promo, school & public holiday |
# MAGIC | `past_covariates` | `(B, C, T)` | the past only | number of customers |
# MAGIC | `categorical_known_covariates` | `(B, C, T + H)` | past and future, as category codes | holiday type `0/a/b/c` |
# MAGIC
# MAGIC `B` = series in the batch, `V` = target variates, `T` = context length, `H` = `prediction_length`.
# MAGIC
# MAGIC Rules from the model:
# MAGIC * **Known-future covariates must cover `T + H` steps**: you need their values on the days you forecast.
# MAGIC   That is the point: "the store is closed next Sunday, there is a promo next week" is known in advance.
# MAGIC * Information only flows from covariates to targets, and targets stay causal: no leak of future sales.
# MAGIC * Rows that only contain 0 and 1 are treated as indicators. `NaN` = missing value.
# MAGIC * A covariate you do **not** know in the future (customers) can still be used as a past-only covariate.

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
import pandas as pd
import common

cfg = common.with_calendar(spark, common.get_config(dbutils, spark))
common.use_timestral(cfg)  # pinned timestral source from the volume (downloaded by notebook 01)
from timestral import TimestralForecaster

STORE = 470  # the store shown in the Timestral blog post
cut = cfg.cut_dates()[-1]  # last backtest cut: forecast the last 48 days of history

s = (spark.table(cfg.table("sales_history")).where(f"store = {STORE}").orderBy("date").toPandas()
     .assign(date=lambda d: pd.to_datetime(d.date)).set_index("date"))
c = s.index.get_loc(cut)
H = cfg.horizon
y = s.sales.to_numpy(float)
actual = y[c:c + H]
print(f"store {STORE}: {len(s)} days, forecasting {s.index[c].date()} → {s.index[c + H - 1].date()} ({H} days)")

forecaster = TimestralForecaster.from_pretrained(cfg.model_dir, device=common.device())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Target only

# COMMAND ----------

context = y[None, None, :c]  # (B=1, V=1, T): sales strictly before the cut
print("context", context.shape)
fc_none = forecaster.forecast(context, prediction_length=H)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. + known-future covariates
# MAGIC The four calendar columns are passed for the **past and the 48 forecast days**: `T + H` columns.

# COMMAND ----------

known = s[common.COV_KEYS].to_numpy(float).T[None, :, : c + H]  # (1, 4, T + H)
print("known_covariates", known.shape)
fc_known = forecaster.forecast(context, prediction_length=H, known_covariates=known)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. + a past-only covariate
# MAGIC Customers are only known after the fact, so they go in `past_covariates` with `T` columns.

# COMMAND ----------

past = s[["customers"]].to_numpy(float).T[None, :, :c]  # (1, 1, T)
print("past_covariates", past.shape)
fc_past = forecaster.forecast(context, prediction_length=H, known_covariates=known, past_covariates=past)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Categorical covariate
# MAGIC The holiday type (`0` none, `a` public, `b` Easter, `c` Christmas) as numeric codes, e.g. from `pandas.factorize`.
# MAGIC For a univariate target, Timestral target-encodes each category per series (mean sales of that category in the context).

# COMMAND ----------

codes, labels = pd.factorize(s.state_holiday.astype(str))
cat_known = codes.astype(float)[None, None, : c + H]  # (1, 1, T + H)
print("categories", list(labels), "categorical_known_covariates", cat_known.shape)
fc_cat = forecaster.forecast(context, prediction_length=H, known_covariates=known[:, :3],  # drop the 0/1 holiday flag
                             categorical_known_covariates=cat_known)

# COMMAND ----------

def mae(fc):
    return float(np.abs(common.to_numpy(fc.median[0, 0]) - actual).mean())

results = pd.DataFrame([
    {"inputs": "sales only", "MAE (€/day)": mae(fc_none)},
    {"inputs": "+ 4 known-future covariates", "MAE (€/day)": mae(fc_known)},
    {"inputs": "+ known + customers (past-only)", "MAE (€/day)": mae(fc_past)},
    {"inputs": "open, promo, school holiday + holiday type (categorical)", "MAE (€/day)": mae(fc_cat)},
]).round(0)
display(results)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(14, 4.5))
ctx_days = 56
ax.plot(s.index[c - ctx_days:c + H], y[c - ctx_days:c + H], color="black", lw=1.2, label="actual sales")
fd = s.index[c:c + H]
for fc, color, label in [(fc_none, "tab:gray", "sales only"), (fc_known, "tab:orange", "+ known covariates")]:
    ax.plot(fd, common.to_numpy(fc.median[0, 0]), color=color, lw=1.8, label=f"{label} (median)")
    ax.fill_between(fd, common.to_numpy(fc.quantile(0.1)[0, 0]), common.to_numpy(fc.quantile(0.9)[0, 0]), color=color, alpha=0.2)
closed = s.index[c:c + H][s.open.to_numpy()[c:c + H] == 0]
ax.scatter(closed, np.zeros(len(closed)), marker="v", color="tab:red", zorder=5, label="store closed (known in advance)")
ax.axvline(s.index[c], color="red", ls="--", lw=0.8)
ax.set_title(f"Store {STORE}: forecast from {s.index[c].date()}, 10-90% band shaded")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
display(fig)
