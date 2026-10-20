# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Compare Timestral and Prophet, publish the results
# MAGIC
# MAGIC * **Metrics** on the backtest (forecasts joined with the real sales), per model and scenario:
# MAGIC   * **WAPE** = Σ|forecast − actual| / Σ actual: the error as a share of sales, comparable across stores of any size;
# MAGIC   * **WAPE, open days**: the same on the days the store is open. Closed days (sales = 0) are easy once `open` is
# MAGIC     known, so this is the stricter number (the Kaggle competition also scored open days only);
# MAGIC   * **MAE** in € per store-day;
# MAGIC   * **80% coverage**: how often the actual value falls between `q10` and `q90` (should be close to 0.80);
# MAGIC   * **win rate** against Prophet (for Prophet: against Timestral): share of stores where the model has the lower total error.
# MAGIC * Saved to the `backtest_metrics` table and **logged to MLflow** in each model's scenario run, so the runs can be compared
# MAGIC   side by side in the experiment UI.
# MAGIC * The view `sales_with_forecasts` puts the history and the future forecasts of both models in one place
# MAGIC   (`timestral` = the main path on serverless GPU).
# MAGIC * After the optional jobs: the compute variants of Timestral, and the Many Model Forecasting models (notebook 07).

# COMMAND ----------

import json

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

import common

cfg = common.with_calendar(spark, common.get_config(dbutils, spark))
FORECASTS, HISTORY = cfg.table("forecasts"), cfg.table("sales_history")

scored = f"""
  SELECT f.model, f.scenario, f.cut_date, f.store, f.date, f.q10, f.q50, f.q90, f.mlflow_run_id, h.sales, h.open
  FROM {FORECASTS} f JOIN {HISTORY} h USING (store, date)
  WHERE f.forecast_type = 'backtest'
"""
metric_sql = """
  CAST(100 * sum(abs(q50 - sales)) / sum(sales) AS DOUBLE) AS wape_pct,
  CAST(100 * sum(abs(q50 - sales)) FILTER (WHERE open = 1) / sum(sales) FILTER (WHERE open = 1) AS DOUBLE) AS wape_open_pct,
  CAST(avg(abs(q50 - sales)) AS DOUBLE) AS mae,
  CAST(avg(CASE WHEN q10 IS NULL THEN NULL WHEN sales BETWEEN q10 AND q90 THEN 1.0 ELSE 0.0 END) AS DOUBLE) AS coverage_80,
  count(DISTINCT store) AS stores
"""
overall = spark.sql(f"SELECT model, scenario, NULL AS cut_date, {metric_sql}, first(mlflow_run_id) AS mlflow_run_id FROM ({scored}) GROUP BY ALL")
by_cut = spark.sql(f"SELECT model, scenario, cut_date, {metric_sql}, first(mlflow_run_id) AS mlflow_run_id FROM ({scored}) GROUP BY ALL")
win = spark.sql(f"""
  WITH err AS (SELECT model, scenario, store, sum(abs(q50 - sales)) AS e FROM ({scored}) GROUP BY ALL),
  -- every Timestral variant against Prophet, and Prophet against Timestral (GPU)
  pair AS (SELECT a.model, a.scenario, CASE WHEN a.e < b.e THEN 1.0 ELSE 0.0 END AS won
           FROM err a JOIN err b ON a.scenario = b.scenario AND a.store = b.store
            AND b.model = CASE WHEN a.model = 'prophet' THEN 'timestral' ELSE 'prophet' END)
  SELECT model, scenario, CAST(avg(won) AS DOUBLE) AS win_rate FROM pair GROUP BY ALL
""")
metrics = (overall.unionByName(by_cut).join(win, ["model", "scenario"], "left").orderBy("scenario", "cut_date", "model"))
metrics.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(cfg.table("backtest_metrics"))
spark.sql(f"COMMENT ON TABLE {cfg.table('backtest_metrics')} IS 'Backtest accuracy of Timestral vs Prophet; cut_date NULL = all cuts.'")

m = metrics.toPandas()
summary = m[m.cut_date.isna()].drop(columns=["cut_date"])
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log to MLflow
# MAGIC The metrics go into the scenario runs created by notebooks 04 and 05 (per-cut WAPE as a step series), and a
# MAGIC `comparison` run keeps the summary table and the charts.

# COMMAND ----------

client = MlflowClient()
for _, r in summary.iterrows():
    for k in ["wape_pct", "wape_open_pct", "mae", "coverage_80", "win_rate"]:
        if pd.notna(r[k]):  # point-forecast models (MMF) have no coverage
            client.log_metric(r.mlflow_run_id, k, float(r[k]))
    cuts = m[(m.mlflow_run_id == r.mlflow_run_id) & m.cut_date.notna()].sort_values("cut_date")
    for step, (_, c) in enumerate(cuts.iterrows()):
        client.log_metric(r.mlflow_run_id, "wape_pct_by_cut", float(c.wape_pct), step=step)

runtimes = {}
for model in summary.model.unique():
    # latest finished run of that model (a run still in progress has no runtime_s yet)
    parent = mlflow.search_runs(experiment_names=[cfg.experiment_name],
                                filter_string=f"tags.model = '{model}' and tags.level = 'model' and metrics.runtime_s > 0",
                                order_by=["start_time DESC"], max_results=1)
    runtimes[model] = float(parent["metrics.runtime_s"].iloc[0]) if len(parent) else float("nan")

# COMMAND ----------

main = summary[summary.model.isin(["timestral", "prophet"])]
pivot = main.pivot(index="scenario", columns="model", values="wape_pct").loc[list(common.SCENARIOS)][["prophet", "timestral"]]
fig, ax = plt.subplots(figsize=(8, 4))
pivot.plot.bar(ax=ax, color={"timestral": "tab:orange", "prophet": "tab:blue"}, rot=0)
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.1f%%", fontsize=8)
ax.set_ylabel("WAPE (%), lower is better")
ax.set_title(f"Backtest on {int(summary.stores.max())} stores, {cfg.n_cuts} cuts × {cfg.horizon} days")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute variants (only after the `timestral_compute_benchmark` job)
# MAGIC Same model version, same inputs on other compute: the accuracy must match `timestral`; only the runtime differs.

# COMMAND ----------

variants = summary[summary.model.str.startswith("timestral")][["model", "scenario", "wape_pct", "wape_open_pct"]]
variants = variants.assign(runtime_s=variants.model.map(runtimes))
display(variants.sort_values(["scenario", "model"]))

# COMMAND ----------

# MAGIC %md
# MAGIC Row by row: every forecast of each variant against the same forecast of `timestral` (same store, day, scenario and
# MAGIC cut date). The differences are floating-point noise between GPU and CPU kernels.

# COMMAND ----------

display(spark.sql(f"""
  WITH ref AS (SELECT store, date, scenario, forecast_type, cut_date, q50, q90 FROM {FORECASTS} WHERE model = 'timestral')
  SELECT v.model, count(*) AS rows_compared,
         round(max(abs(v.q50 - ref.q50)), 3) AS max_abs_diff_median_eur,
         round(avg(abs(v.q50 - ref.q50)), 4) AS mean_abs_diff_median_eur,
         round(max(abs(v.q90 - ref.q90)), 3) AS max_abs_diff_q90_eur,
         round(avg(ref.q50), 0) AS mean_forecast_eur
  FROM {FORECASTS} v JOIN ref USING (store, date, scenario, forecast_type, cut_date)
  WHERE v.model LIKE 'timestral_%'
  GROUP BY v.model ORDER BY v.model
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Many Model Forecasting models (only after the `timestral_vs_mmf` job)
# MAGIC Every model on the same stores, cut dates and days, most accurate first. `timestral_wins` = share of stores where
# MAGIC Timestral has a lower total error than the model.

# COMMAND ----------

vs_timestral = spark.sql(f"""
  WITH err AS (SELECT model, scenario, store, sum(abs(q50 - sales)) AS e FROM ({scored}) GROUP BY ALL)
  SELECT a.model, a.scenario, CAST(avg(CASE WHEN t.e < a.e THEN 1.0 ELSE 0.0 END) AS DOUBLE) AS timestral_wins
  FROM err a JOIN err t ON t.model = 'timestral' AND t.scenario = a.scenario AND t.store = a.store
  WHERE a.model NOT LIKE 'timestral%' GROUP BY ALL
""").toPandas()
leaderboard = (summary[~summary.model.str.startswith("timestral_")]
               .merge(vs_timestral, on=["model", "scenario"], how="left")
               [["scenario", "model", "wape_pct", "wape_open_pct", "mae", "coverage_80", "stores", "timestral_wins"]]
               .sort_values(["scenario", "wape_pct"]))
fig3 = None
if leaderboard.model.str.startswith("mmf_").any():
    display(leaderboard)
    shown = [s for s in ["full_cov", "full_nocov"] if s in set(leaderboard.scenario)]
    fig3, axes3 = plt.subplots(1, len(shown), figsize=(7 * len(shown), 5), squeeze=False)
    for ax, scenario in zip(axes3[0], shown):
        d = leaderboard[leaderboard.scenario == scenario].sort_values("wape_pct", ascending=False)
        colors = ["tab:orange" if m == "timestral" else "tab:blue" if m == "prophet" else "tab:gray" for m in d.model]
        ax.barh(d.model.str.removeprefix("mmf_"), d.wape_pct, color=colors)
        ax.bar_label(ax.containers[0], fmt="%.1f%%", fontsize=8)
        ax.set_title(f"{scenario}: WAPE (%), lower is better")
    plt.tight_layout()
    display(fig3)

# COMMAND ----------

example = spark.sql(f"""
  SELECT model, store, date, q10, q50, q90 FROM {FORECASTS}
  WHERE forecast_type = 'future' AND store IN (1, 470, 262, 1000)
""").toPandas()
hist = spark.sql(f"""SELECT store, date, sales FROM {HISTORY}
                     WHERE store IN (1, 470, 262, 1000) AND date > date_sub('{cfg.history_end}', 56)""").toPandas()
fig2, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=True)
for ax, (store, g) in zip(axes.ravel(), example.groupby("store")):
    hs = hist[hist.store == store].sort_values("date")
    ax.plot(hs.date, hs.sales, color="black", lw=1, label="history")
    for model, color in [("timestral", "tab:orange"), ("prophet", "tab:blue")]:
        f = g[g.model == model].sort_values("date")
        ax.plot(f.date, f.q50, color=color, lw=1.4, label=model)
        ax.fill_between(f.date, f.q10, f.q90, color=color, alpha=0.15)
    ax.set_title(f"store {store}: future forecast {cfg.future_start} → {cfg.future_end}")
axes[0, 0].legend(fontsize=8)
plt.tight_layout()
display(fig2)

# COMMAND ----------

mlflow.set_experiment(cfg.experiment_name)
with mlflow.start_run(run_name="comparison") as run:
    mlflow.set_tags({"level": "comparison"})
    mlflow.log_params({"horizon": cfg.horizon, "cut_dates": ",".join(str(c.date()) for c in cfg.cut_dates()),
                       "checkpoint": cfg.model_id})
    for _, r in summary.iterrows():
        mlflow.log_metric(f"{r.model}_{r.scenario}_wape_pct", float(r.wape_pct))
    for model, secs in runtimes.items():
        mlflow.log_metric(f"{model}_runtime_s", secs)
    mlflow.log_table(summary, "backtest_summary.json")
    mlflow.log_figure(fig, "wape_by_scenario.png")
    mlflow.log_figure(fig2, "future_forecast_examples.png")
    if fig3 is not None:
        mlflow.log_figure(fig3, "wape_all_models.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results next to the data
# MAGIC `sales_with_forecasts`: one row per store and day: the history with its covariates, then the 48 future days with
# MAGIC the forecasts of both models (`full_cov` scenario). Ready for a dashboard, Genie or a downstream job.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {cfg.table('sales_with_forecasts')}
COMMENT 'Rossmann history + 48-day future forecasts of Timestral and Prophet (scenario full_cov).'
AS
WITH fut AS (
  SELECT store, date,
         max_by(q10, created_at) FILTER (WHERE model = 'timestral') AS timestral_q10,
         max_by(q50, created_at) FILTER (WHERE model = 'timestral') AS timestral_q50,
         max_by(q90, created_at) FILTER (WHERE model = 'timestral') AS timestral_q90,
         max_by(q10, created_at) FILTER (WHERE model = 'prophet') AS prophet_q10,
         max_by(q50, created_at) FILTER (WHERE model = 'prophet') AS prophet_q50,
         max_by(q90, created_at) FILTER (WHERE model = 'prophet') AS prophet_q90
  FROM {FORECASTS} WHERE forecast_type = 'future' AND scenario = 'full_cov' GROUP BY store, date
)
SELECT store, date, 'history' AS period, sales, open, promo, school_holiday, public_holiday,
       CAST(NULL AS DOUBLE) AS timestral_q10, CAST(NULL AS DOUBLE) AS timestral_q50, CAST(NULL AS DOUBLE) AS timestral_q90,
       CAST(NULL AS DOUBLE) AS prophet_q10, CAST(NULL AS DOUBLE) AS prophet_q50, CAST(NULL AS DOUBLE) AS prophet_q90
FROM {HISTORY}
UNION ALL
SELECT c.store, c.date, 'forecast', NULL, c.open, c.promo, c.school_holiday, c.public_holiday,
       timestral_q10, timestral_q50, timestral_q90, prophet_q10, prophet_q50, prophet_q90
FROM {cfg.table('future_covariates')} c LEFT JOIN fut USING (store, date)
""")
display(spark.sql(f"SELECT * FROM {cfg.table('sales_with_forecasts')} WHERE store = 470 AND date >= '2015-07-25' ORDER BY date"))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"summary": summary.drop(columns=["mlflow_run_id"]).to_dict("records"),
                                  "runtime_s": runtimes, "comparison_run_id": run.info.run_id}))
