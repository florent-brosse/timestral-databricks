"""Shared configuration and forecasting functions for the Timestral-on-Databricks notebooks.

* Notebooks 01-02 use Timestral directly (`use_timestral` + `from_pretrained`). From notebook 04 on, Timestral is
  the Unity Catalog model `<catalog>.<schema>.timestral@champion` (registered by notebook 03): `resolve_champion`
  downloads that version once to the models volume, and every task or Spark worker loads the local copy with
  `mlflow.pyfunc.load_model`.
* `timestral_forecast_group` forecasts a batch of stores in one process (notebook 04), or one group per Spark
  task (notebook 04b). `prophet_forecast_group` fits one Prophet model per store, one store per Spark task.
  `run_grouped` registers this module with cloudpickle, so the functions are shipped to the workers.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

_MODELS: dict = {}  # loaded models, reused across batches

COV_KEYS = ["open", "promo", "school_holiday", "public_holiday"]

# Timestral is pure Python and not on PyPI: the source at this commit is downloaded once into the models volume and
# imported from there (no build step, works on any compute and on Spark workers).
TIMESTRAL_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"

# scenario -> (history length in days, None = everything; use the known-future covariates)
SCENARIOS = {
    "full_cov": (None, True),
    "full_nocov": (None, False),
    "short_cov": ("short", True),
}

FORECAST_SCHEMA = (
    "model string, scenario string, forecast_type string, cut_date date, store int, date date, "
    "horizon_day int, q10 double, q50 double, q90 double, quantiles array<double>"
)


@dataclass
class Config:
    catalog: str
    schema: str
    model_id: str
    experiment_name: str
    horizon: int
    n_cuts: int
    cut_step_days: int
    short_history_days: int
    max_stores: int
    # Calendar, filled from the data by `with_calendar`
    history_start: str = ""
    history_end: str = ""
    future_start: str = ""
    future_end: str = ""
    # Registered model, filled by `resolve_champion`
    model_version: str = ""
    model_path: str = ""

    @property
    def fqn(self) -> str:
        return f"`{self.catalog}`.`{self.schema}`"

    def table(self, name: str) -> str:
        return f"{self.fqn}.`{name}`"

    @property
    def raw_volume(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/raw"

    @property
    def model_name(self) -> str:
        """Unity Catalog registered model."""
        return f"{self.catalog}.{self.schema}.timestral"

    @property
    def timestral_src(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/models/timestral-src-{TIMESTRAL_COMMIT[:7]}"

    @property
    def model_dir(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/models/{self.model_id.replace('/', '__')}"

    def cut_dates(self) -> list[pd.Timestamp]:
        """Rolling-origin backtest: the last cut ends exactly on the last day of history."""
        last_cut = pd.Timestamp(self.history_end) - pd.Timedelta(days=self.horizon - 1)
        return sorted(last_cut - pd.Timedelta(days=self.cut_step_days * i) for i in range(self.n_cuts))


def use_timestral(cfg: Config) -> None:
    """Make `import timestral` work: download the pinned source into the volume once, add it to sys.path."""
    import io
    import sys
    import urllib.request
    import zipfile

    pkg = f"{cfg.timestral_src}/timestral/__init__.py"
    if not os.path.exists(pkg):
        url = f"https://github.com/mistralai/timestral/archive/{TIMESTRAL_COMMIT}.zip"
        prefix = f"timestral-{TIMESTRAL_COMMIT}/src/"
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())) as z:
            for name in z.namelist():
                if name == f"timestral-{TIMESTRAL_COMMIT}/LICENSE":
                    os.makedirs(cfg.timestral_src, exist_ok=True)
                    with open(os.path.join(cfg.timestral_src, "LICENSE"), "wb") as f:
                        f.write(z.read(name))
                elif name.startswith(prefix + "timestral/") and name.endswith(".py"):
                    dest = os.path.join(cfg.timestral_src, name[len(prefix):])
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as f:
                        f.write(z.read(name))
    if cfg.timestral_src not in sys.path:
        sys.path.insert(0, cfg.timestral_src)


def to_numpy(x) -> np.ndarray:
    """Forecast tensors (possibly on the GPU) to a float numpy array."""
    return x.detach().float().cpu().numpy().astype(float) if hasattr(x, "detach") else np.asarray(x, dtype=float)


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_champion(cfg: Config) -> Config:
    """Find the version behind `@champion` and download it once to the models volume.

    Every task and every Spark worker then loads the same local copy (`cfg.model_path`) without calling the registry.
    """
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    cfg.model_version = MlflowClient().get_model_version_by_alias(cfg.model_name, "champion").version
    cfg.model_path = f"/Volumes/{cfg.catalog}/{cfg.schema}/models/registered/timestral-v{cfg.model_version}"
    if not os.path.exists(f"{cfg.model_path}/MLmodel"):
        mlflow.artifacts.download_artifacts(f"models:/{cfg.model_name}/{cfg.model_version}", dst_path=cfg.model_path)
    return cfg


def load_forecaster(cfg: Config):
    """The native `TimestralForecaster` inside the registered model (on the GPU when there is one)."""
    import mlflow

    if cfg.model_path not in _MODELS:
        _MODELS[cfg.model_path] = mlflow.pyfunc.load_model(cfg.model_path).unwrap_python_model().forecaster
    return _MODELS[cfg.model_path]


def is_serverless(spark) -> bool:
    """Classic clusters expose their node type in the Spark conf; serverless does not."""
    try:
        return not spark.conf.get("spark.databricks.clusterUsageTags.clusterNodeType", "")
    except Exception:
        return True


def timestral_variant(spark) -> str:
    """Label of the forecasts by compute: `timestral` is the main path (serverless GPU)."""
    import torch

    gpu = torch.cuda.is_available()
    if is_serverless(spark):
        return "timestral" if gpu else "timestral_serverless_cpu"
    return "timestral_classic_gpu" if gpu else "timestral_classic_cpu"


def get_config(dbutils, spark) -> Config:
    """Read the job parameters; when run interactively, the widgets below hold the defaults."""
    defaults = {
        "catalog": "main", "schema": "timestral_forecast", "model_id": "mistralai/Timestral-209M",
        "experiment_name": "", "horizon": "48", "n_cuts": "3", "cut_step_days": "28",
        "short_history_days": "90", "max_stores": "0",
    }
    for k, v in defaults.items():
        dbutils.widgets.text(k, v)
    p = {k: dbutils.widgets.get(k) for k in defaults}
    if not p["experiment_name"]:
        p["experiment_name"] = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/timestral_rossmann"
    ints = ["horizon", "n_cuts", "cut_step_days", "short_history_days", "max_stores"]
    return Config(**{k: (int(v) if k in ints else v) for k, v in p.items()})


def with_calendar(spark, cfg: Config) -> Config:
    """Read the date range of the history and of the future covariates from the tables."""
    h = spark.table(cfg.table("sales_history")).selectExpr("min(date)", "max(date)").first()
    f = spark.table(cfg.table("future_covariates")).selectExpr("min(date)", "max(date)").first()
    cfg.history_start, cfg.history_end = str(h[0]), str(h[1])
    cfg.future_start, cfg.future_end = str(f[0]), str(f[1])
    n_future = (pd.Timestamp(cfg.future_end) - pd.Timestamp(cfg.future_start)).days + 1
    assert n_future == cfg.horizon, f"future covariates cover {n_future} days, horizon is {cfg.horizon}"
    return cfg


def load_series(spark, cfg: Config):
    """Model input: one row per store and day, history followed by the future covariates (sales = NULL)."""
    hist = spark.table(cfg.table("sales_history")).select("store", "date", "sales", *COV_KEYS)
    fut = spark.table(cfg.table("future_covariates")).selectExpr("store", "date", "CAST(NULL AS DOUBLE) AS sales", *COV_KEYS)
    df = hist.unionByName(fut)
    if cfg.max_stores:
        df = df.where(f"store <= {cfg.max_stores}")
    return df


def _dense(pdf: pd.DataFrame, start: str, end: str) -> tuple[list[int], pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Pivot long rows to `(n_stores, n_days)` arrays on a full daily calendar; missing days are NaN."""
    dates = pd.date_range(start, end, freq="D")
    stores = sorted(pdf.store.unique())
    pdf = pdf.assign(date=pd.to_datetime(pdf.date))
    cube = {c: pdf.pivot(index="store", columns="date", values=c).reindex(index=stores, columns=dates).to_numpy(float)
            for c in ["sales", *COV_KEYS]}
    return stores, dates, cube


def _rows(model, scenario, kind, cut, stores, dates, q10, q50, q90, quantiles=None) -> pd.DataFrame:
    n, h = q50.shape
    return pd.DataFrame({
        "model": model, "scenario": scenario, "forecast_type": kind, "cut_date": cut.date(),
        "store": np.repeat(np.asarray(stores, dtype=np.int32), h),
        "date": np.tile(dates.date, n), "horizon_day": np.tile(np.arange(1, h + 1, dtype=np.int32), n),
        "q10": q10.ravel(), "q50": q50.ravel(), "q90": q90.ravel(),
        "quantiles": list(quantiles.reshape(n * h, -1)) if quantiles is not None else None,
    })


def _tasks(cfg: Config):
    """(scenario, forecast_type, cut) to compute: every scenario on every backtest cut, plus the future forecast."""
    for scenario in SCENARIOS:
        for cut in cfg.cut_dates():
            yield scenario, "backtest", cut
    yield "full_cov", "future", pd.Timestamp(cfg.future_start)


# --------------------------------------------------------------------------------------------
# Timestral: one zero-shot model, loaded once, a batch of stores per call
# --------------------------------------------------------------------------------------------



def timestral_forecast_group(pdf: pd.DataFrame, cfg_dict: dict) -> pd.DataFrame:
    import torch

    cfg = Config(**cfg_dict)
    # On a CPU cluster several Spark tasks share a worker: TIMESTRAL_TORCH_THREADS (cluster env var) = cores per task.
    torch.set_num_threads(int(os.environ.get("TIMESTRAL_TORCH_THREADS", os.cpu_count() or 1)))
    model = load_forecaster(cfg)
    stores, dates, cube = _dense(pdf, cfg.history_start, cfg.future_end)
    has_future = ~np.isnan(cube["open"][:, dates.get_loc(pd.Timestamp(cfg.future_start)):]).all(axis=1)

    out = []
    for scenario, kind, cut in _tasks(cfg):
        history, use_cov = SCENARIOS[scenario]
        c = dates.get_loc(cut)
        s0 = max(0, c - cfg.short_history_days) if history == "short" else 0
        e = c + cfg.horizon
        rows = np.flatnonzero(has_future) if kind == "future" else np.arange(len(stores))
        if len(rows) == 0:
            continue
        target = cube["sales"][rows, None, s0:c]  # (B, 1, T): sales up to the day before the cut
        kwargs = {}
        if use_cov:
            # (B, 4, T + horizon): the covariates must also be known over the forecast range
            kwargs["known_covariates"] = np.stack([cube[k][rows, s0:e] for k in COV_KEYS], axis=1)
        with torch.inference_mode():
            fc = model.forecast(target, prediction_length=cfg.horizon, **kwargs)
        q = fc.quantiles[:, 0].float().cpu().numpy().astype(float)  # (B, H, 21)
        lv = list(fc.quantile_levels)
        out.append(_rows("timestral", scenario, kind, cut, [stores[i] for i in rows], dates[c:e],
                         q[..., lv.index(0.1)], q[..., lv.index(0.5)], q[..., lv.index(0.9)], q))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------------------------
# Prophet: one model fitted per store, scenario and cut
# --------------------------------------------------------------------------------------------

def prophet_forecast_group(pdf: pd.DataFrame, cfg_dict: dict) -> pd.DataFrame:
    from prophet import Prophet

    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).disabled = True
    cfg = Config(**cfg_dict)
    stores, dates, cube = _dense(pdf, cfg.history_start, cfg.future_end)
    store, i = stores[0], 0  # one store per group
    out = []
    for scenario, kind, cut in _tasks(cfg):
        history, use_cov = SCENARIOS[scenario]
        c = dates.get_loc(cut)
        s0 = max(0, c - cfg.short_history_days) if history == "short" else 0
        e = c + cfg.horizon
        if kind == "future" and np.isnan(cube["open"][i, c:e]).all():
            continue  # store not in the future covariates file
        y = cube["sales"][i, s0:c]
        keep = ~np.isnan(y)  # Prophet handles gaps by dropping the missing days
        hist = pd.DataFrame({"ds": dates[s0:c][keep], "y": y[keep]})
        fut = pd.DataFrame({"ds": dates[c:e]})
        m = Prophet(weekly_seasonality=True, yearly_seasonality=history != "short", daily_seasonality=False,
                    interval_width=0.8)
        if use_cov:
            for k in COV_KEYS:
                m.add_regressor(k)
                hist[k] = np.nan_to_num(cube[k][i, s0:c][keep], nan=0.0)
                # Prophet cannot take NaN regressors; an unknown future "open" is assumed open
                fut[k] = np.nan_to_num(cube[k][i, c:e], nan=1.0 if k == "open" else 0.0)
        m.fit(hist)
        p = m.predict(fut)
        clip = lambda a: np.clip(a.to_numpy(float), 0, None)[None]  # sales cannot be negative
        out.append(_rows("prophet", scenario, kind, cut, [store], dates[c:e],
                         clip(p.yhat_lower), clip(p.yhat), clip(p.yhat_upper)))
    return pd.concat(out, ignore_index=True)


def run_grouped(df, fn, group_col: str, n_groups: int, cfg: Config):
    """Run `fn` once per group with applyInPandas; repartition so every group gets its own task."""
    import cloudpickle
    import common

    cloudpickle.register_pickle_by_value(common)
    cfg_dict = asdict(cfg)
    return (df.repartition(n_groups, group_col)
              .groupBy(group_col)
              .applyInPandas(lambda pdf: fn(pdf.drop(columns=[group_col]), cfg_dict), FORECAST_SCHEMA))


def forecast_batches(pdf: pd.DataFrame, cfg: Config, fn, stores_per_call: int, label: str = "") -> pd.DataFrame:
    """Run `fn` on batches of `stores_per_call` stores in this process (no Spark needed)."""
    t0 = time.perf_counter()
    stores = sorted(pdf.store.unique())
    parts = []
    for b0 in range(0, len(stores), stores_per_call):
        parts.append(fn(pdf[pdf.store.isin(stores[b0:b0 + stores_per_call])], asdict(cfg)))
        print(f"  {label}: {min(b0 + stores_per_call, len(stores))}/{len(stores)} stores, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    return pd.concat(parts, ignore_index=True)


FORECAST_COLUMNS = ["model", "model_version", "scenario", "forecast_type", "cut_date", "store", "date", "horizon_day",
                    "q10", "q50", "q90", "quantiles", "mlflow_run_id", "created_at"]


def forecast_and_save(spark, cfg: Config, model: str, model_version: str, fn, params: dict, distributed: bool,
                      stores_per_call: int = 1) -> dict:
    """Forecast every store with `fn`, track it in MLflow and overwrite the model's partition of `forecasts`.

    distributed=True: Spark runs `fn` on groups of `stores_per_call` stores with applyInPandas (Prophet: one store per
    group; Timestral on a CPU cluster: a batch of stores per group, one model per Python worker).
    distributed=False: the driver runs `fn` on batches of `stores_per_call` stores (Timestral: one large model,
    loaded once, on the GPU when there is one).

    One parent MLflow run per model with a child run per scenario; `05_evaluate` logs the metrics into the
    child runs, and every forecast row keeps its child run id in `mlflow_run_id`.
    """
    import mlflow
    from pyspark.sql import functions as F

    df = load_series(spark, cfg)
    n_stores = df.select("store").distinct().count()

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=model) as parent:
        common_params = {"model": model, "model_version": model_version, "horizon": cfg.horizon, "n_stores": n_stores,
                         "cut_dates": ",".join(str(c.date()) for c in cfg.cut_dates()),
                         "future_start": cfg.future_start, "execution": "spark applyInPandas" if distributed else "driver",
                         **params}
        mlflow.log_params(common_params)
        mlflow.set_tags({"model": model, "level": "model"})
        runs = {}
        for scenario, (history, use_cov) in SCENARIOS.items():
            with mlflow.start_run(run_name=f"{model} · {scenario}", nested=True) as child:
                mlflow.log_params({**common_params, "scenario": scenario, "use_covariates": use_cov,
                                   "history": f"last {cfg.short_history_days} days" if history else "full"})
                mlflow.set_tags({"model": model, "scenario": scenario, "level": "scenario"})
                runs[scenario] = child.info.run_id

        t0 = time.perf_counter()
        if distributed:
            group = F.floor((F.col("store") - 1) / stores_per_call)
            n_groups = df.select(group.alias("g")).distinct().count()
            forecasts = run_grouped(df.withColumn("group_id", group), fn, "group_id", n_groups, cfg)
        else:
            pdf = df.toPandas()
            forecasts = spark.createDataFrame(forecast_batches(pdf, cfg, fn, stores_per_call, model), FORECAST_SCHEMA)

        run_map = F.create_map(*[F.lit(x) for kv in runs.items() for x in kv])
        out = (forecasts.withColumn("model", F.lit(model))  # the variant label: timestral, timestral_classic_gpu, ...
               .withColumn("model_version", F.lit(model_version))
               .withColumn("mlflow_run_id", run_map[F.col("scenario")])
               .withColumn("created_at", F.current_timestamp())
               .select(*FORECAST_COLUMNS))
        out.write.mode("overwrite").option("replaceWhere", f"model = '{model}'").saveAsTable(cfg.table("forecasts"))
        runtime = time.perf_counter() - t0
        n_rows = spark.table(cfg.table("forecasts")).where(f"model = '{model}'").count()
        mlflow.log_metrics({"runtime_s": runtime, "forecast_rows": n_rows})  # forecast + Delta write
    print(f"{model}: {n_rows:,} forecast rows for {n_stores} stores in {runtime:.0f}s (MLflow run {parent.info.run_id})")
    return {"parent_run_id": parent.info.run_id, "runtime_s": runtime, "rows": n_rows, "runs": runs}
