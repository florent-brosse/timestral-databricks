"""Timestral as an MLflow model (models-from-code), registered in Unity Catalog by notebook 03.

The logged model packages:
* `artifacts["checkpoint"]`: the Timestral checkpoint (config.json + model.safetensors);
* `code_paths`: the pinned timestral source, so `import timestral` works wherever the model is loaded.

Two ways to use it:
* `mlflow.pyfunc.load_model("models:/<catalog>.<schema>.timestral@champion").unwrap_python_model().forecaster`
  gives the native `TimestralForecaster` for batched numpy calls (what the notebooks do);
* `predict(df)` takes a long DataFrame, so the model also works with `spark_udf` or Model Serving:
  one row per series and day, `y` = NaN on the days to forecast, optional known-future covariate columns.
"""
import mlflow
import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)


class TimestralModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from timestral import TimestralForecaster

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.forecaster = TimestralForecaster.from_pretrained(context.artifacts["checkpoint"], device=device)

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        """Columns: series_id, ds, y (NaN = forecast this day), then any known-future covariates.

        The horizon of each series is its trailing run of NaN `y`; covariates must be filled over it.
        Returns series_id, ds, q10, q50, q90 for the forecast days.
        """
        covs = [c for c in model_input.columns if c not in ("series_id", "ds", "y")]
        out = []
        for sid, g in model_input.sort_values(["series_id", "ds"]).groupby("series_id", sort=False):
            y = g["y"].to_numpy(float)
            observed = np.flatnonzero(~np.isnan(y))
            h = len(y) - (observed[-1] + 1 if len(observed) else 0)
            if h == 0:
                continue
            kwargs = {"known_covariates": g[covs].to_numpy(float).T[None]} if covs else {}
            fc = self.forecaster.forecast(y[None, None, : len(y) - h], prediction_length=h, **kwargs)
            q = {f"q{int(lv * 100)}": np.asarray(fc.quantile(lv)[0, 0].float().cpu(), dtype=float) for lv in QUANTILES}
            out.append(pd.DataFrame({"series_id": sid, "ds": g["ds"].to_numpy()[-h:], **q}))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["series_id", "ds", "q10", "q50", "q90"])


mlflow.models.set_model(TimestralModel())
