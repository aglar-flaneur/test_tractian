from pathlib import Path
import numpy as np
import yaml

from .interface import ModelParams, PredictOutput, TimeSeries, Weights

DEFAULT_PARAMS_PATH = Path("hyperparameters/model_hyperparams.yaml")


def load_model_params(path: Path = DEFAULT_PARAMS_PATH) -> ModelParams:
    if not path.exists():
        return ModelParams()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ModelParams(**data)


class AnomalyModel:
    def __init__(self, params_path: Path | None = None):
        self.weights = Weights()
        self.params = load_model_params(params_path or DEFAULT_PARAMS_PATH)

    def _featuring(self, samples: TimeSeries) -> tuple[np.ndarray, np.ndarray]:
        if not samples.data:
            return np.array([]).reshape(0, 6), np.array([])

        vel_x = [p.vel_x for p in samples.data]
        vel_y = [p.vel_y for p in samples.data]
        vel_z = [p.vel_z for p in samples.data]
        acc_x = [p.acc_x for p in samples.data]
        acc_y = [p.acc_y for p in samples.data]
        acc_z = [p.acc_z for p in samples.data]
        
        X = np.column_stack((vel_x, vel_y, vel_z, acc_x, acc_y, acc_z))
        acc = np.column_stack((acc_x, acc_y, acc_z))
        acc_norm = np.linalg.norm(acc, axis=1)

        return X, acc_norm

    def fit(self, fitting_samples: TimeSeries) -> None:
        X, acc_norm = self._featuring(fitting_samples)
        if len(X) == 0:
            return

        is_on = acc_norm > self.params.uptime_acc_threshold

        if np.sum(is_on) == 0:
            is_on = np.ones(len(X), dtype=bool)

        on_X = X[is_on]

        self.weights = Weights(
            fitted=True,
            mean=on_X.mean(axis=0).tolist(),
            std=on_X.std(axis=0).tolist()
        )

    def predict(self, samples: TimeSeries) -> PredictOutput:
        if not self.weights.fitted:
            raise RuntimeError("Model not fitted")
        if not samples.data:
            raise ValueError("Cannot predict on empty TimeSeries")

        X, acc_norm = self._featuring(samples)
        is_on = acc_norm > self.params.uptime_acc_threshold

        if np.sum(is_on) == 0:
            return PredictOutput(
                anomaly_status=False,
                timestamp=samples.data[-1].timestamp,
            )

        on_X = X[is_on]
        
        mean_arr = np.array(self.weights.mean)
        std_arr = np.array(self.weights.std)

        Z = (on_X - mean_arr) / (std_arr + 1e-9)

        anomalous_points = np.any(Z > self.params.z_threshold, axis=1)
        deviation_ratio = float(np.sum(anomalous_points) / len(on_X))

        is_anomalous = deviation_ratio >= self.params.window_anomaly_ratio
        
        ordered = sorted(samples.data, key=lambda p: p.timestamp)
        if is_anomalous:
            z_scores = np.zeros(len(X))
            z_scores[is_on] = np.max(Z, axis=1)
            max_idx = int(np.argmax(z_scores))
            ts = ordered[max_idx].timestamp
        else:
            ts = ordered[-1].timestamp

        return PredictOutput(
            anomaly_status=is_anomalous,
            timestamp=ts,
        )
