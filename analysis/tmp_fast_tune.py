import multiprocessing
import numpy as np
import warnings
from classes.data_pipeline import AnomalyPipeline
from classes.interface import PipelineParams, PredictOutput
from utils.utils import evaluate, aggregate_results
from utils.data_loading import load_all_scenarios

warnings.filterwarnings('ignore')
scenarios = load_all_scenarios()

class MahalanobisModel:
    def __init__(self):
        pass
        
    def _featuring(self, samples):
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

    def fit(self, fitting_samples):
        X, acc_norm = self._featuring(fitting_samples)
        if len(X) == 0:
            return
        is_on = acc_norm > self.params.uptime_acc_threshold
        if np.sum(is_on) == 0:
            is_on = np.ones(len(X), dtype=bool)
        
        on_X = X[is_on]
        self.mean = np.mean(on_X, axis=0)
        self.cov = np.cov(on_X, rowvar=False)
        self.inv_cov = np.linalg.pinv(self.cov)
        self.fitted = True

    def predict(self, samples):
        X, acc_norm = self._featuring(samples)
        is_on = acc_norm > self.params.uptime_acc_threshold
        if np.sum(is_on) == 0:
            return PredictOutput(anomaly_status=False, timestamp=samples.data[-1].timestamp)
            
        on_X = X[is_on]
        diff = on_X - self.mean
        dist = np.sqrt(np.sum(np.dot(diff, self.inv_cov) * diff, axis=1))
        
        anomalous_points = dist > self.params.z_threshold
        deviation_ratio = float(np.sum(anomalous_points) / len(on_X))
        
        is_anomalous = deviation_ratio >= self.params.window_anomaly_ratio
        
        ordered = sorted(samples.data, key=lambda p: p.timestamp)
        if is_anomalous:
            z_scores = np.zeros(len(X))
            z_scores[is_on] = dist
            max_idx = int(np.argmax(z_scores))
            ts = ordered[max_idx].timestamp
        else:
            ts = ordered[-1].timestamp
            
        return PredictOutput(anomaly_status=is_anomalous, timestamp=ts)


from classes.alert_engine import AlertEngine

def eval_hparams(params):
    z, r, up = params
    class MockParams:
        z_threshold = z
        window_anomaly_ratio = r
        uptime_acc_threshold = up

    metrics_list = {}
    for i, (fit_ts, predict_ts, true_incidents) in enumerate(scenarios, 1):
        model = MahalanobisModel()
        model.params = MockParams()
        model.fit(fit_ts)
        
        pp = PipelineParams()
        pipeline = AnomalyPipeline(model, pipeline_params=pp)
        pipeline.engine = AlertEngine() # use current alert logic
        preds = pipeline.predict(predict_ts)
        m = evaluate(true_incidents, preds)
        metrics_list[f"scenario_{i}"] = m
        
    agg = aggregate_results(metrics_list)
    return z, r, up, agg

if __name__ == '__main__':
    grid = []
    # Test a wide range of Mahalanobis thresholds (which scale differently)
    for z in [3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]:
        for r in [0.05, 0.1, 0.2, 0.3]:
            grid.append((z, r, 0.05))
                
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(eval_hparams, grid)
        
    best_f1 = -1
    best_res = None
    for z, r, up, agg in results:
        f1 = agg['f1_global']
        if f1 > best_f1:
            best_f1 = f1
            best_res = (z, r, up, agg)
            
    print("BEST MAHALANOBIS:", best_res[:3])
    print("METRICS:", best_res[3])
