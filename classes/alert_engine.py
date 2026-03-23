from datetime import datetime, timezone
from typing import List

from .interface import AlertDecision, PredictOutput

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class AlertEngine:
    def __init__(
        self,
    ):
        self.locked = False

    def _has_alert(self, window_predictions: List[PredictOutput]) -> bool:
        return any(prediction.anomaly_status for prediction in window_predictions)

    def predict(self, window_predictions: List[PredictOutput]) -> AlertDecision:
        if len(window_predictions) == 0:
            return AlertDecision(
                alert=False,
                timestamp_range=(_EPOCH, _EPOCH),
                message="Empty window.",
            )
            
        has_alert = self._has_alert(window_predictions)
        
        anomalies = [p for p in window_predictions if p.anomaly_status]
        if anomalies:
            t_range = (anomalies[-1].timestamp, anomalies[-1].timestamp)
        else:
            t_range = (
                window_predictions[0].timestamp,
                window_predictions[-1].timestamp,
            )
        
        if has_alert:
            if self.locked:
                return AlertDecision(
                    alert=False,
                    timestamp_range=t_range,
                    message="System already entered abnormal state earlier.",
                )
            else:
                self.locked = True
                return AlertDecision(
                    alert=True,
                    timestamp_range=t_range,
                    message="Abnormal vibration detected.",
                )
        else:
            self.locked = False
            return AlertDecision(
                alert=False,
                timestamp_range=t_range,
                message="No persistent abnormal vibration.",
            )
