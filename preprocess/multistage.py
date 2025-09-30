from collections import deque
import numpy as np
import pandas as pd

class RollingZScore:
    def __init__(self, window=24, clip=8.0):
        self.buf = deque(maxlen=window); self.clip = clip
    def update(self, residual: float) -> float:
        self.buf.append(residual)
        if len(self.buf) < 8: return 0.0
        arr = np.array(self.buf); m, s = arr.mean(), arr.std() + 1e-9
        return float(np.clip((residual - m) / s, -self.clip, self.clip))

class MaintenanceRule:
    def __init__(self, flag_col="is_maintenance", drop_limit=-1.0, lookback=2):
        self.flag_col, self.drop_limit = flag_col, drop_limit
        self.last = deque(maxlen=lookback+1)
    def update(self, X_row: pd.DataFrame, y_now: float) -> bool:
        if self.flag_col in X_row.columns and int(X_row[self.flag_col].iloc[0]) == 1:
            return True
        self.last.append(y_now)
        if len(self.last) == self.last.maxlen and (self.last[-1] - self.last[0]) <= self.drop_limit:
            return True
        return False

class MultiStageDetector:

    def __init__(self, forecaster, feature_cols, z_window=24, z_thr=2.5,
                 maintenance_flag="is_maintenance", maintenance_drop=-1.0,
                 ml_detector=None):
        self.f = forecaster
        self.cols = feature_cols
        self.z_thr = z_thr
        self.rz = RollingZScore(window=z_window)
        self.mrule = MaintenanceRule(flag_col=maintenance_flag, drop_limit=maintenance_drop)
        self.ml = ml_detector

    def step(self, X_row: pd.DataFrame, y_true: float | None):
        y_pred = float(self.f.predict(X_row[self.cols])[0])
        residual, z = (None, 0.0) if y_true is None else (y_true - y_pred, 0.0)
        if y_true is not None:
            z = self.rz.update(residual)

        # orange: maintenance
        is_maint = self.mrule.update(X_row, y_true if y_true is not None else y_pred)

        # red: ML anomaly
        is_ml, ml_score = False, None
        if self.ml is not None:
            try:
                ml_score = float(self.ml.predict_proba(X_row[self.cols])[:, 1][0])
                is_ml = ml_score >= 0.5
            except Exception:
                is_ml = bool(self.ml.predict(X_row[self.cols])[0])

        # green: soft anomaly (ใช้ก็ต่อเมื่อไม่ใช่ maintenance)
        is_soft = (abs(z) >= self.z_thr) if (y_true is not None and not is_maint) else False

        if is_ml:
            label, color = "ml_anomaly", "red"
        elif is_maint:
            label, color = "maintenance", "orange"
        elif is_soft:
            label, color = "soft_anomaly", "green"
        else:
            label, color = "normal", "blue"

        alert = color in ["green", "red"]
        return {
            "label": label, "color": color, "alert": alert,
            "y_pred": y_pred, "residual": residual, "z": z, "ml_score": ml_score
        }
