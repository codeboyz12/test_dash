
from __future__ import annotations
from collections import deque
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class AnomalyDetector:

    def __init__(
        self,
        contamination: float = 0.05,
        maintenance_col: str = "is_maintenance",
        y_col: str = "CF-Total-Today",
        window: int = 14,
        k_soft: float = 2.0,     # soft anomaly: z <= -k_soft
        k_maint: float = 3.0,    # maintenance rule: y <= mu - k_maint*std
        ml_model: Optional[Any] = None,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.maintenance_col = maintenance_col
        self.y_col = y_col
        self.window = window
        self.k_soft = k_soft
        self.k_maint = k_maint
        self.random_state = random_state


        self.ml_model = ml_model or IsolationForest(
            n_estimators=300, contamination=contamination, random_state=random_state
        )
        self.feature_cols: List[str] = []
        # online buffers
        self._y_buf = deque(maxlen=window)
        self._mu = None
        self._sd = None
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "AnomalyDetector":
        self.feature_cols = list(X.columns)
        self.ml_model.fit(X[self.feature_cols])
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """RED only (ML anomaly)"""
        self._ensure_fitted()
        try:
            # IsolationForest: predict -> 1(normal) / -1(anomaly)
            pred = self.ml_model.predict(X[self.feature_cols])
            return (pred == -1)
        except Exception:
            scores = self.get_anomaly_scores(X)
            thr = np.percentile(scores, 100 * (1 - self.contamination))
            return scores >= thr

    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        self._ensure_fitted()
        try:
            # IsolationForest: score_samples (สูง = ปกติ) -> ใช้สัญญาณกลับ
            return -self.ml_model.score_samples(X[self.feature_cols])
        except Exception:
            # fallback
            return np.zeros(len(X), dtype=float)

    # ----------------- Batch labeling (สำหรับกราฟ) -----------------
    def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        เพิ่มคอลัมน์:
          roll_mean, roll_std, zscore,
          soft_anomaly(bool), ml_anomaly(bool), is_maintenance(bool),
          anomaly_score(float), label(str), color(str), alert(bool)
        ใช้ no look-ahead: mean/std คำนวณบน y.shift(1)
        """
        out = df.copy()

        # 1) rolling stats 
        y = out[self.y_col].astype(float)
        mu = y.shift(1).rolling(self.window, min_periods=max(8, self.window // 2)).mean()
        sd = y.shift(1).rolling(self.window, min_periods=max(8, self.window // 2)).std()
        z = (y - mu) / (sd.replace(0, np.nan))
        out["roll_mean"] = mu
        out["roll_std"] = sd
        out["zscore"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 2) maintenance (rule + external flag if present)
        rule_maint = y <= (mu - self.k_maint * sd)
        has_flag = self.maintenance_col in out.columns
        if has_flag:
            out[self.maintenance_col] = out[self.maintenance_col].astype(bool) | rule_maint.fillna(False)
        else:
            out[self.maintenance_col] = rule_maint.fillna(False)

        # 3) soft anomaly (GREEN) — only if not maintenance
        out["soft_anomaly"] = ((out["zscore"] <= -self.k_soft) & (~out[self.maintenance_col]))

        # 4) ML anomaly (RED)
        ml_mask = self.predict_safe(out)
        out["ml_anomaly"] = ml_mask
        out["anomaly_score"] = self.get_anomaly_scores_safe(out)

        # 5) label/color/alert (priority: RED > ORANGE > GREEN > BLUE)
        out["label"] = np.select(
            [out["ml_anomaly"], out[self.maintenance_col], out["soft_anomaly"]],
            ["ml_anomaly", "maintenance", "soft_anomaly"],
            default="normal",
        )
        out["color"] = out["label"].map({
            "ml_anomaly": "red",
            "maintenance": "orange",
            "soft_anomaly": "green",
            "normal": "blue",
        })
        out["alert"] = out["label"].isin(["ml_anomaly", "soft_anomaly"])
        return out

    # ----------------- Online per-row (สำหรับสตรีม/Flask) -----------------
    def step_online(self, X_row: pd.DataFrame, y_now: Optional[float] = None) -> Dict[str, Any]:
        """
        ให้ 1 แถว (DataFrame 1xN) แล้วคืน dict: {label,color,alert,y,mu,sd,z,ml_score}
        ใช้ rolling buffer ภายในคลาส (no look-ahead)
        """
        assert isinstance(X_row, pd.DataFrame) and len(X_row) == 1
        # ML anomaly
        ml_anom, ml_score = False, None
        if self._fitted:
            ml_anom = bool(self.predict(X_row)[0])
            ml_score = float(self.get_anomaly_scores(X_row)[0])

        # rolling z-score
        z = 0.0
        if y_now is not None:
            self._y_buf.append(float(y_now))
            if len(self._y_buf) >= max(8, self.window // 2):
                arr = np.array(list(self._y_buf)[:-1], dtype=float)  # shift(1)
                if len(arr) > 0:
                    mu = arr.mean()
                    sd = arr.std() + 1e-9
                    z = float((y_now - mu) / sd)
                    self._mu, self._sd = mu, sd
            else:
                self._mu, self._sd = None, None

        # maintenance
        maint_flag = False
        if self.maintenance_col in X_row.columns:
            maint_flag = bool(int(X_row[self.maintenance_col].iloc[0]) == 1)
        # rule-based (เฉพาะตอนรู้ y_now และมีสถิติ)
        rule_maint = False
        if (y_now is not None) and (self._mu is not None) and (self._sd is not None):
            rule_maint = (y_now <= (self._mu - self.k_maint * self._sd))
        is_maintenance = maint_flag or rule_maint

        # soft anomaly (ไม่ใช่ maintenance)
        is_soft = (y_now is not None) and (not is_maintenance) and (z <= -self.k_soft)

        # priority: RED > ORANGE > GREEN > BLUE
        if ml_anom:
            label, color = "ml_anomaly", "red"
        elif is_maintenance:
            label, color = "maintenance", "orange"
        elif is_soft:
            label, color = "soft_anomaly", "green"
        else:
            label, color = "normal", "blue"

        alert = (color in ["green", "red"])
        return {
            "label": label,
            "color": color,
            "alert": alert,
            "y": y_now,
            "mu": self._mu,
            "sd": self._sd,
            "z": z,
            "ml_score": ml_score
        }

    # ----------------- helpers -----------------
    def predict_safe(self, df: pd.DataFrame) -> np.ndarray:
        try:
            return self.predict(df[self.feature_cols])
        except Exception:
            # ถ้า fit ด้วยฟีเจอร์ชุดหนึ่ง แต่ df ยังไม่มี ให้พยายามจับคู่ชื่อ
            cols = [c for c in self.feature_cols if c in df.columns]
            if not cols:
                return np.zeros(len(df), dtype=bool)
            return self.predict(df[cols])

    def get_anomaly_scores_safe(self, df: pd.DataFrame) -> np.ndarray:
        try:
            return self.get_anomaly_scores(df[self.feature_cols])
        except Exception:
            cols = [c for c in self.feature_cols if c in df.columns]
            if not cols:
                return np.zeros(len(df), dtype=float)
            return self.get_anomaly_scores(df[cols])

    def _ensure_fitted(self):
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit(X) first.")
