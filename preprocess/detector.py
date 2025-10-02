from __future__ import annotations
from collections import deque
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    """
    Combined detector:
      - BLUE  : normal
      - ORANGE: maintenance (rule or flag)
      - GREEN : soft anomaly (z-score on y, no look-ahead)
      - RED   : ML anomaly (unsupervised model)
    Alerts fire only for GREEN/RED.

    Methods
    -------
    fit(X):                           train ML anomaly model (IsolationForest by default)
    predict(X) -> np.ndarray[bool]:   ML anomaly (RED) mask
    get_anomaly_scores(X) -> np.ndarray[float]
    transform_batch(df) -> pd.DataFrame: add columns roll_mean/roll_std/zscore/labels/colors/alert
    step_online(X_row, y_now) -> dict: online per-row labeling with rolling state
    """

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
        alert_include_maintenance: bool = False   # <— ใหม่: จะให้นับ ORANGE เป็น alert มั้ย
    ):
        self.alert_include_maintenance = alert_include_maintenance
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

    # ----------------- ML anomaly API (เดิม) -----------------
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
            # fallback: prob-based threshold
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

    def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # 1) rolling stats (no look-ahead)
        y = out[self.y_col].astype(float)
        minp = max(8, self.window // 2)
        mu = y.shift(1).rolling(self.window, min_periods=minp).mean()
        sd = y.shift(1).rolling(self.window, min_periods=minp).std()
        z = (y - mu) / (sd.replace(0, np.nan))
        out["roll_mean"] = mu
        out["roll_std"]  = sd
        out["zscore"]    = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 2) maintenance (rule + external flag)
        rule_maint = y <= (mu - self.k_maint * sd)
        if self.maintenance_col in out.columns:
            out[self.maintenance_col] = out[self.maintenance_col].astype(bool) | rule_maint.fillna(False)
        else:
            out[self.maintenance_col] = rule_maint.fillna(False)

        # 3) soft anomaly (GREEN) — ไม่ใช่ maintenance
        out["soft_anomaly"] = ((out["zscore"] <= -self.k_soft) & (~out[self.maintenance_col]))

        # 4) ML anomaly (RED)
        ml_mask = self.predict_safe(out)
        out["ml_anomaly"] = ml_mask
        out["anomaly_score"] = self.get_anomaly_scores_safe(out)

        # 5) label/color (priority: RED > ORANGE > GREEN > BLUE)
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

        # 6) alert policy (ตามพารามิเตอร์)
        if self.alert_include_maintenance:
            out["alert"] = out["label"].isin(["ml_anomaly", "soft_anomaly", "maintenance"])
        else:
            out["alert"] = out["label"].isin(["ml_anomaly", "soft_anomaly"])

        # 7) สร้าง episode_id สำหรับแต่ละชนิด (ช่วยให้ “แบ่งเป็นช่วง” เหมือนในรูป)
        out["episode_id"] = self._build_episode_id(out["label"])
        out["maint_episode_id"] = self._build_episode_id(out["label"].eq("maintenance"))
        out["ml_episode_id"]    = self._build_episode_id(out["label"].eq("ml_anomaly"))
        out["soft_episode_id"]  = self._build_episode_id(out["label"].eq("soft_anomaly"))

        return out

    # --- helpers สำหรับ episode ---
    def _build_episode_id(self, mask_or_series: pd.Series) -> pd.Series:
        """รับ mask/series แล้วนับ episode เฉพาะช่วงที่เป็น True/label นั้นๆ ต่อเนื่องกัน"""
        s = mask_or_series.copy()
        if s.dtype != bool:
            # ถ้าเป็น label (str) แปลงเป็น mask ของ label ที่สนใจก่อน (ผู้ใช้จะส่งมาเป็น Series str ได้)
            # ในที่นี้รองรับทั้ง str/bool: ถ้าเป็น str จะถือว่า True เมื่อ "มีค่า" (ไม่ว่าง)
            s = s.astype(bool)

        # เมื่อ False->True เริ่มตอนใหม่: cumsum ของ transition
        start = (~s.shift(1, fill_value=False) & s).astype(int)
        eid = (start.cumsum()) * s.astype(int)
        # ถ้าอยากให้ episode ไล่จาก 1,2,... และศูนย์เมื่อไม่ใช่ episode
        return eid.astype(int)
        # out = df.copy()

        # # rolling stats (no look-ahead)
        # y = out[self.y_col].astype(float)
        # mu = y.shift(1).rolling(self.window, min_periods=max(8, self.window // 2)).mean()
        # sd = y.shift(1).rolling(self.window, min_periods=max(8, self.window // 2)).std()
        # z = (y - mu) / (sd.replace(0, np.nan))
        # out["roll_mean"] = mu
        # out["roll_std"] = sd
        # out["zscore"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # # maintenance (rule + external flag if present)
        # rule_maint = y <= (mu - self.k_maint * sd)
        # has_flag = self.maintenance_col in out.columns
        # if has_flag:
        #     out[self.maintenance_col] = out[self.maintenance_col].astype(bool) | rule_maint.fillna(False)
        # else:
        #     out[self.maintenance_col] = rule_maint.fillna(False)

        # # soft anomaly (GREEN) — only if not maintenance
        # out["soft_anomaly"] = ((out["zscore"] <= -self.k_soft) & (~out[self.maintenance_col]))

        # #  ML anomaly (RED)
        # ml_mask = self.predict_safe(out)
        # out["ml_anomaly"] = ml_mask
        # out["anomaly_score"] = self.get_anomaly_scores_safe(out)

        # # label/color/alert (priority: RED > ORANGE > GREEN > BLUE)
        # out["label"] = np.select(
        #     [out["ml_anomaly"], out[self.maintenance_col], out["soft_anomaly"]],
        #     ["ml_anomaly", "maintenance", "soft_anomaly"],
        #     default="normal",
        # )
        # out["color"] = out["label"].map({
        #     "ml_anomaly": "red",
        #     "maintenance": "orange",
        #     "soft_anomaly": "green",
        #     "normal": "blue",
        # })
        # out["alert"] = out["label"].isin(["ml_anomaly", "soft_anomaly"])
        # return out

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

        # rolling z
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

        # maintenance (flag + rule)
        maint_flag = False
        if self.maintenance_col in X_row.columns:
            v = X_row[self.maintenance_col].iloc[0]
            try:
                if isinstance(v, (bool, np.bool_)):
                    maint_flag = bool(v)
                elif isinstance(v, (int, np.integer)):
                    maint_flag = (v == 1)
                elif isinstance(v, (float, np.floating)):
                    maint_flag = (not np.isnan(v)) and (int(v) == 1)
            except:
                maint_flag = False

        rule_maint = False
        if (y_now is not None) and (self._mu is not None) and (self._sd is not None):
            rule_maint = (y_now <= (self._mu - self.k_maint * self._sd))
        is_maintenance = maint_flag or rule_maint

        # soft anomaly
        is_soft = (y_now is not None) and (not is_maintenance) and (z <= -self.k_soft)

        # priority: RED > ORANGE > GREEN > BLUE
        reasons = []
        if ml_anom:
            label, color = "ml_anomaly", "red"
            reasons.append("ML anomaly (IsolationForest)")
        elif is_maintenance:
            label, color = "maintenance", "orange"
            r = []
            if maint_flag: r.append("flag")
            if rule_maint: r.append("rule")
            reasons.append("Maintenance: " + "+".join(r))
        elif is_soft:
            label, color = "soft_anomaly", "green"
            reasons.append(f"Soft anomaly (z={z:.2f} <= -{self.k_soft})")
        else:
            label, color = "normal", "blue"
            reasons.append("Normal")

        alert = (label in ["ml_anomaly", "soft_anomaly"]) or (self.alert_include_maintenance and label=="maintenance")

        return {
            "label": label,
            "color": color,
            "alert": alert,
            "y": y_now,
            "mu": self._mu,
            "sd": self._sd,
            "z": z,
            "ml_score": ml_score,
            "reasons": reasons
        }

        # assert isinstance(X_row, pd.DataFrame) and len(X_row) == 1
        # # ML anomaly
        # ml_anom, ml_score = False, None
        # if self._fitted:
        #     ml_anom = bool(self.predict(X_row)[0])
        #     ml_score = float(self.get_anomaly_scores(X_row)[0])

        # # rolling z-score
        # z = 0.0
        # if y_now is not None:
        #     self._y_buf.append(float(y_now))
        #     if len(self._y_buf) >= max(8, self.window // 2):
        #         arr = np.array(list(self._y_buf)[:-1], dtype=float)  # shift(1)
        #         if len(arr) > 0:
        #             mu = arr.mean()
        #             sd = arr.std() + 1e-9
        #             z = float((y_now - mu) / sd)
        #             self._mu, self._sd = mu, sd
        #     else:
        #         self._mu, self._sd = None, None

        # # maintenance
        # maint_flag = False
        # if self.maintenance_col in X_row.columns:
        #     v = X_row[self.maintenance_col].iloc[0]
        #     try:
        #         if isinstance(v, (bool, np.bool_)):
        #             maint_flag = bool(v)
        #         elif isinstance(v, (int, np.integer)):
        #             maint_flag = (v == 1)
        #         elif isinstance(v, (float, np.floating)):
        #             maint_flag = (not np.isnan(v)) and (int(v) == 1)
        #     except:
        #             maint_flag = False
        # # rule-based 
        # rule_maint = False
        # if (y_now is not None) and (self._mu is not None) and (self._sd is not None):
        #     rule_maint = (y_now <= (self._mu - self.k_maint * self._sd))
        # is_maintenance = maint_flag or rule_maint

        # # soft anomaly (ไม่ใช่ maintenance)
        # is_soft = (y_now is not None) and (not is_maintenance) and (z <= -self.k_soft)

        # # priority: RED > ORANGE > GREEN > BLUE
        # if ml_anom:
        #     label, color = "ml_anomaly", "red"
        #     reasons = ["Hard"]
        # elif is_maintenance:
        #     label, color = "maintenance", "orange"
        #     reasons = ["Maintenance (rule)"]
        # elif is_soft:
        #     label, color = "soft_anomaly", "green"
        #     reasons =  ["Soft anomaly (rolling z-score)"]
        

        # else:
        #     label, color = "normal", "blue"
        #     reasons = ["(forced) test"]


        # alert = (color in ["green", "red","orange"])
        # return {
        #     "label": label,
        #     "color": color,
        #     "alert": alert,
        #     "y": y_now,
        #     "mu": self._mu,
        #     "sd": self._sd,
        #     "z": z,
        #     "ml_score": ml_score,
        #     "reasons": reasons
        # }

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
