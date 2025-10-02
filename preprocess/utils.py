import joblib
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_FILE = "xgb_model_simple.joblib"

def load_model_dict(path: str):
    obj = joblib.load(path)
    return obj["model"], obj["feature_list"], obj.get("q_abs", None)


def load_model_dict_simple(models_dir: Path | None = None, filename: str = MODEL_FILE):
    """โหลดโมเดลจาก models/<filename> รองรับทั้ง dict และ tuple payload"""
    base = models_dir or (Path(__file__).resolve().parents[1] / "models")
    p = base / filename
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    with p.open("rb") as f:
        obj = joblib.load(f)

    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj["feature_list"], obj.get("q_abs_last") or obj.get("q_abs")
    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        model = obj[0]
        feats = obj[1]
        q_abs = obj[2] if len(obj) > 2 else None
        return model, feats, q_abs
    raise ValueError(f"Unsupported model payload at {p}: {type(obj)}")

def compute_q_abs_from_train(model, X_train: pd.DataFrame, y_train: pd.Series, alpha=0.2) -> float:
    """Conformal absolute quantile (ง่ายๆ) จาก residual ของชุดเทรน/วาเลฯ"""
    preds = model.predict(X_train)
    q = float(np.quantile(np.abs(y_train.values - preds), 1 - alpha))
    return q

def rolling_flags(y_hist: list[float], window=14, k_soft=2.0, k_maint=3.0):
    """คืน (z, is_soft, is_maint) แบบ causal ใช้ค่าอดีต (shift(1))"""
    if len(y_hist) <= window:
        return None, False, False
    arr = np.array(y_hist, dtype=float)
    win = arr[-window-1:-1]  # shift(1)
    mu = win.mean()
    sd = win.std(ddof=0)
    if sd <= 0: sd = 1.0
    z = (arr[-1] - mu) / sd
    return float(z), (z < -k_soft), (z < -k_maint)
