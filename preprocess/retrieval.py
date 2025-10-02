from pathlib import Path
import pandas as pd
import numpy as np
from preprocess.feature import FeaturePreprocessor
import joblib
from preprocess.utils import load_model_dict



def _load_model_dict_simple(filename: str = "new_model_mono.joblib"):
    p = Path(__file__).resolve().parents[1] / "models" / filename
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    with p.open("rb") as f:
        obj = joblib.load(f)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj["feature_list"], obj.get("q_abs_last") or obj.get("q_abs")
    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        return obj[0], obj[1], (obj[2] if len(obj) > 2 else None)
    raise ValueError(f"Unsupported model payload at {p}: {type(obj)}")

def prediction():
    """คืน DataFrame สำหรับกราฟ Predicted (+ CI ถ้ามี)"""
    model, feature_list_trained, q_abs = _load_model_dict_simple("new_model_mono.joblib")
    X, y, _ = dataModel() 

    missing = [c for c in feature_list_trained if c not in X.columns]
    if missing:
        raise ValueError(f"Missing features in X: {missing[:8]} (total {len(missing)})")

    Xp = X[feature_list_trained].copy().sort_index()
    yhat = model.predict(Xp)

    df_pred = pd.DataFrame({
        "datetime": Xp.index,
        "y_pred": yhat,
    }).reset_index(drop=True)

    if q_abs is not None:
        df_pred["y_lo"] = df_pred["y_pred"] - float(q_abs)
        df_pred["y_hi"] = df_pred["y_pred"] + float(q_abs)

    df_pred["value"] = df_pred["y_pred"]
    return df_pred[["datetime","value","y_lo","y_hi"] if "y_lo" in df_pred.columns else ["datetime","value"]]

def get_data():
    df = load_raw()
    date_rng = df.index
    values = df['CF-Total-Today']
    return pd.DataFrame({'datetime': date_rng, 'value': values})


def load_raw():
    df = pd.read_csv('preprocess/fullData.csv')
    df = df.set_index('time_block')
    df.index = pd.to_datetime(df.index)

    return df

def dataModel():
    df = load_raw()
    features = [col for col in df.columns if 'CF-Total-Today' not in col]
    fe = FeaturePreprocessor()
    X, y, feature_list = fe.make_features(
        df,  feature_cols=features, target_col="CF-Total-Today", horizon_steps=1
    )

    return X, y, feature_list

def recently_mock():

    df_mock = load_raw()

    return df_mock

