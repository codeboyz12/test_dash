from pathlib import Path
from threading import Lock
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template
from flask_socketio import SocketIO
from datetime import datetime

import shap

from preprocess.utils import load_model_dict_simple, compute_q_abs_from_train, rolling_flags
from preprocess.mainpipe import ga_shap_narrative
from preprocess.ga import GeneticAlgorithm
from preprocess.detector import AnomalyDetector

TARGET_COL = "CF-Total-Today"
ROLL_W     = int(os.getenv("ROLL_W", "14"))
K_SOFT     = float(os.getenv("K_SOFT", "2.0"))
K_MAINT    = float(os.getenv("K_MAINT", "3.0"))
SPEED      = float(os.getenv("SPEED", "4.0"))
BASE_STEP  = float(os.getenv("BASE_STEP", "0.25"))
CHUNK_EMIT = int(os.getenv("CHUNK_EMIT", "5"))
STEP_DELAY = BASE_STEP / max(1.0, SPEED)
EVENT_STREAM = "updateSensorData"
EVENT_ALERT  = "alertEvent"
FORCE_ALERT_AFTER = int(os.getenv("FORCE_ALERT_AFTER", "0"))



try:
    from preprocess.retrieval import dataModel
except Exception:
    dataModel = None

app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = "dev"
socketio = SocketIO(app, cors_allowed_origins="*")
_thread = None
_lock = Lock()

@app.route("/")
def index():
    return render_template("index.html")

def _load_data_for_stream(model, feature_list_trained):
    if dataModel is None:
        raise RuntimeError("preprocess.retrieval.dataModel() not found. Please provide it.")
    X, y, _ = dataModel()
    missing = [c for c in feature_list_trained if c not in X.columns]
    if missing:
        raise ValueError(f"Missing features in X: {missing[:8]} (total {len(missing)})")
    X = X[feature_list_trained].copy().sort_index()
    df = X.copy()
    df[TARGET_COL] = y.loc[X.index].values
    return df

def _build_alert_card(ts, y_true, y_pred, lo, label, reasons, z, *,
                      X_one=None, model=None, feature_names=None):
    """เด้งป๊อปอัพ: รวม GA/SHAP/Narrative"""
    variant = "info" if label == "maintenance" else "danger"
    title   = "Maintenance Window" if label == "maintenance" else "Yield Risk Detected"
    summary = ("ช่วงบำรุงรักษา/ผลผลิตลดลงชั่วคราว" if label == "maintenance"
               else "คาดว่าประสิทธิภาพลดลงจากแนวโน้มล่าสุด")

    details = {
        "label": label, "reasons": reasons or [], "zscore": z,
        "y_true": y_true, "y_pred": y_pred, "low_band": lo,
        "shap_top": [], "ga": {"suggestion": {}, "best_score": None},
        "genai": {"summary": "", "actions": [], "caveats": []},
    }

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_one)

    # ต่อ GA/SHAP/Narrative 
    if label in ("soft_anomaly", "ml_anomaly") and X_one is not None:
        try:
            res = ga_shap_narrative(
                X_one=X_one, model=model, feature_names=list(X_one.columns),
                y_true=y_true, yield_threshold=lo,
                shap_explainer=explainer, 
                controllable_features=["R411_Temp", "R412_Temp", "R422_Temp", "R423_Temp ", "R424_Temp", "R425_Temp", "R426_Temp", "R427_Temp ", "FIC421", "FIC422"],
                controllable_bounds=None,
            )
            details["shap_top"] = [[str(k), float(v)] for (k, v) in res.get("shap_top", [])]
            g = res.get("ga", {})
            details["ga"]["suggestion"] = g.get("suggestion", {})
            details["ga"]["best_score"] = g.get("best_score", None)
            details["genai"] = res.get("gen_ai", details["genai"])
        except Exception as e:
            details["genai"]["summary"] = f"(pipeline error: {e})"

    return {
        "ts": ts, "variant": variant, "title": title, "summary": summary, "details": details
    }



EVENT_STREAM = "updateSensorData"
EVENT_ALERT  = "alertEvent"

def _build_alert_card_from_result(ts, y_true, result):
    """map output จาก pipeline -> alert card ที่หน้าเว็บใช้"""
    return {
        "ts": ts,
        "variant": "danger" if result["label"] in ("soft_anomaly","ml_anomaly") else "info",
        "title": "Yield Risk Detected" if result["label"]!="maintenance" else "Maintenance Window",
        "summary": result.get("gen_ai",{}).get("summary") or "Detected anomaly",
        "details": {
            "label": result["label"],
            "reasons": result.get("alert_reasons", []),
            "zscore": result.get("anomaly_score"),
            "y_true": y_true,
            "y_pred": result.get("predicted_yield"),
            "low_band": None,  # (ถ้าคุณมี dynamic band ก็ใส่เพิ่มได้)
            "shap_top": [[str(k), float(v)] for (k, v) in result.get("top_shap", [])],
            "ga": {
                "suggestion": result.get("ga",{}).get("suggestion", {}),
                "best_score": result.get("ga",{}).get("best_score", None),
            },
            "genai": result.get("gen_ai", {"summary":"", "actions":[], "caveats":[]}),
        }
    }

def load_n_fit_detect(df: pd.DataFrame, features: list,ycol="CF-Total-Today") -> AnomalyDetector:#<-- แก้ใหม่ตรงนี้
    detector = AnomalyDetector(
            contamination=0.05,
            maintenance_col="is_maintenance",
            y_col=ycol,
            window=14,
            k_soft=2.0,
            k_maint=3.0
    )
    detector.fit(df[features])
    return detector



def stream_loop():
    model, feat_list, q_abs = load_model_dict_simple()
    df = _load_data_for_stream(model, feat_list)
    df['is_maintenance'] = (df['CF-Total-Today'] < abs(df['CF-Total-Today'].mean()-df['CF-Total-Today'].std())).astype(int)#<-- แก้ใหม่ตรงนี้ด้วย
    
    detector = load_n_fit_detect(df, feat_list) #<-- แก้ใหม่ตรงนี้

 
    if q_abs is None:
        q_abs = compute_q_abs_from_train(model, df[feat_list], df[TARGET_COL])

    y_hist: list[float] = []
    rows = list(df[feat_list].iterrows())
    i, n = 0, len(rows)

    while i < n:
        stream_payloads = []
        alert_cards     = []

        for _ in range(CHUNK_EMIT):
            if i >= n: break
            idx, row = rows[i]
            X_one  = row.to_frame().T
            y_true = float(df.loc[idx, TARGET_COL])
            y_hist.append(y_true)

            try:
                y_pred = float(model.predict(X_one)[0])
            except Exception:
                y_pred = float("nan")

            lo = (y_pred - q_abs) if np.isfinite(y_pred) else None
            hi = (y_pred + q_abs) if np.isfinite(y_pred) else None

            out = detector.step_online(X_one, y_now=y_true)#<-- แก้ใหม่ตรงนี้
            out.setdefault("y_pred", y_pred)
            out.setdefault("z", None)
            out.setdefault("ml_score", None)
            out.setdefault("color", "red" if out.get("label") == "ml_anomaly" else "blue")#<-- แก้ใหม่ตรงนี้

            # z, is_soft, is_maint = rolling_flags(y_hist, window=ROLL_W, k_soft=K_SOFT, k_maint=K_MAINT)
            # label, color, alert, reasons = "normal", "blue", False, []
            # if is_maint:
            #     label, color, alert, reasons = "maintenance", "orange", True, ["Maintenance (rule)"]
            # elif is_soft:
            #     label, color, alert, reasons = "soft_anomaly", "green", True, ["Soft anomaly (rolling z-score)"]

            # if FORCE_ALERT_AFTER and i == FORCE_ALERT_AFTER:
            #     alert = True; label, color, reasons = "soft_anomaly", "green", ["(forced) test"]

            ts = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
            stream_payloads.append({
                "ts": ts, "y_true": y_true, "y_pred": y_pred,
                "y_lo": lo, "y_hi": hi, "label": out['label'], "color": out['color'],
                "alert": out['alert'], "reasons": out['reasons'], "z": out['z'],
            })

            if out['alert']:
                card = _build_alert_card(ts, y_true, y_pred, lo, out['label'], out['reasons'], out['z'],
                                         X_one=X_one, model=model, feature_names=list(X_one.columns))
                alert_cards.append(card)

            i += 1

        if stream_payloads:
            socketio.emit(EVENT_STREAM, stream_payloads)
        if alert_cards:
            print(f"[server] alertEvent x{len(alert_cards)}")
            socketio.emit(EVENT_ALERT,  alert_cards)

        socketio.sleep(STEP_DELAY)

from flask_socketio import emit

@socketio.on("connect")
def on_connect():
    global _thread
    print("[socket] client connected")
    with _lock:
        if _thread is None:
            _thread = socketio.start_background_task(stream_loop)

@socketio.on("disconnect")
def on_disconnect():
    print("[socket] client disconnected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080, debug=True)
