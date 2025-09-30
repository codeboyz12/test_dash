"""
-> Main funtion work:
    ethanol_pipeline_full_with_ga() : pipeline detect + alert + predict + shap
    run_monitor_loop() : run function above since fisrt row in dataset until last

-> Work Flow
    -read df
    -define all attribute that parameter want
    -use run_monitor_loop()

    example:
    --read df--
    df = pd.read_csv('data/fullData.csv')
    df = df.set_index('time_block')
    df.index = pd.to_datetime(df.index)

    --define all attribute--
    for fold, X_train, y_train, X_val, y_val, X_test, y_test in time_series_train_val_test_split(df, featuresXX, "CF-Total-Today", n_splits=3, val_size=0.1):
        print(f"Fold {fold}:")
        print(f"  Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    
    target_col = 'CF-Total-Today'
    forecast_horizon = 4           
    featuresXX = [col for col in df.columns]            
    window = 14                    
    zscore_threshold = -1.5    

    featuresXX = [col for col in df.columns if 'CF-Total-Today' not in col]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    controllable_features = ["Inprocess_Preferment", "Inprocess_Ferment", "Inprocess_Total","CFconvert-Total-Today"]
    cont_bounds = [
        (
            df_train[f].mean() - 2 * df_train[f].std(),
            df_train[f].mean() + 2 * df_train[f].std()
        )
        for f in controllable_features
    ]
    

    ycol = "CF-Total-Today"
    detector = AnomalyDetector(
        contamination=0.05,
        maintenance_col="is_maintenance",
        y_col=ycol,
        window=14,
        k_soft=2.0,
        k_maint=3.0
    )
    detector.fit(df[featuresXX])

    policy = ThresholdPolicy(static_threshold=0.10, window=168, percentile=0.10, source="y")

    --run main loop--
    events_df = run_monitor_loop(
        df=df,
        featuresXX=featuresXX,
        model=model,
        shap_explainer=explainer,                 # ถ้าไม่มี ใส่ None ได้
        detector=detector,                        # ถ้า detector คุณไม่มี step_online ให้ใส่ None เพื่อให้ลูปสร้างตัวที่มีได้เอง
        df_train=X_train,
        controllable_features=["Inprocess_Preferment","Inprocess_Ferment","Inprocess_Total","CFconvert-Total-Today"],
        max_rows=200,                             # จำกัดจำนวนสำหรับเทส
        target_col="CF-Total-Today",
        window=14,
        k_soft=2.0,
        k_maint=3.0,
        yield_threshold=0.10,                     # fallback เผื่อช่วงข้อมูลยังน้อย
        threshold_policy=policy,                  # << ใช้ threshold ไดนามิก
        log_events=True,
        return_events_df=True,                    # << ขอคืนเป็น DataFrame
        save_events_csv=None,                     # จะบันทึก CSV ก็ใส่พาธไฟล์
        sse_queue=None                            # << ไม่ใช้ Flask/SSE
    )
"""


from detector import AnomalyDetector
from collections import deque
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import warnings
import joblib
import os
import json
from typing import Dict, Any
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from utils import Utils

warnings.filterwarnings('ignore')

path = ''

def _topn_from_dict(d: Dict[str, float], n: int = 5):
    return sorted(d.items(), key=lambda x: abs(x[1]), reverse=True)[:n]

def _normalize_ga(ga_raw) -> Dict[str, Any]:
    out = {"suggestion": {}, "best_score": None, "full_vector": {}, "history": []}
    if ga_raw is None:
        return out
    if isinstance(ga_raw, tuple):
        sug, best = ga_raw
        if isinstance(sug, dict):
            out["suggestion"] = sug
        out["best_score"] = best
        return out
    if isinstance(ga_raw, dict):
        if "best_individual" in ga_raw or "best_score" in ga_raw:
            out["best_score"] = ga_raw.get("best_score")
            out["history"] = ga_raw.get("history", [])
            best_ind = ga_raw.get("best_individual")
            out["full_vector"] = ga_raw
            return out
        out["suggestion"] = ga_raw
        return out
    return out

def normalize_alert_info(alert_info):
    if isinstance(alert_info.get("shap_values"), (list, tuple)):
        alert_info["shap_values"] = dict(alert_info["shap_values"])
    if isinstance(alert_info.get("ga_suggestion"), (list, tuple)):
        alert_info["ga_suggestion"] = dict(alert_info["ga_suggestion"])
    return alert_info

def build_alert_prompt(alert_info: Dict[str, Any]) -> str:
    pred = alert_info.get("predicted_value")
    step = alert_info.get("step")
    shap_vals = alert_info.get("shap_values") or {}
    ga_result = alert_info.get("ga_suggestion") or {}
    anomaly_score = alert_info.get("anomaly_score")

    best_ind = ga_result.get("best_individual")
    best_score = ga_result.get("best_score")
    top_shap = (
        ", ".join([f"{k}:{v:.3f}" for k, v in _topn_from_dict(shap_vals, 6)])
        if shap_vals else "N/A"
    )
    if isinstance(best_ind, dict):
        top_ga = ", ".join([f"{k}:{v:.3f}" for k, v in list(best_ind.items())[:6]])
    elif isinstance(best_ind, (list, tuple)):
        top_ga = ", ".join([f"x{i}:{v:.3f}" for i, v in enumerate(best_ind[:6])])
    else:
        top_ga = "N/A"
    ga_info = f"Best score={best_score:.4f}, Best individual=({top_ga})" if best_score is not None else "N/A"

    prompt = f"""
You are an industrial process assistant for ethanol production.
A risk alert was raised for forecast horizon +{step}h:
- Predicted ethanol yield: {pred:.4f}
- Anomaly model score: {anomaly_score}

Top contributing features (SHAP): {top_shap}
GA optimization results: {ga_info}

Please:
1) Provide a concise (2-3 sentence) explanation of why the yield may drop.
2) List 3 prioritized actionable steps (short imperative instructions) that operators can try now, based on the GA suggestion and typical constraints.
3) List any caveats or checks the operator should perform before applying changes.

Format the response as JSON with keys: "summary", "actions", "caveats".
"""
    return prompt.strip()

def call_gen_ai(alert_info: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_alert_prompt(alert_info)
    shap_top = _topn_from_dict(alert_info.get("shap_values", {}), 3)
    shap_names = [k for k, _ in shap_top]
    text = {
        "summary": f"Predicted yield {alert_info.get('predicted_value'):.4f} with high anomaly score {alert_info.get('anomaly_score')}. Key drivers: {', '.join(shap_names)}.",
        "actions": [
            "Verify sensor readings and recent setpoint changes.",
            "Apply GA-suggested setpoint adjustments within safe bounds.",
            "Monitor yield trend for the next 2 cycles and reassess."
        ],
        "caveats": [
            "Confirm maintenance status and cleaning cycles.",
            "Check raw material quality variation.",
            "Validate that adjustments do not violate safety constraints."
        ]
    }
    return text

def ethanol_pipeline_full_with_ga(
    features: pd.DataFrame,                 # 1 แถว
    forecaster,                             # โมเดล forecast
    shap_explainer=None,
    detector=None,                          # รองรับทั้ง MultiStageDetector และ AnomalyDetector เดิม
    y_true: float | None = None,            # ถ้ามี (offline/replay) จะใช้ soft anomaly ได้ทันที
    ga_optimizer=None,
    yield_threshold: float | None = None,
    feature_names: list | None = None,
    df_train: pd.DataFrame | None = None,
    controllable_features: list | None = None,
    controllable_bounds: dict | None = None,
    step_ahead: int = 4,
    run_ga_labels: tuple = ("soft_anomaly","ml_anomaly")
):
    # ---------- sanitize inputs ----------
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame([features])
    if feature_names is None:
        feature_names = list(features.columns)

    def _detect_with_fallback(detector, X_row: pd.DataFrame):
        # พยากรณ์เสมอ (ใช้แสดงผล/threshold)
        y_pred = float(forecaster.predict(X_row[feature_names])[0])

        # maintenance flag จากฟีเจอร์ (ถ้ามี)
        maint_flag = False
        if "is_maintenance" in X_row.columns:
            try:
                maint_flag = bool(int(X_row["is_maintenance"].iloc[0]) == 1)
            except Exception:
                maint_flag = bool(X_row["is_maintenance"].iloc[0])

        # 1) ถ้ามี .step(...) (MultiStageDetector)
        if hasattr(detector, "step") and callable(getattr(detector, "step")):
            out = detector.step(X_row, y_true=y_true)
            out.setdefault("y_pred", y_pred)
            out.setdefault("z", None)
            out.setdefault("ml_score", None)
            return out

        # 2) ถ้ามี .step_online(...)
        if hasattr(detector, "step_online") and callable(getattr(detector, "step_online")):
            out = detector.step_online(X_row, y_now=y_true)
            out.setdefault("y_pred", y_pred)
            out.setdefault("z", None)
            out.setdefault("ml_score", None)
            out.setdefault("color", "red" if out.get("label") == "ml_anomaly" else "blue")
            return out

        # 3) fallback: มีแต่ predict/score (AnomalyDetector เดิม)
        ml_flag = False
        ml_score = None
        try:
            ml_arr = np.asarray(detector.predict(X_row[feature_names])).ravel()
            if ml_arr.size > 0:
                ml_flag = bool(ml_arr[0])
        except Exception:
            pass
        try:
            sc_arr = np.asarray(detector.get_anomaly_scores(X_row[feature_names])).ravel()
            if sc_arr.size > 0:
                ml_score = float(sc_arr[0])
        except Exception:
            pass

        if ml_flag:
            label, color, alert = "ml_anomaly", "red", True
        elif maint_flag:
            label, color, alert = "maintenance", "orange", False
        else:
            label, color, alert = "normal", "blue", False

        return {"label": label, "color": color, "alert": alert,
                "y_pred": y_pred, "z": None, "ml_score": ml_score}

    # ถ้าไม่ได้ส่ง detector มา สร้าง MultiStageDetector เอง (ค่า default)
    if detector is None:
        try:
            detector = MultiStageDetector(
                forecaster=forecaster,
                feature_cols=feature_names,
                z_window=24, z_thr=2.5,
                maintenance_flag="is_maintenance",
                maintenance_drop=-1.0,
                ml_detector=None
            )
        except Exception:
            detector = object()  # บังคับให้ไปใช้ fallback แบบปกติ

    det = _detect_with_fallback(detector, features)
    pred   = float(det.get("y_pred"))
    label  = det.get("label", "normal")
    color  = det.get("color", "blue")
    is_alert = bool(det.get("alert", False))

    # ---------- alert reasons ----------
    alert_reasons = []
    if label == "ml_anomaly":    alert_reasons.append("ML anomaly")
    if label == "soft_anomaly":  alert_reasons.append("Soft anomaly (rolling z-score)")
    if label == "maintenance":   alert_reasons.append("Maintenance (rule)")
    if yield_threshold is not None and pred < yield_threshold:
        is_alert = True
        alert_reasons.append(f"Predicted yield {pred:.4f} below threshold {yield_threshold}")

    out = {
        "label": label,                         # normal/maintenance/soft_anomaly/ml_anomaly
        "color": color,                         # blue/orange/green/red
        "alert": is_alert,                      # True เฉพาะ green/red + เงื่อนไขเสริม
        "alert_reasons": alert_reasons,
        "predicted_yield": pred,
        "is_anomaly": label in ["soft_anomaly", "ml_anomaly"],
        "anomaly_score": (abs(det["z"]) if det.get("z") is not None else det.get("ml_score")),
    }

    # ---------- SHAP (เฉพาะตอน alert) ----------
    shap_dict = {}
    if is_alert and shap_explainer is not None:
        try:
            sv = shap_explainer.shap_values(features[feature_names])
            if isinstance(sv, list) or (hasattr(sv, "ndim") and sv.ndim > 2):
                sv = sv[0]
            shap_vals = np.array(sv).reshape(-1)
            shap_dict = dict(zip(feature_names, shap_vals))
        except Exception:
            shap_dict = {}
    out["top_shap"] = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10] if shap_dict else []
    if shap_dict:
        out["shap_values"] = shap_dict

    # ---------- GA (เฉพาะตอน alert และ label อยู่ใน run_ga_labels) ----------
    ga_block = {"suggestion": {}, "best_score": pred, "full_vector": {}, "history": []}
    run_ga = is_alert and (label in run_ga_labels)
    if run_ga:
        if controllable_features is None:
            controllable_features = [
                "Inprocess_Preferment","Inprocess_Ferment",
                "Inprocess_Total","CFconvert-Total-Today"
            ]
        if controllable_bounds is None:
            base = features.iloc[0]
            controllable_bounds = {
                f: (float(base[f]) * 0.8, float(base[f]) * 1.2)
                for f in controllable_features if f in features.columns
            }
        controllable_features = [c for c in controllable_features if c in features.columns]
        bounds_list = [controllable_bounds[c] for c in controllable_features if c in controllable_bounds]

        def _predict_fn(ind):
            row = features.copy(); idx0 = row.index[0]
            for i, f in enumerate(controllable_features):
                row.at[idx0, f] = float(ind[i])
            return float(forecaster.predict(row[feature_names])[0])

        ga_res = None
        if bounds_list:
            if ga_optimizer is None:
                from Ga.GA import GeneticAlgorithm
                ga = GeneticAlgorithm(
                    cont_bounds=bounds_list, n_binary=0, predict_fn=_predict_fn,
                    pop_size=24, generations=30, crossover_rate=0.7,
                    mutation_rate=0.2, elite_frac=0.15, maximize=True
                )
                ga_res = ga.evolve(verbose=False)
            else:
                ga_res = ga_optimizer.evolve_with_predictor(_predict_fn, bounds=bounds_list)

        if isinstance(ga_res, dict) and "best_individual" in ga_res:
            best = ga_res["best_individual"]
            sugg = {f: float(best[i]) for i, f in enumerate(controllable_features)}
            ga_block = {
                "suggestion": sugg,
                "best_score": float(ga_res.get("best_score", pred)),
                "full_vector": ga_res,
                "history": ga_res.get("history", [])
            }
    out["ga"] = ga_block

    # ---------- Narrative (เฉพาะตอน alert) ----------
    if is_alert:
        top_feats = out.get("top_shap", [])
        ga_sug   = out["ga"].get("suggestion", {})
        text = (
            f"Label={label}. Pred={pred:.4f}. "
            f"Top: {', '.join([f'{k}:{v:+.3f}' for k, v in top_feats[:5]]) or '-'}; "
            f"GA: {', '.join([f'{k}->{v:.3f}' for k, v in ga_sug.items()]) or '-'}."
        )
        out["gen_ai"] = {"summary": text, "actions": [], "caveats": []}
    else:
        out["gen_ai"] = {"summary": "", "actions": [], "caveats": []}

    return out

def show_ga_result_strict(ga_result, controllable_features, base_row, model, feature_names):
    """
    แสดงผล GA suggestion โดยเทียบ prediction ก่อน/หลังแก้
    - ga_result: dict ที่ได้จาก GA
    - controllables: list ชื่อฟีเจอร์ที่ GA ปรับได้
    - base_row: DataFrame (1 แถว) ที่เป็นจุดเริ่ม
    - model: โมเดล forecaster
    - feature_names: ฟีเจอร์ทั้งหมดที่โมเดลใช้ train
    """
    if "best_individual" in ga_result:
        best_ind = ga_result["best_individual"]
        sug_map = {f: float(best_ind[i]) for i, f in enumerate(controllable_features)}
        best_score = ga_result.get("best_score")
    else:
        sug_map = ga_result.get("suggestion", {})
        best_ind = [sug_map.get(f, float(base_row[f].iloc[0])) for f in controllable_features]
        best_score = ga_result.get("best_score")

    pred_before = float(model.predict(base_row[feature_names])[0])
    row_after = base_row.copy()
    for i, f in enumerate(controllable_features):
        row_after.at[row_after.index[0], f] = float(best_ind[i])
    pred_after = float(model.predict(row_after[feature_names])[0])

    print("=== GA Suggestion Result ===")
    print(f"Prediction before: {pred_before:.4f}")
    print(f"Prediction after : {pred_after:.4f}")
    if best_score is not None:
        print(f"GA best score   : {best_score:.4f}")
    print(f"Improvement      : {pred_after - pred_before:.4f}")

    print("\nAdjusted features:")
    for i, f in enumerate(controllable_features):
        old_val = float(base_row[f].iloc[0])
        new_val = float(row_after[f].iloc[0])
        if abs(new_val - old_val) > 1e-9:
            print(f" - {f}: {old_val:.4f} -> {new_val:.4f}")

class ThresholdPolicy:
    """
    Threshold แบบไดนามิกจากเปอร์เซ็นไทล์ของค่าล่าสุดในหน้าต่างเวลา
    - static_threshold: ค่า fallback ช่วงข้อมูลยังน้อย
    - window: ความยาวบัฟเฟอร์ (เช่น 168 ชั่วโมง = 7 วัน)
    - percentile: ใช้เปอร์เซ็นไทล์เท่าไร (เช่น 0.10 = P10)
    - source: 'y' ใช้ค่าจริง, 'pred' ใช้ค่าพยากรณ์
    """
    def __init__(self, static_threshold: float = None,
                 mode: str = "rolling_percentile",
                 window: int = 168,
                 percentile: float = 0.10,
                 source: str = "y"):
        self.static = static_threshold
        self.mode = mode
        self.window = window
        self.percentile = percentile
        self.source = source
        self.buf = deque(maxlen=window)

    def update_and_get(self, y_pred: float = None, y_true: float = None) -> float:
        x = y_true if (self.source == "y" and y_true is not None) else y_pred
        if x is not None:
            self.buf.append(float(x))
        if self.mode != "rolling_percentile" or len(self.buf) < max(8, self.window // 4):
            return self.static
        arr = np.array(self.buf, dtype=float)
        return float(np.quantile(arr, self.percentile))

import json
import pandas as pd

def run_monitor_loop(
    df: pd.DataFrame,
    featuresXX: list,
    model,
    shap_explainer=None,
    detector=None,                    # ต้องมี step_online() ตามที่เราทำไว้
    ga_optimizer=None,
    # ---- threshold: ส่งทั้ง static และ/หรือ policy ได้ (ถ้าส่ง policy จะใช้มัน) ----
    yield_threshold: float = 0.1,
    threshold_policy: ThresholdPolicy = None,
    # ---- training/context ----
    df_train: pd.DataFrame = None,
    controllable_features: list = None,
    show_ga_result=None,
    max_rows: int = None,
    # ---- กติกา ----
    target_col: str = "CF-Total-Today",
    window: int = 14,
    k_soft: float = 2.0,
    k_maint: float = 3.0,
    # ---- Logging / Streaming ----
    log_events: bool = True,
    return_events_df: bool = True,      # True → return DataFrame ตอนจบ
    save_events_csv: str = None,        # path เช่น "/tmp/events.csv"
    sse_queue=None                      # queue.Queue สำหรับส่ง event แบบ SSE
):
    """
    สี/พฤติกรรม:
      - normal (ฟ้า)            : ไม่ alert
      - maintenance (ส้ม)       : alert อย่างเดียว (ไม่รัน GA)
      - soft anomaly (เขียว)    : **alert ก็ต่อเมื่อ value < threshold(ไดนามิก/สแตติก)** + GA
      - ml anomaly (แดง)        : alert + GA
    """
    # --- maintenance (no look-ahead) ---
    y = df[target_col].astype(float)
    mu = y.shift(1).rolling(window, min_periods=max(8, window // 2)).mean()
    sd = y.shift(1).rolling(window, min_periods=max(8, window // 2)).std()
    rule_maint = (y <= (mu - k_maint * sd))
    if "is_maintenance" in df.columns:
        df["is_maintenance"] = df["is_maintenance"].astype(bool) | rule_maint.fillna(False)
    else:
        df["is_maintenance"] = rule_maint.fillna(False)

    # --- detector (มี step_online) ---
    if detector is None:
        detector = AnomalyDetector(
            contamination=0.05,
            maintenance_col="is_maintenance",
            y_col=target_col,
            window=window,
            k_soft=k_soft,
            k_maint=k_maint
        )
    det_features = list(featuresXX)
    if "is_maintenance" not in det_features:
        det_features += ["is_maintenance"]
    detector.fit(df[det_features])

    # --- event accumulator ---
    events = []

    processed = 0
    for idx, row in df[featuresXX].iterrows():
        X_one = row.to_frame().T
        is_maint_flag = bool(df.loc[idx, "is_maintenance"])
        X_det = X_one.copy()
        X_det["is_maintenance"] = is_maint_flag

        y_now = float(df.loc[idx, target_col]) if target_col in df.columns else None
        res = detector.step_online(X_det[det_features], y_now=y_now)
        label, color = res["label"], res["color"]

        # baseline forecast
        y_pred = float(model.predict(X_one)[0])

        # --- dynamic threshold (ใช้ policy ถ้ามี) ---
        dyn_thr = threshold_policy.update_and_get(y_pred=y_pred, y_true=y_now) if threshold_policy else yield_threshold
        val_for_thr = y_now if y_now is not None else y_pred

        # --- gating ---
        alert_maintenance = (label == "maintenance")
        alert_ml = (label == "ml_anomaly")
        alert_soft = (label == "soft_anomaly") and (dyn_thr is None or val_for_thr < dyn_thr)
        alert_any = alert_maintenance or alert_ml or alert_soft

        if alert_any:
            if alert_maintenance:
                print(f"\n=== ALERT (maintenance) @ {idx} ===  pred={y_pred:.4f}  thr={dyn_thr}")
                print("Maintenance alert: skip GA")
                result = {
                    "label": label, "color": color, "alert_reasons": ["Maintenance (rule)"],
                    "ga": {"suggestion": {}, "best_score": y_pred}, "top_shap": []
                }
            else:
                # เขียว/แดง → รัน pipeline (GA/SHAP)
                print(f"\n=== ALERT @ {idx} === [{label}]  pred={y_pred:.4f}  thr={dyn_thr}")
                result = ethanol_pipeline_full_with_ga(
                    features=X_one,
                    forecaster=model,
                    shap_explainer=shap_explainer,
                    detector=detector,
                    y_true=y_now,
                    ga_optimizer=ga_optimizer,
                    yield_threshold=dyn_thr,          # ส่ง threshold ที่ใช้จริง
                    feature_names=featuresXX,
                    df_train=df_train,
                    controllable_features=controllable_features,
                    run_ga_labels=("soft_anomaly","ml_anomaly")
                )

                # แสดงผลย่อ
                reasons = result.get("alert_reasons") or [label]
                print("Alert reasons:", reasons)
                top_shap = result.get("top_shap", [])
                if top_shap:
                    print("Top SHAP features:", [(k, round(v, 4)) for k, v in top_shap])
                ga = result.get("ga", {})
                if ga and ga.get("suggestion"):
                    print("GA suggested features:", ga.get("suggestion", {}))
                    print("GA predicted yield:", ga.get("best_score"))
                else:
                    print("GA: (no suggestion)")
        else:
            if label == "soft_anomaly":
                print(f"{idx}: Soft anomaly but value≥thr ({val_for_thr:.4f} ≥ {dyn_thr}); no alert.")
            else:
                print(f"{idx}: Normal ({label}), predicted={y_pred:.4f}")
            result = {"label": label, "color": color, "ga": {"suggestion": {}, "best_score": y_pred}, "top_shap": []}

        # --- log event (สำหรับ dashboard/CSV/SSE) ---
        if log_events and (alert_any or label != "normal"):
            event = {
                "ts": idx,                          # timestamp/index
                "label": result.get("label", label),
                "color": result.get("color", color),
                "predicted": y_pred,
                "y_true": y_now,
                "threshold_used": dyn_thr,
                "value_for_threshold": val_for_thr,
                "z": res.get("z"),
                "anomaly_score": res.get("ml_score"),
                "reasons": result.get("alert_reasons", []),
                "ga_suggestion": result.get("ga", {}).get("suggestion", {}),
                "ga_best_score": result.get("ga", {}).get("best_score", None),
                "top_shap": result.get("top_shap", [])
            }
            events.append(event)

            # ส่ง SSE ไปหน้าเว็บ (เฉพาะตอน alert)
            if sse_queue is not None and (alert_any):
                try:
                    sse_queue.put(event, timeout=0.01)
                except Exception:
                    pass

        processed += 1
        if max_rows is not None and processed >= max_rows:
            break

    # --- สรุป/บันทึก ---
    if log_events and len(events) > 0:
        events_df = pd.DataFrame(events)
        if save_events_csv:
            events_df.to_csv(save_events_csv, index=False)
        if return_events_df:
            return events_df
    return None
