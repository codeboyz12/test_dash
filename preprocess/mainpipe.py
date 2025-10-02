import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from preprocess.ga import GeneticAlgorithm
from agent.rag_agent import ask_alert


def ethanol_pipeline_full_with_ga(
    features: pd.DataFrame,                 # 1 แถว
    forecaster,                             # โมเดล forecast (เช่น XGB, RF)
    shap_explainer=None,
    detector=None,                          # เช่น combined_model.AnomalyDetector 
    y_true: float | None = None,            # สำหรับ online/replay
    ga_optimizer=None,
    yield_threshold: float | None = None,   # dynamic threshold (เช่น lo band)
    feature_names: list | None = None,
    df_train: pd.DataFrame | None = None,
    controllable_features: list | None = None,
    controllable_bounds: dict | None = None,
    step_ahead: int = 4,
    run_ga_labels: tuple = ("soft_anomaly","ml_anomaly")
):
    # ---- prep ----
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame([features])
    if feature_names is None:
        feature_names = list(features.columns)

    # ---- detector wrapper (รองรับ step / step_online / predict-only) ----
    def _detect_with_fallback(detector, X_row: pd.DataFrame):
        y_pred = float(forecaster.predict(X_row[feature_names])[0])

        # maintenance flag จากฟีเจอร์ (ถ้ามี)
        maint_flag = False
        if "is_maintenance" in X_row.columns:
            try:
                maint_flag = bool(int(X_row["is_maintenance"].iloc[0]) == 1)
            except Exception:
                maint_flag = bool(X_row["is_maintenance"].iloc[0])

        # 1) multi-stage ที่มี .step(...)
        if hasattr(detector, "step") and callable(getattr(detector, "step")):
            out = detector.step(X_row, y_true=y_true)
            out.setdefault("y_pred", y_pred)
            out.setdefault("z", None)
            out.setdefault("ml_score", None)
            return out

        
        if hasattr(detector, "step_online") and callable(getattr(detector, "step_online")):
            out = detector.step_online(X_row, y_now=y_true)
            out.setdefault("y_pred", y_pred)
            out.setdefault("z", None)
            out.setdefault("ml_score", None)
            out.setdefault("color", "red" if out.get("label") == "ml_anomaly" else "blue")
            return out

        # 3) fallback: ใช้ predict / score ของ ML anomaly โดยตรง
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

    # สร้าง detector เองถ้าไม่ส่งมา (จะเข้าสู่ fallback ทันที)
    if detector is None:
        detector = object()

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

    out: Dict[str, Any] = {
        "label": label,                         # normal/maintenance/soft_anomaly/ml_anomaly
        "color": color,                         # blue/orange/green/red
        "alert": is_alert,                      # True สำหรับ green/red (+ เงื่อนไขเสริม)
        "alert_reasons": alert_reasons,
        "predicted_yield": pred,
        "is_anomaly": label in ["soft_anomaly", "ml_anomaly"],
        "anomaly_score": (abs(det["z"]) if det.get("z") is not None else det.get("ml_score")),
    }

    # ---------- SHAP (เฉพาะตอน alert) + Fallback ----------
    shap_dict: Dict[str, float] = {}
    if is_alert:
        if shap_explainer is not None:
            try:
                sv = shap_explainer.shap_values(features[feature_names])
                if isinstance(sv, list) or (hasattr(sv, "ndim") and sv.ndim > 2):
                    sv = sv[0]
                shap_vals = np.array(sv).reshape(-1)
                shap_dict = dict(zip(feature_names, shap_vals))
            except Exception:
                shap_dict = {}
        # Fallback: ใช้ feature_importances_ เป็น proxy ของความสำคัญ
        if not shap_dict and hasattr(forecaster, "feature_importances_"):
            try:
                imp = np.asarray(getattr(forecaster, "feature_importances_")).ravel()
                if imp.size == len(feature_names):
                    shap_dict = dict(zip(feature_names, imp))
            except Exception:
                pass

    out["top_shap"] = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10] if shap_dict else []
    if shap_dict:
        out["shap_values"] = shap_dict

    # ---------- GA (รันเมื่อ label อยู่ใน run_ga_labels) ----------
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
                try:
                    from ga import GeneticAlgorithm
                    ga = GeneticAlgorithm(
                        cont_bounds=bounds_list, n_binary=0, predict_fn=_predict_fn,
                        pop_size=24, generations=30, crossover_rate=0.7,
                        mutation_rate=0.2, elite_frac=0.15, maximize=True
                    )
                    ga_res = ga.evolve(verbose=False)
                except Exception:
                    # fallback: random search เบาๆ
                    best_score, best = -np.inf, None
                    for _ in range(80):
                        ind = np.array([np.random.uniform(a,b) for (a,b) in bounds_list], float)
                        sc  = _predict_fn(ind)
                        if sc > best_score:
                            best_score, best = sc, ind
                    ga_res = {"best_individual": best, "best_score": best_score, "history": []}
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




def _random_ga_suggest(
    predict_fn,             
    bounds: List[tuple],     
    n_iter: int = 80,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    best_score = -np.inf
    best_ind = None
    hist = []
    for _ in range(n_iter):
        ind = np.array([rng.uniform(a, b) for (a, b) in bounds], dtype=float)
        score = float(predict_fn(ind))
        hist.append(score)
        if score > best_score:
            best_score = score
            best_ind = ind.copy()
    return {"best_individual": best_ind, "best_score": best_score, "history": hist}

try:
    def _run_ga(predict_fn, bounds):
        ga = GeneticAlgorithm(
            cont_bounds=bounds, n_binary=0, predict_fn=predict_fn,
            pop_size=24, generations=10, crossover_rate=0.7,
            mutation_rate=0.2, elite_frac=0.15, maximize=True
        )
        print("Ga running ...")
        return ga.evolve(verbose=False)
except Exception:
    GeneticAlgorithm = None
    def _run_ga(predict_fn, bounds):
        print("Not GA.........")
        return _random_ga_suggest(predict_fn, bounds, n_iter=100)


def _compute_shap(shap_explainer, X: pd.DataFrame, feature_names: List[str]) -> Dict[str, float]:
    if shap_explainer is None:
        return {}
    try:
        sv = shap_explainer.shap_values(X[feature_names])
        if isinstance(sv, list):
            sv = sv[0]
        vals = np.array(sv).reshape(-1)
        return dict(zip(feature_names, vals))
    except Exception:
        return {}

def _heuristic_narrative(label: str, y_pred: float, low_band: Optional[float],
                         shap_top: List[tuple], ga_sug: Dict[str, float]) -> Dict[str, Any]:
    summary = {
        "maintenance": "อยู่ในช่วง maintenance/ผลผลิตลดลงชั่วคราว",
        "soft_anomaly": "รูปแบบล่าสุดบ่งชี้ความเสี่ยงที่ผลผลิตจะลดลง",
        "ml_anomaly": "โมเดลระบุความผิดปกติที่อาจทำให้ผลผลิตลดลง",
        "normal": "คาดการณ์อยู่ในช่วงปกติ",
    }.get(label, "พบสัญญาณที่ต้องติดตาม")

    actions = []
    ga_prompt = ""
    if ga_sug:
        for k, v in list(ga_sug.items())[:3]:
            actions.append(f"ปรับ {k} → {v:.3f} ตามข้อเสนอ GA")
            ga_prompt += f"ฟีเจอร์ {k} มีค่า GA ที่ {v:.3f}"
    if not actions:
        actions = ["ทวนสอบสัญญาณหน้างาน", "ตรวจสอบข้อจำกัดการเดินเครื่อง", "ยืนยันข้อมูลขาเข้า"]
        ga_prompt = "ไม่มีค่า GA"

    caveats = ""

    actions = ""
    if low_band is not None and y_pred is not None:
        actions += f"ตรวจสอบว่า ŷ={y_pred:.4f} ไม่ต่ำกว่า band ล่าง ({low_band:.4f}) ต่อเนื่องหลายช่วง"
    if shap_top:
        shap_texts = []
        for feature, shap_val in shap_top:
            shap_texts.append(f"ฟีเจอร์ {feature} มีค่า SHAP ที่ {shap_val:.4f}")
        shap_prompt = " | ".join(shap_texts)
    caveats += "ยืนยัน constraint/ความปลอดภัยกระบวนการก่อนปรับทุกครั้ง"

    final_prompt = f"พิจรณาค่า GA ในแต่ละฟีเจอร์นี้ {ga_prompt}, {caveats} และ SHAP ต่อไปนี้ {shap_prompt} จากข้อมูลช่วยเขียนคำแนะนำสำหรับข้อมูลต่อไปนี้ออกมา"
    caveats_advice = ask_alert(final_prompt)

    return {"summary": summary, "actions": actions, "caveats": [caveats_advice]}

def ga_shap_narrative(
    X_one: pd.DataFrame,
    model,
    feature_names: List[str],
    *,
    y_true: Optional[float] = None,
    yield_threshold: Optional[float] = None,
    shap_explainer=None,
    controllable_features: Optional[List[str]] = None,
    controllable_bounds: Optional[Dict[str, tuple]] = None,
) -> Dict[str, Any]:
    """คืน dict ที่มี shap_top, ga, genai (narrative) — รองรับ GA แบบ batch + fallback ของเดิมอัตโนมัติ"""

    shap_vals = _compute_shap(shap_explainer, X_one, feature_names) 
    shap_neg = {k: v for k, v in shap_vals.items() if v < 0}
    shap_top_neg = sorted(shap_neg.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    if 'R421_Temp' in X_one.columns:
        print(True)
    else:
        print(False)

    if controllable_features is None:
        candidates = ['FIC421', 'FIC422', 'R411_Temp', 'R412_Temp', 'R421_Temp',
                      'R422_Temp', 'R423_Temp ', 'R424_Temp', 'R425_Temp', 'R426_Temp', 'R427_Temp ']

        controllable_features = []
        for f in candidates:
            f2 = f.strip()
            if f2 in X_one.columns and f2 not in controllable_features:
                controllable_features.append(f2)

    if controllable_bounds is None:
        base = X_one.iloc[0]
        controllable_bounds = {
            f: (float(base[f]) * 0.85, float(base[f]) * 1.15) for f in controllable_features
        }

    bounds_list = [controllable_bounds[f] for f in controllable_features]

    base_vec = X_one[feature_names].iloc[0].to_numpy(dtype=float)  
    try:
        ctrl_idx = np.array([feature_names.index(f) for f in controllable_features], dtype=int)
    except ValueError as e:
        valid_feats = [f for f in controllable_features if f in feature_names]
        ctrl_idx = np.array([feature_names.index(f) for f in valid_feats], dtype=int)
        controllable_features = valid_feats
        bounds_list = [controllable_bounds[f] for f in controllable_features]

    def batch_predict_fn(X_batch: np.ndarray) -> np.ndarray:
        """
        X_batch: (M, len(controllable_features)) เฉพาะค่าที่ควบคุมได้ ตามลำดับ controllable_features
        return:  (M,)
        """
        if X_batch.ndim == 1:
            X_batch = X_batch.reshape(1, -1)
        M = X_batch.shape[0]
        X_full = np.repeat(base_vec[None, :], M, axis=0)
        X_full[:, ctrl_idx] = X_batch.astype(float, copy=False)
        preds = model.predict(X_full)
        return np.asarray(preds, dtype=float).reshape(-1)

 
    def _predict_fn(ind: np.ndarray) -> float:
        vec = base_vec.copy()
        # เขียนเฉพาะฟีเจอร์ที่ควบคุม (ตามลำดับ controllable_features)
        n_ctrl = len(controllable_features)
        vec[ctrl_idx] = np.asarray(ind[:n_ctrl], dtype=float)
        return float(model.predict(vec.reshape(1, -1))[0])

   
    ga_res = None
    try:
        ga_res = _run_ga(
            _predict_fn,                
            bounds_list,
            batch_predict_fn=batch_predict_fn,
            use_batch=True
        )
    except TypeError:
        try:
            ga_res = _run_ga(_predict_fn, bounds_list)
        except Exception:
            ga_res = None
    except Exception:
        ga_res = None

    ga_block = {"suggestion": {}, "best_score": None}
    if isinstance(ga_res, dict) and "best_individual" in ga_res:
        best = ga_res["best_individual"]
        n_ctrl = len(controllable_features)
        best = np.asarray(best, dtype=float).ravel()[:n_ctrl]
        sug = {f: float(best[i]) for i, f in enumerate(controllable_features)}
        ga_block = {"suggestion": sug, "best_score": float(ga_res.get("best_score", np.nan))}

    y_pred = float(model.predict(X_one[feature_names])[0])
    label_for_text = "soft_anomaly" if (yield_threshold is not None and y_pred < yield_threshold) else "normal"
    genai = _heuristic_narrative(label_for_text, y_pred, yield_threshold, shap_top_neg, ga_block["suggestion"])

    return {"shap_top": shap_top_neg, "ga": ga_block, "gen_ai": genai}




# def ga_shap_narrative(
#     X_one: pd.DataFrame,
#     model,
#     feature_names: List[str],
#     *,
#     y_true: Optional[float] = None,
#     yield_threshold: Optional[float] = None,
#     shap_explainer=None,
#     controllable_features: Optional[List[str]] = None,
#     controllable_bounds: Optional[Dict[str, tuple]] = None,
# ) -> Dict[str, Any]:
#     """คืน dict ที่มี shap_top, ga, genai (narrative)"""
#     #SHAP
#     shap_vals = _compute_shap(shap_explainer, X_one, feature_names)
#     shap_neg = {k: v for k, v in shap_vals.items() if v < 0}
#     shap_top_neg = sorted(shap_neg.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
#     # shap_top = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

#     if 'R421_Temp' in X_one.columns:
#         print(True)
#     else:
#         print(False)
#     #GA
#     if controllable_features is None:
#         controllable_features = [f for f in ['FIC421', 'FIC422','R411_Temp', 'R412_Temp', 'R421_Temp', 'R422_Temp', 'R423_Temp ', 'R424_Temp', 'R425_Temp', 'R426_Temp', 'R427_Temp '] if f in X_one.columns]
#     if controllable_bounds is None:
#         base = X_one.iloc[0]
#         controllable_bounds = {f: (float(base[f])*0.85, float(base[f])*1.15) for f in controllable_features}
#     bounds_list = [controllable_bounds[f] for f in controllable_features]

#     def _predict_fn(ind):
#         row = X_one.copy()
#         i0 = row.index[0]
#         for i, f in enumerate(controllable_features):
#             row.at[i0, f] = float(ind[i])
#         return float(model.predict(row[feature_names])[0])

#     try:
#         ga_res = _run_ga(_predict_fn, bounds_list)
#     except Exception:
#         ga_res = None

#     ga_block = {"suggestion": {}, "best_score": None}
#     if isinstance(ga_res, dict) and "best_individual" in ga_res:
#         best = ga_res["best_individual"]
#         sug = {f: float(best[i]) for i, f in enumerate(controllable_features)}
#         ga_block = {"suggestion": sug, "best_score": float(ga_res.get("best_score", np.nan))}

#     #Narrative
#     y_pred = float(model.predict(X_one[feature_names])[0])
#     label_for_text = "soft_anomaly" if (yield_threshold is not None and y_pred < yield_threshold) else "normal"
#     genai = _heuristic_narrative(label_for_text, y_pred, yield_threshold, shap_top_neg, ga_block["suggestion"])

#     return {"shap_top": shap_top_neg, "ga": ga_block, "gen_ai": genai}

