# def stream_loop(df: pd.DataFrame, feature_list, model, shap_explainer, detector: AnomalyDetector,
#                 target_col="CF-Total-Today", chunk_emit=5, step_delay=0.25):
#     rows = list(df[feature_list].iterrows())
#     i, n = 0, len(rows)
#     while i < n:
#         stream_payloads = []
#         alert_cards     = []
#         for _ in range(chunk_emit):
#             if i >= n: break
#             idx, row = rows[i]
#             X_one  = row.to_frame().T
#             y_true = float(df.loc[idx, target_col])

#             # เรียก pipeline เดียวพอ: จะ detect + GA + SHAP + Narrative ครบ
#             result = ethanol_pipeline_full_with_ga(
#                 features=X_one,
#                 forecaster=model,
#                 shap_explainer=shap_explainer,    # None ก็ได้: fallback จะทำงาน
#                 detector=detector,                 # <<< ใช้คลาสของคุณ
#                 y_true=y_true,
#                 ga_optimizer=None,
#                 yield_threshold=None,              # จะใช้เฉพาะถ้าคุณมี dyn. threshold
#                 feature_names=feature_list,
#                 df_train=None,
#                 controllable_features=["Inprocess_Preferment","Inprocess_Ferment","Inprocess_Total","CFconvert-Total-Today"],
#                 controllable_bounds=None,
#                 run_ga_labels=("soft_anomaly","ml_anomaly")   # <<< soft ก็รัน GA
#             )

#             ts = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
#             stream_payloads.append({
#                 "ts": ts,
#                 "y_true": y_true,
#                 "y_pred": result["predicted_yield"],
#                 "y_lo": None,
#                 "y_hi": None,
#                 "label": result["label"],
#                 "color": result["color"],
#                 "alert": result["alert"],
#                 "reasons": result.get("alert_reasons", []),
#                 "z": result.get("anomaly_score"),
#             })

#             if result["alert"]:
#                 alert_cards.append(_build_alert_card_from_result(ts, y_true, result))

#             i += 1

#         if stream_payloads:
#             socketio.emit(EVENT_STREAM, stream_payloads)
#         if alert_cards:
#             socketio.emit(EVENT_ALERT,  alert_cards)

#         socketio.sleep(step_delay)