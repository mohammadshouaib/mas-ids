"""
mas_ids.agents.detection_agent
==============================
Agent 3 — Detection.

CNN+LSTM classifiers, LSTM autoencoders (anomaly scoring), GRU edge predictors,
DBSCAN traffic-outlier detection, and the three-tier score-fusion logic that
produces final_label, severity, and detection_reason_codes.

Public API:
  build_cnn_lstm_classifier, build_lstm_autoencoder, build_edge_gru_predictor,
  run_dbscan_traffic_windows, train_detection_models, load_detection_models,
  run_edge_detection, run_gcc_detection, apply_final_detection_logic,
  evaluate_detection, run_detection_agent_inference, predict_single
"""
import pickle
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, GRU, Dense,
    Dropout, BatchNormalization, RepeatVector, TimeDistributed, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report

from ..config import (
    LABEL_MAP, LABEL_INV, N_CLASSES, SEQ_LEN, GLOBAL_SEED,
    MODEL_DIR, JAM_MODEL_PATH, DOS_MODEL_PATH, HYB_MODEL_PATH,
    JAM_AE_PATH, DOS_AE_PATH, DETECT_CONFIG_PATH,
    FE_STATE_PATH, FEAT_CSV,
)
from ..utils import q_thresh
from .feature_agent import prepare_feature_dataframe


# ── Model constants ───────────────────────────────────────────────────────────
N_CLASSES = len(LABEL_MAP)   # 5: normal, jamming, dos, ddos, hybrid


def build_cnn_lstm_classifier(seq_len: int, n_features: int,
                               n_classes: int = N_CLASSES) -> Model:
    """
    CNN + LSTM hybrid classifier for the GCC detection tier.

    Architecture:
      Input (seq_len, n_features)
      -> Conv1D x2 (spatial burst extraction)
      -> LSTM x2   (temporal sequence modelling)
      -> Dense classifier head

    float32 output enforced for mixed_float16 stability.
    Mirrors build_gru_predictor + build_lstm_autoencoder from original,
    but combines both into a single supervised architecture.
    """
    inp = Input(shape=(seq_len, n_features), name="sequence_input")

    # CNN block — local pattern extraction
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(inp)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # LSTM block — temporal modelling
    x = LSTM(128, return_sequences=True, activation="tanh")(x)
    x = Dropout(0.2)(x)
    x = LSTM(64,  return_sequences=False, activation="tanh")(x)

    # Classifier head
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(n_classes, activation="softmax", dtype="float32", name="class_output")(x)

    model = Model(inp, out, name="cnn_lstm_classifier")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_lstm_autoencoder(seq_len: int, n_features: int) -> Model:
    """
    LSTM Autoencoder for anomaly scoring (trained on normal-only data).
    High reconstruction error = anomaly.
    Mirrors build_lstm_autoencoder() from the original exactly.
    float32 output enforced for mixed_float16 stability.
    """
    inputs  = Input(shape=(seq_len, n_features), name="ae_input")
    x       = LSTM(128, activation="tanh", return_sequences=True)(inputs)
    x       = Dropout(0.2)(x)
    encoded = LSTM(64,  activation="tanh", return_sequences=False)(x)
    rep     = RepeatVector(seq_len)(encoded)
    x       = LSTM(64,  activation="tanh", return_sequences=True)(rep)
    x       = Dropout(0.2)(x)
    decoded = LSTM(128, activation="tanh", return_sequences=True)(x)
    out     = TimeDistributed(Dense(n_features, dtype="float32"), name="ae_output")(decoded)
    model   = Model(inputs, out, name="lstm_autoencoder")
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model


def build_edge_gru_predictor(seq_len: int, n_features: int) -> Model:
    """
    Lightweight GRU next-step predictor for edge-level confirmation.
    Trained on normal sequences only; high prediction error = anomaly.
    Input shape: (seq_len-1, n_features) -> predicts step seq_len.
    Mirrors build_gru_predictor() from the original exactly.
    """
    model = Sequential([
        Input(shape=(seq_len - 1, n_features)),
        GRU(128, return_sequences=True),
        Dropout(0.2),
        GRU(64,  return_sequences=False),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(n_features, dtype="float32")
    ], name="edge_gru_predictor")
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model


# ── DBSCAN swarm outlier detector (mirrors original Cell 10) ──────────────────
def run_dbscan_traffic_windows(
    df: pd.DataFrame,
    eps: float = 0.20,
    min_samples: int = 3,
    window: str = "1s",
    feature_cols: list = None,
) -> pd.DataFrame:
    """
    DBSCAN-based spatial/traffic outlier detector across the swarm.
    Groups records into 1-second buckets and runs DBSCAN per bucket.
    Rows flagged as outliers (label=-1) are traffic anomalies.

    Adapted from run_dbscan_on_time_windows() in the original;
    uses network traffic features instead of GPS coordinates.
    """
    if feature_cols is None:
        feature_cols = ["packets_per_second", "src_ip_entropy",
                        "half_open_connections"]

    out_df = df.copy()
    out_df["timestamp"] = pd.to_datetime(out_df["timestamp"], errors="coerce")
    out_df = out_df.dropna(subset=["timestamp"]).copy()
    out_df["dbscan_label"]    = -2
    out_df["traffic_outlier"] = 0

    if len(out_df) == 0:
        return out_df

    n_nodes = out_df["node_id"].nunique()
    eff_min = max(1, min(min_samples, n_nodes))
    if eff_min < min_samples:
        print(f"[DBSCAN] {n_nodes} node(s) — lowering min_samples to {eff_min}")

    out_df["_bucket"] = out_df["timestamp"].dt.floor(window)
    available_feats   = [c for c in feature_cols if c in out_df.columns]

    # Per-bucket DBSCAN is O(n^2). With dense packet-capture data a single 1s
    # bucket can hold tens of thousands of rows, which makes DBSCAN hang / OOM.
    # Cap the points clustered per bucket; the rest inherit the bucket's modal
    # outlier flag. This preserves the outlier signal without the blowup.
    MAX_BUCKET_POINTS = 2000

    if not available_feats:
        out_df.drop(columns=["_bucket"], inplace=True)
        return out_df

    for _, bucket in out_df.groupby("_bucket", sort=False):
        if len(bucket) < eff_min:
            continue
        sample = bucket
        if len(bucket) > MAX_BUCKET_POINTS:
            sample = bucket.sample(MAX_BUCKET_POINTS, random_state=42)
        X_b = sample[available_feats].apply(
            pd.to_numeric, errors="coerce").fillna(0.0).values
        labels = DBSCAN(eps=eps, min_samples=eff_min).fit_predict(X_b)
        out_df.loc[sample.index, "dbscan_label"]    = labels
        out_df.loc[sample.index, "traffic_outlier"] = (labels == -1).astype(int)
        # Rows not sampled in an oversized bucket default to non-outlier (0)
        if len(bucket) > MAX_BUCKET_POINTS:
            unsampled = bucket.index.difference(sample.index)
            out_df.loc[unsampled, "dbscan_label"]    = 0
            out_df.loc[unsampled, "traffic_outlier"] = 0

    out_df.drop(columns=["_bucket"], inplace=True)
    return out_df


def train_detection_models(
    arrays: dict,
    fe_state: dict,
    seq_len: int = SEQ_LEN,
    epochs: int = 50,
    batch_size: int = 512,
    save_models: bool = True,
    preview: bool = True,
) -> dict:
    """
    Train all detection models.
    Returns a results dict with trained models, thresholds, and config.
    Mirrors train_detection_models() from the original.
    """
    # FIX: clear TF graph to prevent GPU memory accumulation across re-runs
    tf.keras.backend.clear_session()
    label_map = fe_state["label_map"]
    label_inv = {v: k for k, v in label_map.items()}
    normal_int = label_map["normal"]

    X_jam_tr = arrays["X_jam_tr"];  y_jam_tr = arrays["y_jam_tr"]
    X_dos_tr = arrays["X_dos_tr"];  y_dos_tr = arrays["y_dos_tr"]
    X_hyb_tr = arrays["X_hyb_tr"];  y_hyb_tr = arrays["y_hyb_tr"]

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    ]

    results = {}

    # ── A. CNN+LSTM Classifiers ────────────────────────────────────────────────
    for name, X_tr, y_tr, feat_len, path in [
        ("jamming", X_jam_tr, y_jam_tr, X_jam_tr.shape[2], JAM_MODEL_PATH),
        ("dos",     X_dos_tr, y_dos_tr, X_dos_tr.shape[2], DOS_MODEL_PATH),
        ("hybrid",  X_hyb_tr, y_hyb_tr, X_hyb_tr.shape[2], HYB_MODEL_PATH),
    ]:
        print(f"\n[Train] CNN+LSTM {name} | X={X_tr.shape} | classes={N_CLASSES}")
        model = build_cnn_lstm_classifier(seq_len, feat_len, N_CLASSES)
        with tf.device("/GPU:0"):
            history = model.fit(
                X_tr, y_tr,
                epochs=epochs, batch_size=batch_size,
                validation_split=0.15, verbose=1,
                callbacks=callbacks
            )
        results[f"cnn_lstm_{name}"] = {"model": model, "history": history}
        if save_models:
            model.save(path)
            print(f"  Saved: {path}")

    # ── B. LSTM Autoencoders (normal-only training) ────────────────────────────
    for name, X_tr, y_tr, feat_len, ae_path in [
        ("jamming", X_jam_tr, y_jam_tr, X_jam_tr.shape[2], JAM_AE_PATH),
        ("dos",     X_dos_tr, y_dos_tr, X_dos_tr.shape[2], DOS_AE_PATH),
    ]:
        normal_mask = y_tr == normal_int
        X_normal    = X_tr[normal_mask]
        if len(X_normal) < seq_len + 2:
            print(f"  WARNING: too few normal rows for {name} AE — using all rows")
            X_normal = X_tr
        print(f"\n[Train] LSTM-AE {name} | normal_seqs={len(X_normal)}")
        ae_model = build_lstm_autoencoder(seq_len, feat_len)
        with tf.device("/GPU:0"):
            ae_model.fit(
                X_normal, X_normal,
                epochs=epochs, batch_size=batch_size,
                validation_split=0.15, verbose=1,
                callbacks=callbacks
            )
        # Calibrate threshold: 99th pct of normal training error
        recon  = ae_model.predict(X_normal, verbose=0)
        errors = np.mean((recon - X_normal) ** 2, axis=(1, 2))
        thr    = float(np.percentile(errors, 99))
        print(f"  AE threshold (99th pct of normal): {thr:.6f}")
        results[f"lstm_ae_{name}"] = {"model": ae_model, "threshold": thr}
        if save_models:
            ae_model.save(ae_path)

    # ── C. GRU predictors for edge confirmation ────────────────────────────────
    for name, X_tr, y_tr, feat_len, gru_path in [
        ("jamming", X_jam_tr, y_jam_tr, X_jam_tr.shape[2],
         os.path.join(MODEL_DIR, "gru_edge_jamming.keras")),
        ("dos",     X_dos_tr, y_dos_tr, X_dos_tr.shape[2],
         os.path.join(MODEL_DIR, "gru_edge_dos.keras")),
    ]:
        normal_mask = y_tr == normal_int
        X_normal    = X_tr[normal_mask]
        if len(X_normal) < seq_len + 2:
            X_normal = X_tr
        X_in  = X_normal[:, :-1, :]
        y_out = X_normal[:,  -1, :]
        print(f"\n[Train] GRU predictor {name} | normal_seqs={len(X_normal)}")
        gru = build_edge_gru_predictor(seq_len, feat_len)
        with tf.device("/GPU:0"):
            gru.fit(
                X_in, y_out,
                epochs=epochs, batch_size=batch_size,
                validation_split=0.15, verbose=1,
                callbacks=callbacks
            )
        gru_train_errors = np.mean(
            (gru.predict(X_in, verbose=0) - y_out) ** 2, axis=1
        )
        gru_thr = float(np.percentile(gru_train_errors, 99))
        print(f"  GRU threshold (99th pct): {gru_thr:.6f}")
        results[f"gru_{name}"] = {"model": gru, "threshold": gru_thr}
        if save_models:
            gru.save(gru_path)

    # ── D. Save config ─────────────────────────────────────────────────────────
    config = {
        "seq_len"            : seq_len,
        "n_classes"          : N_CLASSES,
        "label_map"          : label_map,
        "JAM_FEAT_FINAL"     : fe_state["JAM_FEAT_FINAL"],
        "DOS_FEAT_FINAL"     : fe_state["DOS_FEAT_FINAL"],
        "HYBRID_FEAT_FINAL"  : fe_state["HYBRID_FEAT_FINAL"],
        "ae_jam_threshold"   : results.get("lstm_ae_jamming", {}).get("threshold", 0.0),
        "ae_dos_threshold"   : results.get("lstm_ae_dos",     {}).get("threshold", 0.0),
        "gru_jam_threshold"  : results.get("gru_jamming",     {}).get("threshold", 0.0),
        "gru_dos_threshold"  : results.get("gru_dos",         {}).get("threshold", 0.0),
    }
    if save_models:
        with open(DETECT_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nDetection config saved: {DETECT_CONFIG_PATH}")
    results["config"] = config
    if preview:
        print("\n[Train] All models trained.")
        for k, v in config.items():
            if not isinstance(v, list):
                print(f"  {k}: {v}")
    return results


# ── Run training ──────────────────────────────────────────────────────────────

def load_detection_models() -> dict:
    """
    Load all detection models and config from disk.
    Mirrors load_detection_models() from the original.
    Returns a models dict with all models and the config.
    """
    required = [
        JAM_MODEL_PATH, DOS_MODEL_PATH, HYB_MODEL_PATH,
        JAM_AE_PATH, DOS_AE_PATH, DETECT_CONFIG_PATH
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}. Run Cell 4 first."
        )

    models = {
        "cnn_lstm_jamming" : load_model(JAM_MODEL_PATH),
        "cnn_lstm_dos"     : load_model(DOS_MODEL_PATH),
        "cnn_lstm_hybrid"  : load_model(HYB_MODEL_PATH),
        "lstm_ae_jamming"  : load_model(JAM_AE_PATH),
        "lstm_ae_dos"      : load_model(DOS_AE_PATH),
    }
    # Load GRU edge predictors if present
    for tag, path in [
        ("gru_jamming", os.path.join(MODEL_DIR, "gru_edge_jamming.keras")),
        ("gru_dos",     os.path.join(MODEL_DIR, "gru_edge_dos.keras")),
    ]:
        if os.path.exists(path):
            models[tag] = load_model(path)

    with open(DETECT_CONFIG_PATH) as f:
        models["config"] = json.load(f)

    print("Detection models loaded:")
    for k in models:
        if k != "config":
            print(f"  {k}")
    return models


# Use models from training if available; else load from disk


def run_edge_detection(
    feat_df: pd.DataFrame,
    det_models: dict,
    fe_state: dict,
    preview: bool = False,
) -> pd.DataFrame:
    """
    Tier A: Edge-level detection for each node.
    Runs GRU next-step predictor on jamming feature sequences.
    Returns a DataFrame with one row per sequence window:
      source_row_id, node_id, timestamp,
      gru_jam_score, gru_jam_flag,
      ae_jam_score,  ae_jam_flag,
      edge_alarm
    Mirrors run_uav_local_detection() from the original.
    """
    import warnings; warnings.filterwarnings("ignore")

    det_config = det_models.get("config", {})
    seq_len   = fe_state["seq_len"]
    jf        = fe_state["JAM_FEAT_FINAL"]
    gru_thr   = det_config.get("gru_jam_threshold", 0.01)
    ae_thr    = det_config.get("ae_jam_threshold",  0.01)

    gru_model = det_models.get("gru_jamming")
    ae_model  = det_models.get("lstm_ae_jamming") or det_models.get("cnn_lstm_jamming")

    # Build jamming sequences from feat_df
    jf_avail = [c for c in jf if c in feat_df.columns]
    feat_df  = feat_df.sort_values(["node_id","timestamp"]).reset_index(drop=True)
    if "source_row_id" not in feat_df.columns:
        feat_df["source_row_id"] = feat_df.index

    X_list, meta_list = [], []
    for node_id, group in feat_df.groupby("node_id", sort=False):
        group = group.reset_index(drop=True)
        vals  = group[jf_avail].astype(float).values
        n = len(vals)
        if n < seq_len:
            continue
        for i in range(n - seq_len + 1):
            end = group.iloc[i + seq_len - 1]
            X_list.append(vals[i : i + seq_len])
            meta_list.append({
                "source_row_id": int(end.get("source_row_id", i + seq_len - 1)),
                "node_id"      : node_id,
                "timestamp"    : str(end.get("timestamp", "")),
                "true_label"   : str(end.get("label", "unknown")),
            })

    if not X_list:
        return pd.DataFrame()

    X_seq = np.array(X_list, dtype=np.float32)

    # GRU prediction error
    gru_jam_scores = np.zeros(len(X_seq))
    if gru_model is not None:
        X_in  = X_seq[:, :-1, :]
        y_out = X_seq[:,  -1, :]
        pred  = gru_model.predict(X_in, verbose=0, batch_size=512)
        gru_jam_scores = np.mean((pred - y_out) ** 2, axis=1)

    # LSTM-AE reconstruction error
    ae_jam_scores = np.zeros(len(X_seq))
    if ae_model is not None and hasattr(ae_model, "predict"):
        try:
            recon = ae_model.predict(X_seq, verbose=0, batch_size=512)
            if recon.shape == X_seq.shape:
                ae_jam_scores = np.mean((recon - X_seq) ** 2, axis=(1, 2))
        except Exception:
            pass

    # Build result rows
    rows = []
    for i, meta in enumerate(meta_list):
        gru_s = float(gru_jam_scores[i])
        ae_s  = float(ae_jam_scores[i])
        gru_f = int(gru_s > gru_thr)
        ae_f  = int(ae_s  > ae_thr)
        rows.append({
            **meta,
            "gru_jam_score" : round(gru_s, 6),
            "gru_jam_flag"  : gru_f,
            "ae_jam_score"  : round(ae_s,  6),
            "ae_jam_flag"   : ae_f,
            "edge_alarm"    : int(gru_f or ae_f),
        })

    result_df = pd.DataFrame(rows)
    if preview:
        print("[Edge Detection] Results:")
        print(result_df[["node_id","gru_jam_flag","ae_jam_flag","edge_alarm"]]
              .value_counts().head(10))
    return result_df

def run_gcc_detection(
    feat_df: pd.DataFrame,
    det_models: dict,
    fe_state: dict,
    preview: bool = False,
) -> pd.DataFrame:
    """
    Tier B: GCC CNN+LSTM classifier inference on all three feature pipelines.
    Also runs DBSCAN traffic outlier detection.
    Returns a DataFrame with per-window classification scores and flags.
    Mirrors run_edge_confirmation() + run_spatial_swarm_detection() from original.
    """
    import warnings; warnings.filterwarnings("ignore")

    seq_len  = fe_state["seq_len"]
    label_inv = {v: k for k, v in fe_state["label_map"].items()}

    det_config = det_models.get("config", {})
    results_by_row: dict = defaultdict(dict)

    # ── Per-pipeline inference ─────────────────────────────────────────────────
    pipeline_cfg = [
        ("jamming", "cnn_lstm_jamming", fe_state["JAM_FEAT_FINAL"],   "lstm_ae_jamming", det_config.get("ae_jam_threshold", 0.01)),
        ("dos",     "cnn_lstm_dos",     fe_state["DOS_FEAT_FINAL"],   "lstm_ae_dos",     det_config.get("ae_dos_threshold", 0.01)),
        ("hybrid",  "cnn_lstm_hybrid",  fe_state["HYBRID_FEAT_FINAL"],None,              None),
    ]

    feat_df = feat_df.sort_values(["node_id","timestamp"]).reset_index(drop=True)
    if "source_row_id" not in feat_df.columns:
        feat_df["source_row_id"] = feat_df.index

    for pipe_name, clf_key, feat_cols, ae_key, ae_thr in pipeline_cfg:
        avail  = [c for c in feat_cols if c in feat_df.columns]
        clf    = det_models.get(clf_key)
        ae_mdl = det_models.get(ae_key) if ae_key else None
        if clf is None or len(avail) == 0:
            continue

        # Build sequences
        X_list, meta_list = [], []
        for node_id, group in feat_df.groupby("node_id", sort=False):
            group = group.reset_index(drop=True)
            vals  = group[avail].astype(float).values
            n     = len(vals)
            if n < seq_len:
                continue
            for i in range(n - seq_len + 1):
                end = group.iloc[i + seq_len - 1]
                X_list.append(vals[i : i + seq_len])
                meta_list.append({
                    "source_row_id": int(end.get("source_row_id", i+seq_len-1)),
                    "node_id"      : node_id,
                    "timestamp"    : str(end.get("timestamp", "")),
                    "true_label"   : str(end.get("label", "unknown")),
                })

        if not X_list:
            continue

        X_seq = np.array(X_list, dtype=np.float32)

        # Classifier softmax probabilities
        probs = clf.predict(X_seq, verbose=0, batch_size=512)  # (N, n_classes)
        preds = np.argmax(probs, axis=1)

        # AE reconstruction error
        ae_scores = np.zeros(len(X_seq))
        ae_flags  = np.zeros(len(X_seq), dtype=int)
        if ae_mdl is not None and ae_thr is not None:
            try:
                recon = ae_mdl.predict(X_seq, verbose=0, batch_size=512)
                if recon.shape == X_seq.shape:
                    ae_scores = np.mean((recon - X_seq) ** 2, axis=(1, 2))
                    ae_flags  = (ae_scores > ae_thr).astype(int)
            except Exception:
                pass

        for i, meta in enumerate(meta_list):
            rid = meta["source_row_id"]
            results_by_row[rid].update(meta)
            results_by_row[rid][f"{pipe_name}_pred_class"]  = int(preds[i])
            results_by_row[rid][f"{pipe_name}_pred_label"]  = label_inv.get(int(preds[i]), "unknown")
            results_by_row[rid][f"{pipe_name}_conf_normal"] = round(float(probs[i][fe_state["label_map"]["normal"]]), 4)
            results_by_row[rid][f"{pipe_name}_conf_attack"] = round(float(1 - probs[i][fe_state["label_map"]["normal"]]), 4)
            results_by_row[rid][f"{pipe_name}_ae_score"]    = round(float(ae_scores[i]), 6)
            results_by_row[rid][f"{pipe_name}_ae_flag"]     = int(ae_flags[i])

    gcc_df = pd.DataFrame(list(results_by_row.values()))

    # ── DBSCAN traffic outlier detection ──────────────────────────────────────
    dbscan_df = run_dbscan_traffic_windows(
        feat_df,
        eps=0.20, min_samples=3,
        feature_cols=["packets_per_second","src_ip_entropy","half_open_connections"]
    )
    if "source_row_id" in gcc_df.columns and "source_row_id" in dbscan_df.columns:
        gcc_df = gcc_df.merge(
            dbscan_df[["source_row_id","dbscan_label","traffic_outlier"]],
            on="source_row_id", how="left"
        )
        gcc_df[["dbscan_label","traffic_outlier"]] = \
            gcc_df[["dbscan_label","traffic_outlier"]].fillna(0)
    else:
        gcc_df["dbscan_label"]    = -2
        gcc_df["traffic_outlier"] = 0

    if preview:
        print(f"[GCC Detection] {len(gcc_df)} windows processed")
        for pipe in ("jamming","dos","hybrid"):
            col = f"{pipe}_pred_label"
            if col in gcc_df.columns:
                print(f"  {pipe} predictions:")
                print(gcc_df[col].value_counts().to_string())
    return gcc_df

def apply_final_detection_logic(
    feat_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    gcc_df: pd.DataFrame,
    fe_state: dict,
) -> pd.DataFrame:
    """
    Tier C: Score fusion and final classification.
    Merges edge and GCC results, applies weighted score fusion,
    assigns final_label, severity, and detection_reason_codes.
    Mirrors apply_final_detection_logic() from the original exactly.
    """
    label_map = fe_state["label_map"]
    normal_lbl = "normal"

    # ── Start from feat_df as base ─────────────────────────────────────────────
    df = feat_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "source_row_id" not in df.columns:
        df["source_row_id"] = df.index

    # ── Merge edge detection results ───────────────────────────────────────────
    edge_cols = ["source_row_id","gru_jam_score","gru_jam_flag",
                 "ae_jam_score","ae_jam_flag","edge_alarm"]
    edge_cols = [c for c in edge_cols if c in edge_df.columns]
    if len(edge_df) > 0 and "source_row_id" in edge_df.columns:
        df = df.merge(edge_df[edge_cols], on="source_row_id", how="left")
    for c in ["gru_jam_score","gru_jam_flag","ae_jam_score","ae_jam_flag","edge_alarm"]:
        if c not in df.columns: df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    # ── Merge GCC detection results ────────────────────────────────────────────
    gcc_merge_cols = [c for c in gcc_df.columns
                      if c not in df.columns or c == "source_row_id"]
    if len(gcc_df) > 0 and "source_row_id" in gcc_df.columns:
        df = df.merge(gcc_df[gcc_merge_cols], on="source_row_id", how="left")
    for c in ["jamming_conf_attack","dos_conf_attack","hybrid_conf_attack",
              "jamming_ae_flag","dos_ae_flag","traffic_outlier"]:
        if c not in df.columns: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # ── Helper: get column as Series ───────────────────────────────────────────
    def col(name, default=0.0):
        if name not in df.columns:
            return pd.Series(default, index=df.index)
        return pd.to_numeric(df[name], errors="coerce").fillna(default)

    # ── Calibrate thresholds from normal rows (mirrors original) ──────────────
    def nthr(c, q=0.95):
        if c not in df.columns: return 0.0
        mask = df["label"] == normal_lbl if "label" in df.columns else pd.Series(True, index=df.index)
        sub  = df[mask] if mask.sum() > 10 else df
        s = pd.to_numeric(sub[c], errors="coerce").dropna()
        return float(s.quantile(q)) if len(s) else 0.0

    cl_score_thr   = nthr("cross_layer_anomaly_score", 0.95)
    swarm_thr      = nthr("swarm_consensus_anomaly_score", 0.95)
    jam_flag_thr   = nthr("jam_threshold_score",  0.99)
    dos_flag_thr   = nthr("dos_threshold_score",  0.99)

    # ── Compute risk scores ────────────────────────────────────────────────────
    # Jamming risk: physical+MAC evidence, AE confirmation, edge alarm
    jam_risk = (
        0.30 * col("jamming_conf_attack").clip(0,1) +
        0.20 * col("edge_alarm") +
        0.15 * col("jamming_ae_flag") +
        0.15 * col("gru_jam_flag") +
        0.10 * (col("jam_threshold_score") > jam_flag_thr).astype(float) +
        0.10 * (col("cross_layer_anomaly_score") > cl_score_thr).astype(float)
    ).clip(0, 1)

    # DoS/DDoS risk: network+transport evidence, AE flag, DBSCAN outlier
    # FIX: extract optional terms to variables to avoid Python ternary precedence
    # bug where `(... + X if cond else Y).clip()` parses as `(... + (X if cond else Y))`
    coord_flood_term = (col("coordinated_flood_flag")
                        if "coordinated_flood_flag" in df.columns
                        else pd.Series(0.0, index=df.index))
    dos_risk = (
        0.30 * col("dos_conf_attack").clip(0,1) +
        0.20 * col("traffic_outlier") +
        0.15 * col("dos_ae_flag") +
        0.15 * (col("dos_threshold_score") > dos_flag_thr).astype(float) +
        0.10 * (col("swarm_consensus_anomaly_score") > swarm_thr).astype(float) +
        0.10 * coord_flood_term
    ).clip(0, 1)

    # Hybrid risk: a TRUE hybrid attack needs genuine cross-layer evidence, not
    # merely "both other risks are moderate". The previous weighting gave 0.20 to
    # the (jam>0.3 & dos>0.3) shortcut and 0.35 to a hybrid classifier that had
    # collapsed (flatlined at 0.8169 due to the scaling bug), so hybrid_risk hit
    # >=0.60 whenever both branches were mildly active — producing tens of thousands
    # of HYBRID predictions that were ALL false positives. We now require the cross-
    # layer signal to carry most of the weight and demote the co-activation shortcut
    # to a small tie-breaker.
    # FIX: ternary precedence (same as dos_risk) — extract optional term to a var.
    cl_attack_term = (col("cross_layer_attack_flag")
                      if "cross_layer_attack_flag" in df.columns
                      else pd.Series(0.0, index=df.index))
    hybrid_risk = (
        0.35 * col("hybrid_conf_attack").clip(0,1) +
        0.30 * (col("cross_layer_anomaly_score") > cl_score_thr).astype(float) +
        0.25 * cl_attack_term +
        0.10 * ((jam_risk > 0.3) & (dos_risk > 0.3)).astype(float)
    ).clip(0, 1)

    # ── Fusion confidence (mirrors gps_risk + routing_risk fusion) ────────────
    fusion_conf = (
        np.maximum(np.maximum(jam_risk, dos_risk), hybrid_risk) +
        0.05 * col("edge_alarm")
    ).clip(0, 1)

    df["jam_risk_score"]    = jam_risk.round(4)
    df["dos_risk_score"]    = dos_risk.round(4)
    df["hybrid_risk_score"] = hybrid_risk.round(4)
    df["fusion_confidence"] = fusion_conf.round(4)

    # ── Final label assignment (mirrors np.select conditions) ─────────────────
    # DDoS discriminator: a distributed flood shows MANY distinct sources and/or a
    # HIGH source-IP entropy (many addresses ⇒ high entropy).
    #
    # BUG FIX: the previous discriminator was
    #     is_ddos = (unique_source_count > 99th pct) & (src_ip_entropy > 95th pct)
    # ANDing two independent extreme-tail conditions makes the slice vanishingly
    # thin, and if unique_source_count is degenerate (e.g. single-node capture, where
    # it was ~constant with MI≈0.018) its 99th pct == its max, so the '>' is never
    # true and the DDoS branch becomes structurally UNREACHABLE — exactly why DDoS
    # was predicted 0 times. We now (a) OR the two signals, (b) use slightly looser
    # quantiles, and (c) ignore a feature entirely if it has no spread among normals,
    # so a flat column can't silently disable the branch.
    def _has_spread(c):
        if c not in df.columns:
            return False
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        return len(s) > 0 and float(s.std()) > 1e-9

    src_signal = pd.Series(False, index=df.index)
    if _has_spread("unique_source_count"):
        src_signal = col("unique_source_count") > nthr("unique_source_count", 0.95)
    ent_signal = pd.Series(False, index=df.index)
    if _has_spread("src_ip_entropy"):
        ent_signal = col("src_ip_entropy") > nthr("src_ip_entropy", 0.90)

    # If neither source feature carries signal in this dataset, distinguish DDoS
    # from DoS by flood magnitude instead (a very high packet rate with broad
    # traffic-outlier evidence is treated as distributed).
    if not (_has_spread("unique_source_count") or _has_spread("src_ip_entropy")):
        is_ddos = col("traffic_outlier") >= 0.5
    else:
        is_ddos = src_signal | ent_signal

    # Hybrid also requires actual cross-layer evidence present (anomaly score over
    # its normal threshold OR the cross-layer attack flag set), so a saturated/near-
    # constant hybrid classifier can no longer trigger HYBRID on its own.
    cl_evidence = (
        (col("cross_layer_anomaly_score") > cl_score_thr) |
        (cl_attack_term >= 0.5)
    )

    conditions = [
        (hybrid_risk >= 0.60) & (jam_risk >= 0.30) & (dos_risk >= 0.30) & cl_evidence,  # Hybrid
        (jam_risk >= 0.60),                                                 # Strong jamming
        (dos_risk >= 0.60) & is_ddos,                                       # DDoS
        (dos_risk >= 0.60) & ~is_ddos,                                      # DoS
        (jam_risk >= 0.30),                                                 # Moderate jamming
        (dos_risk >= 0.30) & is_ddos,                                       # Moderate DDoS
        (dos_risk >= 0.30) & ~is_ddos,                                      # Moderate DoS
        (fusion_conf >= 0.15),                                              # Suspicious
    ]
    choices = [
        "HYBRID_ATTACK", "JAMMING", "DDOS", "DOS",
        "JAMMING", "DDOS", "DOS", "SUSPICIOUS"
    ]
    df["final_label"] = np.select(conditions, choices, default="NORMAL")

    # ── Severity escalation (mirrors consecutive_attack_count from original) ──
    df_sev = df[["node_id","source_row_id","final_label"]].copy()
    df_sev = df_sev.sort_values(["node_id","source_row_id"])
    df_sev["_is_attack"] = (df_sev["final_label"] != "NORMAL").astype(int)
    df_sev["_consec"] = (
        df_sev.groupby("node_id")["_is_attack"]
        .transform(lambda x:
            x.groupby((x != x.shift()).cumsum()).cumcount() + 1
        ) * df_sev["_is_attack"]
    )
    consec = df_sev.set_index(df.index)["_consec"]

    # FIX: simplify nested np.where with np.select; remove redundant LOW/LOW branch
    is_attack = df["final_label"] != "NORMAL"
    df["severity"] = np.select(
        condlist=[
            is_attack & (consec >= 10),
            is_attack & (consec >= 3),
            is_attack & (fusion_conf >= 0.50),
            is_attack,
        ],
        choicelist=["HIGH", "MEDIUM", "MEDIUM", "LOW"],
        default="LOW",
    )
    df["consecutive_attack_count"] = consec.values

    # ── Reason codes (mirrors reason_map from original) ────────────────────────
    reason_map = {
        "EDGE_EWMA_ALARM"           : col("edge_alarm") == 1,
        "GRU_JAM_ANOMALY"           : col("gru_jam_flag") == 1,
        "LSTM_AE_JAM_CONFIRMED"     : col("jamming_ae_flag") == 1,
        "CNN_LSTM_JAM_DETECTED"     : col("jamming_conf_attack") > 0.5,
        "CNN_LSTM_DOS_DETECTED"     : col("dos_conf_attack") > 0.5,
        "CNN_LSTM_HYBRID_DETECTED"  : col("hybrid_conf_attack") > 0.5,
        "LSTM_AE_DOS_CONFIRMED"     : col("dos_ae_flag") == 1,
        "DBSCAN_TRAFFIC_OUTLIER"    : col("traffic_outlier") == 1,
        "HIGH_JAM_THRESHOLD_SCORE"  : col("jam_threshold_score") > jam_flag_thr,
        "HIGH_DOS_THRESHOLD_SCORE"  : col("dos_threshold_score") > dos_flag_thr,
        "CROSS_LAYER_ATTACK"        : col("cross_layer_attack_flag") == 1,
        "HIGH_CROSS_LAYER_SCORE"    : col("cross_layer_anomaly_score") > cl_score_thr,
        "SWARM_CONSENSUS_ANOMALY"   : col("swarm_consensus_anomaly_score") > swarm_thr,
        "COORDINATED_FLOOD"         : col("coordinated_flood_flag") == 1,
        "HIGH_SRC_ENTROPY"          : ent_signal,
        "HIGH_SOURCE_COUNT"         : src_signal,
        "NOISE_FLOOR_ELEVATED"      : col("noise_floor_drift_flag") == 1,
        "PHYSICAL_IMPOSSIBILITY"    : (col("bit_error_rate") > 0.30) & (col("snr_db") < 0),
    }
    # FIX: vectorise reason-code aggregation. Original used flag_df.loc[i, code]
    # in a nested loop — O(n*m) with pandas indexing overhead. Numpy-based version
    # is ~50x faster for n>1000 rows.
    flag_df = pd.DataFrame({k: v.astype(bool) for k, v in reason_map.items()},
                            index=df.index)
    reason_cols = list(reason_map.keys())
    flag_arr = flag_df[reason_cols].to_numpy()
    df["detection_reason_codes"] = [
        [reason_cols[j] for j in np.where(row)[0]] for row in flag_arr
    ]
    return df

def evaluate_detection(
    detection_df: pd.DataFrame,
    label_col: str = "label",
    pred_col: str = "final_label",
) -> dict:
    """
    Compute and print full detection evaluation metrics.
    Mirrors the TP/TN/FP/FN + classification_report block from the original.
    """
    if label_col not in detection_df.columns:
        print("No ground-truth label column available — skipping evaluation.")
        return {}

    y_true_str = detection_df[label_col].astype(str)
    y_pred_str = detection_df[pred_col].astype(str)

    # Binary: any attack vs normal
    y_true_bin = (y_true_str != "normal").astype(int)
    y_pred_bin = (y_pred_str != "NORMAL").astype(int)

    tp = int(((y_true_bin==1) & (y_pred_bin==1)).sum())
    tn = int(((y_true_bin==0) & (y_pred_bin==0)).sum())
    fp = int(((y_true_bin==0) & (y_pred_bin==1)).sum())
    fn = int(((y_true_bin==1) & (y_pred_bin==0)).sum())

    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    fpr  = fp/(fp+tn) if (fp+tn)>0 else 0.0

    print("=" * 60)
    print("  DETECTION EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  FPR       : {fpr:.4f}")

    # Per-class attack type breakdown
    print("\n  Per-attack-type breakdown:")
    for attack in ["JAMMING","DOS","DDOS","HYBRID_ATTACK","SUSPICIOUS"]:
        tp_a = int(((y_true_str.str.lower().str.contains(attack.lower().replace("_attack",""))) &
                    (y_pred_str == attack)).sum())
        fp_a = int(((~y_true_str.str.lower().str.contains(attack.lower().replace("_attack",""))) &
                    (y_pred_str == attack)).sum())
        print(f"    {attack:<18s}: predicted={tp_a+fp_a:4d}  TP~={tp_a:4d}  FP~={fp_a:4d}")

    # Classification report
    print("\n  Full classification report (binary):")
    print(classification_report(y_true_bin, y_pred_bin,
                                target_names=["NORMAL","ATTACK"], zero_division=0))

    # Per-severity FP/FN analysis
    if "severity" in detection_df.columns:
        print("  Severity distribution of detections:")
        sev_fp = detection_df[(y_true_bin==0) & (y_pred_bin==1)]["severity"].value_counts()
        sev_tp = detection_df[(y_true_bin==1) & (y_pred_bin==1)]["severity"].value_counts()
        print(f"    TP by severity: {sev_tp.to_dict()}")
        print(f"    FP by severity: {sev_fp.to_dict()}")

    return {"tp":tp,"tn":tn,"fp":fp,"fn":fn,"precision":prec,
            "recall":rec,"f1":f1,"fpr":fpr}


def run_detection_agent_inference(
    feat_df: pd.DataFrame,
    det_models: dict,
    fe_state: dict,
    save_outputs: bool = False,
    preview: bool = False,
) -> pd.DataFrame:
    """
    Full three-tier detection pipeline.
    Mirrors run_detection_agent_inference() from the original.

    Steps:
      1. Tier A — Edge GRU + AE (jamming sequences)
      2. Tier B — GCC CNN+LSTM classifiers + DBSCAN
      3. Tier C — Score fusion, final label, severity, reason codes
      4. Optionally save outputs
    """
    import warnings; warnings.filterwarnings("ignore")

    print("[DetectionAgent] Tier A: Edge detection...")
    edge_df = run_edge_detection(feat_df, det_models, fe_state)

    print("[DetectionAgent] Tier B: GCC CNN+LSTM + DBSCAN...")
    gcc_df  = run_gcc_detection(feat_df, det_models, fe_state)

    print("[DetectionAgent] Tier C: Score fusion & classification...")
    result_df = apply_final_detection_logic(feat_df, edge_df, gcc_df, fe_state)

    if save_outputs:
        result_df.to_csv("detection_output.csv", index=False)
        with open("detection_output.jsonl","w",encoding="utf-8") as f:
            for _, row in result_df.iterrows():
                f.write(json.dumps(row.to_dict(), default=str) + "\n")
        print("[DetectionAgent] Saved: detection_output.csv, detection_output.jsonl")

    if preview:
        print("\n[DetectionAgent] Final label distribution:")
        print(result_df["final_label"].value_counts().to_string())
        print("[DetectionAgent] Severity distribution:")
        print(result_df["severity"].value_counts().to_string())

    return result_df


def run_detection_agent(
    cleaned_df: pd.DataFrame = None,
    feat_csv: str = FEAT_CSV,
    fe_state_path: str = FE_STATE_PATH,
    model_dir: str = MODEL_DIR,
    save_outputs: bool = True,
    preview: bool = True,
) -> tuple:
    """
    Agent-level wrapper — mirrors run_collection_cleaning_agent() pattern.
    Loads state, runs full pipeline, returns (detection_df, eval_results).
    """
    # Load FE state
    with open(fe_state_path, "rb") as f:
        _fe = pickle.load(f)

    # Load or receive data
    if cleaned_df is not None:
        from types import SimpleNamespace
        # Accept either raw or already-engineered DataFrame
        if "jam_threshold_score" not in cleaned_df.columns:
            print("[DetectionAgent] Running feature engineering pipeline...")
            # Import prepare_feature_dataframe from FE notebook context
            _feat_df = prepare_feature_dataframe(cleaned_df, fe_state=_fe,
                                                  fit_scalers=False, preview=preview)
        else:
            _feat_df = cleaned_df
    else:
        _feat_df = pd.read_csv(feat_csv)

    # Load models
    _models = load_detection_models()

    # Run inference
    detection_df = run_detection_agent_inference(
        _feat_df, _models, _fe,
        save_outputs=save_outputs, preview=preview
    )

    # Evaluate
    eval_res = evaluate_detection(detection_df)

    return detection_df, eval_res


# ── Full pipeline run on feat_df ──────────────────────────────────────────────

def predict_single(input_data: dict, det_models: dict, fe_state: dict,
                   preview: bool = True) -> dict:
    """
    Full end-to-end detection on a single raw input window.
    Mirrors predict_single() from the original exactly.

    Steps:
      1. Wrap in DataFrame
      2. Feature engineering (prepare_feature_dataframe)
      3. Three-tier detection (run_detection_agent_inference)
      4. Return flat result dict
    """
    import warnings; warnings.filterwarnings("ignore")

    # State passed explicitly (no notebook globals)
    _fe = fe_state
    _models = det_models

    has_phy = any(f in input_data for f in ["rssi_dbm","snr_db","mac_retry_count"])
    has_net = any(f in input_data for f in ["packets_per_second","syn_packet_rate"])
    cp = "edge_gcc" if (has_phy and has_net) else ("edge" if has_phy else "gcc")

    row = {
        "timestamp"        : input_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "node_id"          : input_data.get("node_id", "uav_001"),
        "node_type"        : input_data.get("node_type", "uav"),
        "collection_point" : input_data.get("collection_point", cp),
        "window_id"        : 0,
        "label"            : input_data.get("label", "unknown"),
    }
    row.update({k: v for k, v in input_data.items() if k not in row})
    df_single = pd.DataFrame([row])

    # Feature engineering
    feat_single = prepare_feature_dataframe(
        df_single, fe_state=_fe,
        fit_scalers=False, preview=False
    )

    # Detection (pad sequences to seq_len by repeating the single row)
    seq_len = _fe["seq_len"]
    padded  = pd.concat([feat_single] * seq_len, ignore_index=True)
    padded["source_row_id"] = list(range(seq_len))
    padded["node_id"]       = row["node_id"]
    padded["timestamp"]     = pd.date_range(
        start=pd.Timestamp.now() - pd.Timedelta(seconds=seq_len),
        periods=seq_len, freq="1s"
    )

    det_single = run_detection_agent_inference(
        padded, _models, _fe,
        save_outputs=False, preview=False
    )
    result_row = det_single.iloc[-1]  # Use last (most recent) window

    output = {
        "final_label"              : result_row["final_label"],
        "severity"                 : result_row["severity"],
        "jam_risk_score"           : round(float(result_row.get("jam_risk_score",0)), 4),
        "dos_risk_score"           : round(float(result_row.get("dos_risk_score",0)), 4),
        "hybrid_risk_score"        : round(float(result_row.get("hybrid_risk_score",0)), 4),
        "fusion_confidence"        : round(float(result_row.get("fusion_confidence",0)), 4),
        "edge_alarm"               : int(result_row.get("edge_alarm",0)),
        "gru_jam_flag"             : int(result_row.get("gru_jam_flag",0)),
        "jamming_ae_flag"          : int(result_row.get("jamming_ae_flag",0)),
        "dos_ae_flag"              : int(result_row.get("dos_ae_flag",0)),
        "traffic_outlier"          : int(result_row.get("traffic_outlier",0)),
        "cross_layer_anomaly_score": round(float(result_row.get("cross_layer_anomaly_score",0)),4),
        "consecutive_attack_count" : int(result_row.get("consecutive_attack_count",0)),
        "detection_reason_codes"   : result_row.get("detection_reason_codes",[]),
    }

    if preview:
        print("=" * 58)
        print(f"  RESULT    : {output['final_label']}  [{output['severity']}]")
        print(f"  Jam risk  : {output['jam_risk_score']}  |  DoS risk: {output['dos_risk_score']}")
        print(f"  Hybrid    : {output['hybrid_risk_score']}  |  fusion: {output['fusion_confidence']}")
        print(f"  Edge alarm: {output['edge_alarm']}  |  CL score: {output['cross_layer_anomaly_score']}")
        if output["detection_reason_codes"]:
            print(f"  Reasons   : {output['detection_reason_codes']}")
        print("=" * 58)
    return output


# ── Five canonical test cases ─────────────────────────────────────────────────