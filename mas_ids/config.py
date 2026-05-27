"""
mas_ids.config
==============
Centralised configuration — single source of truth for all six agents.

DATASET NOTE: this project is fully self-contained. Training data is generated
synthetically by TrafficDataGenerator (mas_ids.agents.data_agent) — no external
CSV upload is required. To plug in a real dataset, replace the call to
generator.generate(...) with pd.read_csv("<your-file>.csv") in your driver.
"""
import os

# ── Reproducibility ───────────────────────────────────────────────────────────
GLOBAL_SEED = 42

# ── Dataset sizes ─────────────────────────────────────────────────────────────
N_NORMAL   = 2000
N_JAMMING  = 800
N_DOS      = 600
N_DDOS     = 600
N_HYBRID   = 400

# ── Sequence / model config ───────────────────────────────────────────────────
SEQ_LEN        = 10      # sliding-window length (1-second windows)
TRAIN_EPOCHS   = 50      # CNN+LSTM / LSTM-AE training epochs
TRAIN_BATCH    = 512     # training batch size
DQN_EPISODES   = 150     # Response DQN episodes
DQN_BATCH      = 64      # DQN minibatch size
DQN_SYNC_EVERY = 10      # target-net sync interval
KPI_WINDOW     = 30      # Management Agent batch size (rows per Nash solve)
LOG_BATCH_SIZE = 10      # Logger Merkle batch size

# ── Shared label map (all agents use this) ────────────────────────────────────
LABEL_MAP = {'normal': 0, 'jamming': 1, 'dos': 2, 'ddos': 3, 'hybrid': 4}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}
N_CLASSES = len(LABEL_MAP)   # 5

# ── Directory / file paths ────────────────────────────────────────────────────
MODEL_DIR          = 'saved_detection_models'
RESPONSE_MODEL_DIR = 'saved_response_models'
LOG_PREFIX         = 'mas_ids_unified'

# Detection model file paths
JAM_MODEL_PATH    = os.path.join(MODEL_DIR, 'cnn_lstm_jamming.keras')
DOS_MODEL_PATH    = os.path.join(MODEL_DIR, 'cnn_lstm_dos.keras')
HYB_MODEL_PATH    = os.path.join(MODEL_DIR, 'cnn_lstm_hybrid.keras')
JAM_AE_PATH       = os.path.join(MODEL_DIR, 'lstm_ae_jamming.keras')
DOS_AE_PATH       = os.path.join(MODEL_DIR, 'lstm_ae_dos.keras')
DETECT_CONFIG_PATH= os.path.join(MODEL_DIR, 'detection_config.json')

# Optional file-based I/O paths (used by standalone agent wrappers)
FE_STATE_PATH = 'feature_engineering_state.pkl'
FEAT_CSV      = 'all_feat_engineered.csv'
ALL_CSV       = 'all_cleaned.csv'

# Response model file paths
ROUTE_POLICY_WEIGHTS = os.path.join(RESPONSE_MODEL_DIR, 'route_policy_torch.pt')
ROUTE_POLICY_CONFIG  = os.path.join(RESPONSE_MODEL_DIR, 'route_policy_config.json')


def setup_environment(verbose: bool = True):
    """
    Configure seeds, GPU memory growth, mixed precision, and create output
    directories. Call once at the start of a driver script / notebook.
    Returns the torch device (always CPU for the DQN — avoids P100 CUDA mismatch).
    """
    import random
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    import torch

    os.makedirs(MODEL_DIR,          exist_ok=True)
    os.makedirs(RESPONSE_MODEL_DIR, exist_ok=True)

    tf.random.set_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    os.environ['TF_CPP_MIN_LOG_LEVEL']      = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for _g in gpus:
            try:
                tf.config.experimental.set_memory_growth(_g, True)
            except RuntimeError:
                pass
        mixed_precision.set_global_policy('mixed_float16')
        if verbose:
            print(f'TF GPU ready: {gpus}  |  policy: {mixed_precision.global_policy()}')
    else:
        if verbose:
            print('TF: CPU only')

    torch_device = torch.device('cpu')
    if verbose:
        print(f'PyTorch device: {torch_device}')
        print('Global configuration loaded.')
        print(f'  SEQ_LEN={SEQ_LEN} | EPOCHS={TRAIN_EPOCHS} | DQN_EPISODES={DQN_EPISODES}')
    return torch_device


# Module-level torch device (importable as a constant)
TORCH_DEVICE = None  # set by setup_environment(); fallback below
try:
    import torch as _torch
    TORCH_DEVICE = _torch.device('cpu')
except Exception:
    pass
