"""
mas_ids.utils
=============
Shared utilities used across all six agents:
  - NullLogger        : no-op logger stub used during training phases
  - numeric helpers   : safe_numeric_df, safe_numeric_fill, safe_div, shannon_entropy
  - threshold helper  : q_thresh (quantile calibrated on normal rows)
  - per-node helpers  : rolling_per_node, diff_per_node
  - scaler helpers    : fit_scaler_on_normal, apply_scaler
  - sequence helpers  : create_sequences, split_train_test, balance_sequences
  - reporting         : print_feature_profile
"""
import random
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import LABEL_MAP


class NullLogger:
    """
    Minimal logger stub used during training phases where the full
    LoggerAgent is not yet initialised. Defined once, reused by all agents.
    """
    def log_event(self, *a, **kw):          pass
    def log_uav_ugv_event(self, *a, **kw):  pass
    def log_edge_event(self, *a, **kw):     pass
    def log_gcc_event(self, *a, **kw):      pass
    def log_response_action(self, *a, **kw):pass
    def flush(self):                        pass


# ── Numeric / DataFrame helpers ───────────────────────────────────────────────
def safe_numeric_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Coerce columns to float, replacing inf/NaN with 0 (FE variant)."""
    df = df.copy()
    for col in cols:
        if col not in df.columns: df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def safe_numeric_fill(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Coerce columns to float, replacing inf/NaN with 0 (scaling variant)."""
    df = df.copy()
    for col in cols:
        if col not in df.columns: df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def safe_div(a, b, fill: float = 0.0, clip_max: float = None) -> pd.Series:
    """Element-wise division guarded against zero denominator."""
    result = a / (b + 1e-9)
    result = result.replace([np.inf, -np.inf], fill).fillna(fill)
    if clip_max is not None: result = result.clip(upper=clip_max)
    return result


def shannon_entropy(series: pd.Series) -> float:
    """Shannon entropy (bits) of a value distribution."""
    counts = series.value_counts(dropna=True)
    total  = counts.sum()
    if total == 0: return 0.0
    probs = counts / total
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def q_thresh(df: pd.DataFrame, col: str, q: float = 0.95,
             label_col: str = 'label', normal_label='normal') -> float:
    """Quantile threshold calibrated on NORMAL rows only."""
    if label_col in df.columns:
        mask   = df[label_col] == normal_label
        subset = df[mask] if mask.sum() > 10 else df
    else:
        subset = df
    s = pd.to_numeric(subset[col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.quantile(q)) if len(s) > 0 else 0.0


def rolling_per_node(df, col, node_col, time_col, window, func, min_periods=1):
    """Rolling aggregation per node sorted by time."""
    result = pd.Series(np.nan, index=df.index)
    df_s = df.sort_values([node_col, time_col])
    for _, g in df_s.groupby(node_col, sort=False):
        vals = pd.to_numeric(g[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        result.loc[g.index] = getattr(vals.rolling(window, min_periods=min_periods), func)().values
    return result.fillna(0.0)


def diff_per_node(df, col, node_col, time_col, periods=1, abs_val=False):
    """Per-node first difference (rate of change)."""
    result = pd.Series(0.0, index=df.index)
    df_s = df.sort_values([node_col, time_col])
    for _, g in df_s.groupby(node_col, sort=False):
        vals = pd.to_numeric(g[col], errors='coerce').fillna(0.0)
        d = vals.diff(periods).fillna(0.0)
        result.loc[g.index] = (d.abs() if abs_val else d).values
    return result


def print_feature_profile(df: pd.DataFrame, feat_cols: list, title: str, top_k: int = 8) -> None:
    """Per-label mean/std for top_k features. Shared across DA and FE."""
    print('=' * 65)
    print(f'  {title}')
    print('=' * 65)
    if 'label' not in df.columns:
        print('  No label column.'); return
    for feat in [c for c in feat_cols if c in df.columns][:top_k]:
        stats_df = df.groupby('label')[feat].agg(['mean', 'std', 'min', 'max'])
        print(f'\n  [{feat}]')
        print(stats_df.round(4).to_string())
    print()


# ── Scaler helpers (used by FE and Detection agents) ──────────────────────────
def fit_scaler_on_normal(df: pd.DataFrame, feat_cols: list,
                          label_col: str = 'label',
                          normal_label: str = 'normal') -> MinMaxScaler:
    """Fit MinMaxScaler only on normal rows."""
    if label_col in df.columns:
        nr = df[df[label_col] == normal_label]
        normal_rows = nr if len(nr) >= 10 else df
    else:
        normal_rows = df
    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit(normal_rows[feat_cols].astype(float))
    return sc


def apply_scaler(df: pd.DataFrame, feat_cols: list, scaler: MinMaxScaler) -> pd.DataFrame:
    df = df.copy()
    df[feat_cols] = scaler.transform(df[feat_cols].astype(float))
    return df


def create_sequences(df, feat_cols, seq_len, node_col='node_id',
                     time_col='timestamp', label_col='label'):
    """Build overlapping sliding-window sequences per node."""
    df_sorted = df.sort_values([node_col, time_col]).reset_index(drop=True)
    X_list, y_list, meta_list = [], [], []
    for node_id, group in df_sorted.groupby(node_col, sort=False):
        group = group.reset_index(drop=True)
        vals  = group[feat_cols].astype(float).values
        lbls  = group[label_col].map(LABEL_MAP).fillna(0).astype(int).values \
                if label_col in group.columns else np.zeros(len(group), dtype=int)
        n = len(vals)
        if n < seq_len: continue
        for i in range(n - seq_len + 1):
            end_row = group.iloc[i + seq_len - 1]
            X_list.append(vals[i: i + seq_len])
            y_list.append(lbls[i + seq_len - 1])
            meta_list.append({'node_id': node_id, 'window_end_idx': i + seq_len - 1,
                              'timestamp': str(end_row.get(time_col, '')),
                              'label': end_row.get(label_col, 'normal')})
    if not X_list:
        return (np.empty((0, seq_len, len(feat_cols)), dtype=np.float32),
                np.empty(0, dtype=np.int32), [])
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int32), meta_list)


def split_train_test(X, y, meta, train_ratio=0.70):
    """Chronological split — preserves temporal order."""
    n = len(X); split = max(1, min(int(n * train_ratio), n - 1))
    return X[:split], X[split:], y[:split], y[split:], meta[:split], meta[split:]


def balance_sequences(X, y, label_map, target_ratio=0.5,
                       noise_std=0.02, random_state=42):
    """Oversample minority attack classes using jitter + interpolation."""
    rng = np.random.default_rng(random_state)
    inv = {v: k for k, v in label_map.items()}
    classes, counts = np.unique(y, return_counts=True)
    count_d = dict(zip(classes.tolist(), counts.tolist()))
    normal_lbl = label_map.get('normal', 0)
    n_normal   = count_d.get(normal_lbl, 0)
    attack_cls = [c for c in classes if c != normal_lbl]
    X_aug, y_aug = [X], [y]
    for cls in attack_cls:
        mask = y == cls; X_c = X[mask]; n_c = len(X_c)
        n_atk = max(n_c, int(n_normal * target_ratio / (1.0 - target_ratio) /
                              max(len(attack_cls), 1)))
        n_need = n_atk - n_c
        if n_need <= 0: continue
        ia = rng.integers(0, n_c, n_need); ib = rng.integers(0, n_c, n_need)
        jit  = (X_c[ia] + np.random.normal(0, noise_std, X_c[ia].shape)).clip(0,1).astype(np.float32)
        alph = np.random.uniform(0.3, 0.7)
        intp = (alph*X_c[ia] + (1-alph)*X_c[ib]).astype(np.float32)
        Xn   = np.concatenate([jit, intp])[:n_need]
        X_aug.append(Xn); y_aug.append(np.full(len(Xn), cls, dtype=np.int32))
    Xb = np.concatenate(X_aug, axis=0).astype(np.float32)
    yb = np.concatenate(y_aug, axis=0).astype(np.int32)
    perm = rng.permutation(len(Xb))
    return Xb[perm], yb[perm]


print('Shared utilities defined.')