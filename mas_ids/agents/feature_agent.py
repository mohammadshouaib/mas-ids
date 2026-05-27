"""
mas_ids.agents.feature_agent
============================
Agent 2 — Feature Engineering.

Temporal/statistical features (EWMA, CUSUM, rolling), cross-layer fusion,
swarm consensus, mutual-information feature selection, online edge detectors,
sequence augmentation, and the prepare_feature_dataframe entry point.

Public API:
  TemporalJammingFeatureBuilder, TemporalDoSFeatureBuilder,
  CrossLayerFeatureBuilder, SwarmConsensusFeatureBuilder,
  select_features_by_mutual_info, OnlineEWMA, OnlineCUSUM,
  EdgeLightweightDetector, jitter_sequences, prepare_feature_dataframe
"""
from datetime import datetime, timezone
from collections import deque, defaultdict
import os
import math
import pickle
import random
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from ..config import LABEL_MAP, SEQ_LEN, GLOBAL_SEED
from ..utils import (
    safe_numeric_df, safe_numeric_fill, safe_div, shannon_entropy, q_thresh,
    rolling_per_node, diff_per_node, fit_scaler_on_normal, apply_scaler,
    create_sequences, split_train_test, balance_sequences,
)
from .data_agent import (
    JAMMING_FEATURES, DOS_FEATURES, ALL_FEATURES,
    JAM_FEAT_COLS, DOS_FEAT_COLS, ALL_FEAT_COLS,
)


class TemporalJammingFeatureBuilder:
    """
    Computes temporal and statistical features for the Jamming detection pipeline.
    Operates per-node over chronologically ordered 1-second windows.

    Output features are used by:
      - Edge EWMA/CUSUM detector (lightweight, real-time)
      - GCC CNN+LSTM confirmation model (temporal sequences)
    """

    # EWMA smoothing factor — higher = more reactive (faster response)
    EWMA_ALPHA     = 0.3

    # Rolling window lengths (in 1-second windows)
    SHORT_WIN      = 5
    MEDIUM_WIN     = 10
    LONG_WIN       = 30

    # CUSUM parameters
    CUSUM_K_FACTOR = 0.5    # slack: allowable "normal" deviation (in units of sigma)
    CUSUM_H_FACTOR = 5.0    # decision threshold (in units of sigma)

    # Warmup rows to skip before calibrating CUSUM (prevents false positives at boot)
    WARMUP_ROWS    = 10

    def __init__(self, node_col: str = "node_id", time_col: str = "timestamp"):
        self.node_col = node_col
        self.time_col = time_col

    # ── A. EWMA Features ──────────────────────────────────────────────────────
    def _add_ewma(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EWMA-smoothed versions of key signal metrics.
        Used by the edge EWMA detector to track slow baseline shifts.
        """
        ewma_targets = [
            "rssi_dbm", "snr_db", "sinr_db",
            "packet_delivery_ratio", "channel_occupancy_pct",
        ]
        df_sorted = df.sort_values([self.node_col, self.time_col])
        for feat in ewma_targets:
            if feat not in df.columns:
                continue
            result = pd.Series(np.nan, index=df.index)
            for _, group in df_sorted.groupby(self.node_col, sort=False):
                vals   = pd.to_numeric(group[feat], errors="coerce").ffill().fillna(0.0)
                ewma   = vals.ewm(alpha=self.EWMA_ALPHA, adjust=False).mean()
                result.loc[group.index] = ewma.values
            df[f"{feat}_ewma"] = result.fillna(0.0)
        return df

    # ── B. Rolling Statistics ─────────────────────────────────────────────────
    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Short / medium / long-window rolling mean and std.
        High rolling std on SNR/RSSI is a key reactive jamming signature.
        """
        stat_targets = {
            "rssi_dbm"               : [self.SHORT_WIN, self.MEDIUM_WIN],
            "snr_db"                 : [self.SHORT_WIN, self.MEDIUM_WIN],
            "sinr_db"                : [self.SHORT_WIN],
            "bit_error_rate"         : [self.SHORT_WIN, self.MEDIUM_WIN],
            "bad_packet_ratio"       : [self.SHORT_WIN],
            "mac_retry_count"        : [self.SHORT_WIN, self.MEDIUM_WIN],
            "retransmission_count"   : [self.SHORT_WIN],
            "channel_occupancy_pct"  : [self.SHORT_WIN, self.LONG_WIN],
            "cca_failure_count"      : [self.SHORT_WIN],
            "packet_delivery_ratio"  : [self.MEDIUM_WIN, self.LONG_WIN],
        }
        for feat, windows in stat_targets.items():
            if feat not in df.columns:
                continue
            for w in windows:
                df[f"{feat}_roll_mean_{w}s"] = rolling_per_node(
                    df, feat, self.node_col, self.time_col, w, "mean")
                df[f"{feat}_roll_std_{w}s"]  = rolling_per_node(
                    df, feat, self.node_col, self.time_col, w, "std")
        return df

    # ── C. Rate-of-Change Features ────────────────────────────────────────────
    def _add_rate_of_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-window delta (Δ) and absolute delta for key metrics.
        Sudden RSSI drop rate and sudden SNR drop rate are strong jamming onset signals.
        """
        roc_targets = [
            "rssi_dbm", "snr_db", "sinr_db",
            "packet_delivery_ratio", "channel_occupancy_pct",
            "mac_retry_count", "cca_failure_count",
        ]
        for feat in roc_targets:
            if feat not in df.columns:
                continue
            df[f"{feat}_delta"]      = diff_per_node(df, feat, self.node_col, self.time_col)
            df[f"{feat}_abs_delta"]  = diff_per_node(df, feat, self.node_col, self.time_col, abs_val=True)

        # Focused drop indicators for signal strength (dropping RSSI / SNR is the
        # strongest single-feature jamming indicator)
        if "rssi_dbm" in df.columns:
            df["rssi_drop"]     = df["rssi_dbm_delta"].clip(upper=0).abs()  # only negative changes
        if "snr_db" in df.columns:
            df["snr_drop"]      = df["snr_db_delta"].clip(upper=0).abs()
        if "packet_delivery_ratio" in df.columns:
            df["pdr_drop"]      = df["packet_delivery_ratio_delta"].clip(upper=0).abs()
        return df

    # ── D. CUSUM Accumulators ─────────────────────────────────────────────────
    def _add_cusum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CUSUM (Cumulative Sum) change-point detectors for downward shifts in SNR
        and upward shifts in retransmission count.

        For each node:
          - Calibrate μ and σ from the first WARMUP_ROWS rows
          - Set slack k = K_FACTOR * σ,  threshold H = H_FACTOR * σ
          - Accumulate S+ (upward) and S- (downward) CUSUM
          - Fire flag when |S| > H

        Mirrors the CUSUM logic required for edge-level jamming detection.
        """
        cusum_targets = [
            ("snr_db",            "down"),   # jamming → SNR drops
            ("sinr_db",           "down"),
            ("mac_retry_count",   "up"),     # jamming → retries increase
            ("cca_failure_count", "up"),
            ("channel_occupancy_pct", "up"),
        ]
        df_sorted = df.sort_values([self.node_col, self.time_col]).copy()

        for feat, direction in cusum_targets:
            if feat not in df.columns:
                continue
            cusum_vals  = np.zeros(len(df_sorted))
            cusum_flags = np.zeros(len(df_sorted), dtype=int)
            cusum_k_arr = np.zeros(len(df_sorted))
            cusum_h_arr = np.zeros(len(df_sorted))

            for _, group in df_sorted.groupby(self.node_col, sort=False):
                idx  = group.index.tolist()
                vals = pd.to_numeric(group[feat], errors="coerce").fillna(0.0).values
                n    = len(vals)

                # Calibrate from warmup window
                warmup = min(self.WARMUP_ROWS, max(3, n // 5))
                mu     = np.mean(vals[:warmup])
                sigma  = np.std(vals[:warmup]) + 1e-6
                k      = self.CUSUM_K_FACTOR * sigma
                H      = self.CUSUM_H_FACTOR  * sigma

                S = 0.0
                for i in range(n):
                    if direction == "down":
                        # S- accumulates when value drops below μ - k
                        S = min(0.0, S + (vals[i] - mu + k))
                        cusum_vals[df_sorted.index.get_loc(idx[i])]  = abs(S)
                        cusum_flags[df_sorted.index.get_loc(idx[i])] = int(abs(S) > H)
                    else:
                        # S+ accumulates when value rises above μ + k
                        S = max(0.0, S + (vals[i] - mu - k))
                        cusum_vals[df_sorted.index.get_loc(idx[i])]  = S
                        cusum_flags[df_sorted.index.get_loc(idx[i])] = int(S > H)
                    cusum_k_arr[df_sorted.index.get_loc(idx[i])] = k
                    cusum_h_arr[df_sorted.index.get_loc(idx[i])] = H

            df_sorted[f"{feat}_cusum"]      = cusum_vals
            df_sorted[f"{feat}_cusum_flag"] = cusum_flags

        # Re-align to original df index
        for col in df_sorted.columns:
            if "_cusum" in col:
                df[col] = df_sorted[col]
        return df

    # ── E. Anomaly Confirmation Flags ─────────────────────────────────────────
    def _add_threshold_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Binary flags based on normal-calibrated thresholds.
        Mirrors suspicious_speed / suspicious_jump logic in the original.
        Each flag is confirmed over a rolling window (prevents single-sample FPs).
        """
        confirm_win = 3   # require N consecutive windows above threshold

        # Calibrate thresholds from normal rows
        thresholds = {
            "rssi_dbm"            : (q_thresh(df, "rssi_dbm",            0.05), "below"),  # low RSSI = bad
            "snr_db"              : (q_thresh(df, "snr_db",              0.05), "below"),
            "sinr_db"             : (q_thresh(df, "sinr_db",             0.05), "below"),
            "bit_error_rate"      : (q_thresh(df, "bit_error_rate",      0.99), "above"),
            "bad_packet_ratio"    : (q_thresh(df, "bad_packet_ratio",    0.99), "above"),
            "mac_retry_count"     : (q_thresh(df, "mac_retry_count",     0.99), "above"),
            "cca_failure_count"   : (q_thresh(df, "cca_failure_count",   0.99), "above"),
            "channel_occupancy_pct": (q_thresh(df, "channel_occupancy_pct", 0.99), "above"),
        }

        df_sorted = df.sort_values([self.node_col, self.time_col])
        for feat, (thr, direction) in thresholds.items():
            if feat not in df.columns:
                continue
            raw_flag_name  = f"{feat}_raw_flag"
            conf_flag_name = f"{feat}_flag"       # confirmed (N-window) version

            # Raw flag
            if direction == "below":
                df[raw_flag_name] = (df[feat] < thr).astype(int)
            else:
                df[raw_flag_name] = (df[feat] > thr).astype(int)

            # Temporally confirmed flag (per node)
            conf_result = pd.Series(0, index=df.index)
            for _, group in df_sorted.groupby(self.node_col, sort=False):
                raw  = df.loc[group.index, raw_flag_name]
                conf = raw.rolling(confirm_win, min_periods=confirm_win).sum() >= confirm_win
                conf_result.loc[group.index] = conf.fillna(False).astype(int).values
            df[conf_flag_name] = conf_result

        df.drop(columns=[c for c in df.columns if c.endswith("_raw_flag")], inplace=True)

        # Composite jamming score: count of fired threshold flags
        flag_cols  = [c for c in df.columns if c.endswith("_flag") and
                      any(f in c for f in JAM_FEAT_COLS)]
        df["jam_threshold_score"] = df[flag_cols].sum(axis=1)
        return df

    # ── F. Noise Floor Drift (Stealthy Jamming) ───────────────────────────────
    def _add_noise_floor_drift(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tracks slow upward drift in ambient noise floor — signature of a
        wideband or low-power jammer that gradually degrades SNR.
        Mirrors add_drift_features() from the original GPS drift detector.
        """
        if "noise_floor_dbm" not in df.columns:
            df["noise_floor_drift_db"]     = 0.0
            df["noise_floor_drift_flag"]   = 0
            return df

        df_sorted = df.sort_values([self.node_col, self.time_col]).copy()
        DRIFT_WIN  = self.LONG_WIN     # 30 windows
        CONFIRM    = 5                 # require 5 consecutive windows elevated

        drift_vals  = np.zeros(len(df_sorted))
        raw_flags   = np.zeros(len(df_sorted), dtype=int)

        for _, group in df_sorted.groupby(self.node_col, sort=False):
            idx    = group.index.tolist()
            vals   = pd.to_numeric(group["noise_floor_dbm"], errors="coerce").fillna(-90.0).values

            # Calibrate anchor from normal rows (first WARMUP_ROWS)
            warmup    = min(self.WARMUP_ROWS, max(3, len(vals) // 5))
            anchor    = np.mean(vals[:warmup])
            drift_thr = max(3.0, float(np.std(vals[:warmup]) * 2.5))  # ≥ 3 dB

            for i in range(len(idx)):
                win_start = max(0, i - DRIFT_WIN)
                recent    = vals[win_start : i + 1]
                drift     = float(np.max(recent) - anchor)   # upward drift from anchor
                loc       = df_sorted.index.get_loc(idx[i])
                drift_vals[loc] = max(drift, 0.0)
                raw_flags[loc]  = int(drift > drift_thr)

        df_sorted["noise_floor_drift_db"] = drift_vals
        df_sorted["_nf_raw_flag"]         = raw_flags

        # Temporal confirmation per node
        conf_result = pd.Series(0, index=df_sorted.index)
        for _, group in df_sorted.groupby(self.node_col, sort=False):
            raw  = df_sorted.loc[group.index, "_nf_raw_flag"]
            conf = raw.rolling(CONFIRM, min_periods=CONFIRM).sum() >= CONFIRM
            conf_result.loc[group.index] = conf.fillna(False).astype(int).values
        df_sorted["noise_floor_drift_flag"] = conf_result
        df_sorted.drop(columns=["_nf_raw_flag"], inplace=True)

        # Merge back
        for col in ["noise_floor_drift_db", "noise_floor_drift_flag"]:
            df[col] = df_sorted[col]
        return df

    # ── Public API ─────────────────────────────────────────────────────────────
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full temporal feature pipeline for jamming detection."""
        df = safe_numeric_df(df, [c for c in JAM_FEAT_COLS if c in df.columns])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values([self.node_col, self.time_col]).reset_index(drop=True)

        print("  [TemporalJamming] EWMA features...")
        df = self._add_ewma(df)
        print("  [TemporalJamming] Rolling stats...")
        df = self._add_rolling_stats(df)
        print("  [TemporalJamming] Rate-of-change...")
        df = self._add_rate_of_change(df)
        print("  [TemporalJamming] CUSUM accumulators...")
        df = self._add_cusum(df)
        print("  [TemporalJamming] Threshold flags...")
        df = self._add_threshold_flags(df)
        print("  [TemporalJamming] Noise floor drift...")
        df = self._add_noise_floor_drift(df)

        new_cols = [c for c in df.columns if c not in JAM_FEAT_COLS + ["label", "timestamp", "node_id", "node_type", "collection_point", "window_id"]]
        print(f"  [TemporalJamming] Done — added {len(new_cols)} temporal features")
        return df


# ── Build temporal jamming features ──────────────────────────────────────────


class TemporalDoSFeatureBuilder:
    """
    Computes temporal and statistical features for the DoS/DDoS detection pipeline.
    Operates at the GCC level on aggregated 1-second traffic windows.
    """

    EWMA_ALPHA = 0.2       # slower EWMA for network traffic (smoother baseline)
    SHORT_WIN  = 5
    MEDIUM_WIN = 10
    LONG_WIN   = 30
    CUSUM_K    = 0.5
    CUSUM_H    = 5.0
    WARMUP     = 10

    def __init__(self, node_col: str = "node_id", time_col: str = "timestamp"):
        self.node_col = node_col
        self.time_col = time_col

    # ── A. Traffic Burst Detection ────────────────────────────────────────────
    def _add_traffic_burst_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Traffic burst index: current value relative to rolling baseline.
        A burst index >> 1 signals a flood onset.
        """
        burst_targets = [
            "packets_per_second", "bytes_per_second",
            "connection_attempts_per_sec", "syn_packet_rate",
            "icmp_packet_rate", "udp_packet_rate",
        ]
        for feat in burst_targets:
            if feat not in df.columns:
                continue
            baseline = rolling_per_node(df, feat, self.node_col, self.time_col,
                                        self.LONG_WIN, "mean")
            df[f"{feat}_burst_index"] = safe_div(df[feat], baseline, fill=1.0, clip_max=1000.0)
            df[f"{feat}_roll_std_{self.MEDIUM_WIN}s"] = rolling_per_node(
                df, feat, self.node_col, self.time_col, self.MEDIUM_WIN, "std")
            df[f"{feat}_ewma"] = rolling_per_node(
                df, feat, self.node_col, self.time_col, self.SHORT_WIN, "mean")
        return df

    # ── B. Source / Destination Entropy Features ──────────────────────────────
    def _add_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling entropy delta — key DoS vs DDoS discriminator:
          - DoS:  src_ip_entropy drops sharply (single source)
          - DDoS: src_ip_entropy spikes then remains elevated (many sources)
        """
        if "src_ip_entropy" in df.columns:
            df["src_entropy_delta"]     = diff_per_node(df, "src_ip_entropy",
                                            self.node_col, self.time_col)
            df["src_entropy_drop"]      = df["src_entropy_delta"].clip(upper=0).abs()
            df["src_entropy_spike"]     = df["src_entropy_delta"].clip(lower=0)
            df["src_entropy_roll_min_{self.MEDIUM_WIN}s"] = rolling_per_node(
                df, "src_ip_entropy", self.node_col, self.time_col,
                self.MEDIUM_WIN, "min")

            # Sustained low entropy (DoS) indicator: rolling min stays low
            low_entropy_thr = q_thresh(df, "src_ip_entropy", 0.10)  # 10th pct of normal
            df["low_src_entropy_flag"] = (
                rolling_per_node(df, "src_ip_entropy",
                                 self.node_col, self.time_col, self.SHORT_WIN, "mean")
                < low_entropy_thr
            ).astype(int)

            # High entropy sustained (DDoS) indicator
            high_entropy_thr = q_thresh(df, "src_ip_entropy", 0.99)
            df["high_src_entropy_flag"] = (
                rolling_per_node(df, "src_ip_entropy",
                                 self.node_col, self.time_col, self.SHORT_WIN, "mean")
                > high_entropy_thr
            ).astype(int)

        if "dst_ip_entropy" in df.columns:
            df["dst_entropy_delta"] = diff_per_node(df, "dst_ip_entropy",
                                          self.node_col, self.time_col)

        if "src_ip_entropy" in df.columns and "dst_ip_entropy" in df.columns:
            # Entropy asymmetry: DDoS targets few destinations from many sources
            df["entropy_asymmetry"] = df["src_ip_entropy"] - df["dst_ip_entropy"]
        return df

    # ── C. SYN Flood CUSUM ────────────────────────────────────────────────────
    def _add_syn_cusum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CUSUM accumulator for upward shifts in SYN packet rate and
        half-open connection count.
        Primary TCP SYN flood and connection exhaustion detectors.
        """
        cusum_targets = [
            "syn_packet_rate",
            "half_open_connections",
            "connection_attempts_per_sec",
            "packets_per_second",
        ]
        df_sorted = df.sort_values([self.node_col, self.time_col]).copy()

        for feat in cusum_targets:
            if feat not in df.columns:
                continue
            cusum_vals  = np.zeros(len(df_sorted))
            cusum_flags = np.zeros(len(df_sorted), dtype=int)

            for _, group in df_sorted.groupby(self.node_col, sort=False):
                idx    = group.index.tolist()
                vals   = pd.to_numeric(group[feat], errors="coerce").fillna(0.0).values
                warmup = min(self.WARMUP, max(3, len(vals) // 5))
                mu     = np.mean(vals[:warmup])
                sigma  = np.std(vals[:warmup]) + 1e-6
                k      = self.CUSUM_K * sigma
                H      = self.CUSUM_H  * sigma
                S = 0.0
                for i in range(len(idx)):
                    S = max(0.0, S + (vals[i] - mu - k))
                    loc = df_sorted.index.get_loc(idx[i])
                    cusum_vals[loc]  = S
                    cusum_flags[loc] = int(S > H)

            df_sorted[f"{feat}_cusum"]      = cusum_vals
            df_sorted[f"{feat}_cusum_flag"] = cusum_flags

        for col in df_sorted.columns:
            if "_cusum" in col:
                df[col] = df_sorted[col]
        return df

    # ── D. Protocol Distribution Features ─────────────────────────────────────
    def _add_protocol_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Protocol proportion features.
        A sudden shift toward TCP SYN or ICMP is a classic DoS/DDoS signature.
        """
        total_pkt = df.get("packets_per_second", pd.Series(1.0, index=df.index))
        total_pkt = total_pkt.replace(0, np.nan)

        if "syn_packet_rate" in df.columns:
            df["syn_fraction"]  = safe_div(df["syn_packet_rate"],  total_pkt, clip_max=1.0)
        if "icmp_packet_rate" in df.columns:
            df["icmp_fraction"] = safe_div(df["icmp_packet_rate"], total_pkt, clip_max=1.0)
        if "udp_packet_rate" in df.columns:
            df["udp_fraction"]  = safe_div(df["udp_packet_rate"],  total_pkt, clip_max=1.0)
        if "fin_rst_rate" in df.columns:
            df["fin_rst_fraction"] = safe_div(df["fin_rst_rate"], total_pkt, clip_max=1.0)

        # Protocol shift rate (delta of SYN fraction — sudden increases are alarming)
        if "syn_fraction" in df.columns:
            df["syn_fraction_delta"] = diff_per_node(
                df, "syn_fraction", self.node_col, self.time_col, abs_val=True)

        # SYN dominance: SYN fraction rolling max — remains high during ongoing flood
        if "syn_fraction" in df.columns:
            df["syn_dominance"] = rolling_per_node(
                df, "syn_fraction", self.node_col, self.time_col, self.SHORT_WIN, "max")
        return df

    # ── E. Connection Exhaustion Features ─────────────────────────────────────
    def _add_connection_exhaustion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features capturing resource-exhaustion patterns:
          - Accumulating half-open connections
          - Error rate growth (server refusing connections)
          - Half-open acceleration
        """
        if "half_open_connections" in df.columns:
            # Acceleration of half-open growth
            df["half_open_accel"] = diff_per_node(
                df, "half_open_connections", self.node_col, self.time_col)
            # Persistent half-open: rolling mean stays high
            df["half_open_roll_mean"] = rolling_per_node(
                df, "half_open_connections",
                self.node_col, self.time_col, self.MEDIUM_WIN, "mean")

        if "error_response_rate" in df.columns:
            # Rising error rate = server becoming overwhelmed
            df["error_rate_delta"]    = diff_per_node(
                df, "error_response_rate", self.node_col, self.time_col, abs_val=True)
            df["error_rate_roll_mean"] = rolling_per_node(
                df, "error_response_rate",
                self.node_col, self.time_col, self.MEDIUM_WIN, "mean")
            # High error rate confirmation flag
            err_thr = q_thresh(df, "error_response_rate", 0.99)
            df["high_error_rate_flag"] = (
                df["error_rate_roll_mean"] > err_thr
            ).astype(int)

        # Source count acceleration (DDoS: new bots joining a botnet)
        if "unique_source_count" in df.columns:
            df["source_count_accel"] = diff_per_node(
                df, "unique_source_count", self.node_col, self.time_col)
            df["source_count_roll_max"] = rolling_per_node(
                df, "unique_source_count",
                self.node_col, self.time_col, self.MEDIUM_WIN, "max")
        return df

    # ── F. Threshold Flags ────────────────────────────────────────────────────
    def _add_threshold_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Temporally confirmed binary flags for DoS/DDoS indicators.
        Mirrors suspicious_* flags from original.
        """
        confirm_win = 3
        thresholds = {
            "packets_per_second"         : (q_thresh(df, "packets_per_second",         0.99), "above"),
            "bytes_per_second"           : (q_thresh(df, "bytes_per_second",           0.99), "above"),
            "unique_source_count"        : (q_thresh(df, "unique_source_count",        0.99), "above"),
            "half_open_connections"      : (q_thresh(df, "half_open_connections",      0.99), "above"),
            "syn_packet_rate"            : (q_thresh(df, "syn_packet_rate",            0.99), "above"),
            "connection_attempts_per_sec": (q_thresh(df, "connection_attempts_per_sec",0.99), "above"),
        }

        df_sorted = df.sort_values([self.node_col, self.time_col])
        for feat, (thr, direction) in thresholds.items():
            if feat not in df.columns:
                continue
            raw_flag_name  = f"{feat}_raw_flag"
            conf_flag_name = f"{feat}_flag"
            df[raw_flag_name] = (df[feat] > thr).astype(int) if direction == "above" \
                                else (df[feat] < thr).astype(int)
            conf_result = pd.Series(0, index=df.index)
            for _, group in df_sorted.groupby(self.node_col, sort=False):
                raw  = df.loc[group.index, raw_flag_name]
                conf = raw.rolling(confirm_win, min_periods=confirm_win).sum() >= confirm_win
                conf_result.loc[group.index] = conf.fillna(False).astype(int).values
            df[conf_flag_name] = conf_result

        df.drop(columns=[c for c in df.columns if c.endswith("_raw_flag")], inplace=True)

        # Composite DoS score
        dos_flag_cols = [c for c in df.columns if c.endswith("_flag") and
                         any(f in c for f in DOS_FEAT_COLS)]
        df["dos_threshold_score"] = df[dos_flag_cols].sum(axis=1)
        return df

    # ── Public API ─────────────────────────────────────────────────────────────
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full temporal feature pipeline for DoS/DDoS detection."""
        df = safe_numeric_df(df, [c for c in DOS_FEAT_COLS if c in df.columns])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values([self.node_col, self.time_col]).reset_index(drop=True)

        print("  [TemporalDoS] Traffic burst features...")
        df = self._add_traffic_burst_features(df)
        print("  [TemporalDoS] Entropy features...")
        df = self._add_entropy_features(df)
        print("  [TemporalDoS] SYN CUSUM accumulators...")
        df = self._add_syn_cusum(df)
        print("  [TemporalDoS] Protocol distribution...")
        df = self._add_protocol_distribution(df)
        print("  [TemporalDoS] Connection exhaustion...")
        df = self._add_connection_exhaustion(df)
        print("  [TemporalDoS] Threshold flags...")
        df = self._add_threshold_flags(df)

        new_cols = [c for c in df.columns if c not in DOS_FEAT_COLS + ["label", "timestamp", "node_id", "node_type", "collection_point", "window_id"]]
        print(f"  [TemporalDoS] Done — added {len(new_cols)} temporal features")
        return df


# ── Build temporal DoS/DDoS features ─────────────────────────────────────────


class CrossLayerFeatureBuilder:
    """
    Builds cross-layer features by fusing signals from Edge (Physical/MAC)
    and GCC (Network/Transport) collection points.

    Handles three cases:
      1. Edge-only rows  (jamming pipeline): net features are NaN → jam cross-layer features
      2. GCC-only rows   (DoS pipeline):     phy features are NaN → dos cross-layer features
      3. Hybrid rows     (Scenario 3):       all features present → full cross-layer fusion
    """

    def __init__(self):
        pass

    # ── A. Jam-Congestion Coupling ─────────────────────────────────────────────
    def _jam_congestion_coupling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects Scenario 3: jamming causes physical degradation that
        simultaneously amplifies network congestion.

        Coupling signal: high retransmissions (MAC) co-occurring with high PPS (L3).
        Both layers being abnormal simultaneously is a strong hybrid attack indicator.
        """
        # Normalised retransmission severity [0, 1]
        retx_max = df["retransmission_count"].quantile(0.99) if "retransmission_count" in df.columns else 1.0
        retx_norm = safe_div(df.get("retransmission_count", pd.Series(0.0, index=df.index)),
                             pd.Series(max(retx_max, 1.0), index=df.index), clip_max=1.0)

        # Normalised PPS severity [0, 1]
        pps_max = df["packets_per_second"].quantile(0.99) if "packets_per_second" in df.columns else 1.0
        pps_norm = safe_div(df.get("packets_per_second", pd.Series(0.0, index=df.index)),
                            pd.Series(max(pps_max, 1.0), index=df.index), clip_max=1.0)

        # Coupling index: geometric mean (both must be high for index to be high)
        df["jam_congestion_coupling"] = np.sqrt(retx_norm * pps_norm)

        # Binary coupling flag: both physical and network anomalies present
        phy_flag = df.get("jam_threshold_score", pd.Series(0, index=df.index)) >= 2
        net_flag = df.get("dos_threshold_score", pd.Series(0, index=df.index)) >= 2
        df["cross_layer_attack_flag"] = (phy_flag & net_flag).astype(int)
        return df

    # ── B. Physical-Logical Loss Correlation ──────────────────────────────────
    def _physical_logical_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correlates packet loss at the MAC layer with error responses at L7.
        High correlation between PDR drop and error_response_rate rise indicates
        that physical jamming is propagating up the stack.
        """
        # Loss at MAC: 1 - PDR
        pdr = df.get("packet_delivery_ratio", pd.Series(1.0, index=df.index))
        df["mac_loss_rate"] = (1.0 - pdr).clip(0, 1)

        # L7 error rate (already in [0,1])
        err = df.get("error_response_rate", pd.Series(0.0, index=df.index))

        # Cross-layer loss index: both layers showing loss simultaneously
        df["cross_layer_loss_index"] = np.sqrt(df["mac_loss_rate"] * err.clip(0, 1))

        # PDR-error divergence: when L7 error is high but PDR is fine → pure DoS
        # When both are high → physical degradation propagating to app layer (jamming+)
        df["phy_app_loss_divergence"] = (err.clip(0, 1) - df["mac_loss_rate"]).abs()
        return df

    # ── C. Noise-Traffic Correlation ──────────────────────────────────────────
    def _noise_traffic_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        In Scenario 3, the attacker uses the RF jammer to disrupt ACKs,
        causing retransmissions that appear as legitimate traffic spikes at L3.
        Feature: rising noise floor concurrent with rising PPS.
        """
        # Noise floor deviation from normal (positive = noise elevated)
        nf_dev = df.get("noise_floor_drift_db",
                        pd.Series(0.0, index=df.index)).clip(lower=0)
        # PPS burst index (already computed if TemporalDoSFeatureBuilder ran)
        pps_burst = df.get("packets_per_second_burst_index",
                           pd.Series(1.0, index=df.index)).clip(lower=0)

        # Normalise noise deviation
        nf_max = nf_dev.quantile(0.99) if nf_dev.max() > 0 else 1.0
        nf_norm = (nf_dev / max(nf_max, 1e-6)).clip(0, 1)

        # Normalise PPS burst (clip at 10x normal)
        pps_norm = ((pps_burst - 1.0).clip(lower=0) / 9.0).clip(0, 1)

        df["noise_traffic_coupling"] = np.sqrt(nf_norm * pps_norm)
        return df

    # ── D. Cross-Layer Consistency Score ──────────────────────────────────────
    def _cross_layer_consistency_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Composite cross-layer anomaly score for the CNN+LSTM model.
        Weighted combination of the three coupling indices.

        Score > threshold → triggers cross-layer detection path in Detection Agent.
        Mirrors gps_risk_score + routing_risk_score fusion from original.
        """
        w1, w2, w3 = 0.40, 0.35, 0.25   # jam_congestion, phy_app_loss, noise_traffic
        score = (
            w1 * df.get("jam_congestion_coupling",  pd.Series(0.0, index=df.index)) +
            w2 * df.get("cross_layer_loss_index",   pd.Series(0.0, index=df.index)) +
            w3 * df.get("noise_traffic_coupling",    pd.Series(0.0, index=df.index))
        ).clip(0, 1)
        df["cross_layer_anomaly_score"] = score

        # Severity tiers (calibrated from normal rows)
        cl_thr_medium = q_thresh(df, "cross_layer_anomaly_score", 0.95)
        cl_thr_high   = q_thresh(df, "cross_layer_anomaly_score", 0.995)
        df["cross_layer_severity"] = np.select(
            [score >= cl_thr_high, score >= cl_thr_medium],
            ["HIGH",               "MEDIUM"],
            default="LOW"
        )
        return df

    # ── E. Rule-Based Scenario Tag ─────────────────────────────────────────────
    def _add_scenario_tag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preliminary scenario classification from cross-layer evidence.
        Used as a soft prior for the CNN+LSTM and as a human-readable audit field.
        """
        jam_score = df.get("jam_threshold_score",       pd.Series(0, index=df.index))
        dos_score = df.get("dos_threshold_score",       pd.Series(0, index=df.index))
        cl_score  = df.get("cross_layer_anomaly_score", pd.Series(0.0, index=df.index))
        cl_flag   = df.get("cross_layer_attack_flag",   pd.Series(0, index=df.index))
        src_count = df.get("unique_source_count",       pd.Series(1, index=df.index))

        conditions = [
            (cl_flag == 1) & (cl_score > 0.3),           # both layers active
            (jam_score >= 3) & (dos_score < 1),            # jamming dominant
            (dos_score >= 3) & (src_count >= 50),           # DDoS (many sources)
            (dos_score >= 3) & (src_count < 50),            # DoS (few sources)
            (jam_score >= 1) | (dos_score >= 1),            # weak signal — suspicious
        ]
        tags = [
            "HYBRID_ATTACK",
            "JAMMING",
            "DDOS",
            "DOS",
            "SUSPICIOUS",
        ]
        df["scenario_tag"] = np.select(conditions, tags, default="NORMAL")
        return df

    # ── Public API ─────────────────────────────────────────────────────────────
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build cross-layer features on a DataFrame that may have either or both
        of jam_threshold_score and dos_threshold_score already computed."""
        print("  [CrossLayer] Jam-congestion coupling...")
        df = self._jam_congestion_coupling(df)
        print("  [CrossLayer] Physical-logical loss correlation...")
        df = self._physical_logical_loss(df)
        print("  [CrossLayer] Noise-traffic coupling...")
        df = self._noise_traffic_correlation(df)
        print("  [CrossLayer] Cross-layer consistency score...")
        df = self._cross_layer_consistency_score(df)
        print("  [CrossLayer] Scenario tag...")
        df = self._add_scenario_tag(df)
        print("  [CrossLayer] Done")
        return df


# ── Apply cross-layer builder to the full (combined) dataset ──────────────────
# First, merge temporal features back into the combined dataframe

# For edge rows: carry over temporal jamming features

# Bring edge temporal features in for rows that are in edge_feat_df

# Bring GCC temporal features in for rows that are in gcc_feat_df


# Apply cross-layer builder


class SwarmConsensusFeatureBuilder:
    """
    Computes swarm-level consensus features.
    Groups nodes by 1-second time buckets and computes peer deviations.
    Mirrors the swarm consensus block from build_enhanced_consistency_features().
    """

    def __init__(self, time_col: str = "timestamp", node_col: str = "node_id"):
        self.time_col = time_col
        self.node_col = node_col

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
        df["_bucket"]     = df[self.time_col].dt.floor("1s")

        # ── Swarm-level aggregates (vectorized groupby transform) ──────────────
        swarm_agg_feats = {
            # Jamming layer: consensus signal quality
            "rssi_dbm"                   : "median",
            "snr_db"                     : "median",
            "channel_occupancy_pct"      : "median",
            "mac_retry_count"            : "median",
            # DoS/DDoS layer: consensus traffic
            "packets_per_second"         : "median",
            "unique_source_count"        : "median",
            "src_ip_entropy"             : "median",
            "half_open_connections"      : "median",
        }
        for feat, agg_fn in swarm_agg_feats.items():
            if feat not in df.columns:
                continue
            consensus_col = f"swarm_{feat}_consensus"
            df[consensus_col] = df.groupby("_bucket")[feat].transform(agg_fn)

            # Deviation: how far is this node from swarm consensus?
            dev_col = f"{feat}_swarm_dev"
            df[dev_col] = (df[feat] - df[consensus_col]).abs()

            # Z-score-like relative deviation (normalised by swarm std)
            std_col = f"_swarm_{feat}_std"
            df[std_col] = df.groupby("_bucket")[feat].transform("std").fillna(1.0)
            df[f"{feat}_swarm_zscore"] = safe_div(df[dev_col], df[std_col], fill=0.0, clip_max=10.0)
            df.drop(columns=[std_col], inplace=True)

        # Neighbor count per bucket
        df["neighbor_count"] = df.groupby("_bucket")[self.node_col].transform("count") - 1

        # ── Consensus anomaly flags ────────────────────────────────────────────
        # Node is isolated outlier: its signal is MUCH worse than all peers
        if "rssi_dbm_swarm_dev" in df.columns:
            rssi_dev_thr = q_thresh(df, "rssi_dbm_swarm_dev", 0.99)
            df["isolated_signal_outlier"] = (
                df["rssi_dbm_swarm_dev"] > rssi_dev_thr
            ).astype(int)

        # Coordinated attack: ALL nodes in bucket show high PPS simultaneously
        if "packets_per_second" in df.columns:
            bucket_min_pps = df.groupby("_bucket")["packets_per_second"].transform("min")
            pps_thr = q_thresh(df, "packets_per_second", 0.99)
            df["coordinated_flood_flag"] = (
                (bucket_min_pps > pps_thr) & (df["neighbor_count"] >= 2)
            ).astype(int)

        # Entropy divergence across swarm: peers disagree on source diversity
        # (can indicate source IP spoofing viewed differently per node)
        if "src_ip_entropy_swarm_dev" in df.columns:
            entropy_dev_thr = q_thresh(df, "src_ip_entropy_swarm_dev", 0.99)
            df["swarm_entropy_divergence_flag"] = (
                df["src_ip_entropy_swarm_dev"] > entropy_dev_thr
            ).astype(int)

        # ── Swarm consensus anomaly score (mirrors neighbor_consensus_error_m) ──
        # Weighted combination of deviation z-scores across swarm-level features
        score_components = []
        for feat in ["rssi_dbm", "snr_db", "packets_per_second", "src_ip_entropy"]:
            zs_col = f"{feat}_swarm_zscore"
            if zs_col in df.columns:
                score_components.append(df[zs_col].clip(0, 5) / 5.0)
        if score_components:
            df["swarm_consensus_anomaly_score"] = (
                sum(score_components) / len(score_components)
            ).clip(0, 1)
        else:
            df["swarm_consensus_anomaly_score"] = 0.0

        # Thresholds for detection agent
        swarm_thr = q_thresh(df, "swarm_consensus_anomaly_score", 0.95)
        df["swarm_anomaly_flag"] = (df["swarm_consensus_anomaly_score"] > swarm_thr).astype(int)

        df.drop(columns=["_bucket"], inplace=True)
        print(f"  [Swarm] Consensus features added")
        return df


# ── Apply swarm features ──────────────────────────────────────────────────────


def select_features_by_mutual_info(
    df: pd.DataFrame,
    candidate_cols: list,
    label_col: str = "label",
    top_k: int = None,
    min_mi: float = 0.01,
) -> list:
    """
    Rank candidate features by mutual information with the label.
    Returns the top_k most informative features (or all with MI >= min_mi).
    Mirrors the implicit feature selection in the original's FEATURE_COLUMNS.
    """
    available = [c for c in candidate_cols if c in df.columns]
    if label_col not in df.columns or len(available) == 0:
        return available

    X = df[available].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[label_col].map(LABEL_MAP).fillna(0).astype(int)

    # Remove zero-variance columns
    var_mask = X.std() > 1e-9
    X = X.loc[:, var_mask]
    available = list(X.columns)

    if len(available) == 0:
        return []

    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    mi_series = pd.Series(mi_scores, index=available).sort_values(ascending=False)

    if top_k is not None:
        selected = mi_series.head(top_k).index.tolist()
    else:
        selected = mi_series[mi_series >= min_mi].index.tolist()

    print(f"  MI selection: {len(available)} candidates → {len(selected)} selected (min_mi={min_mi})")
    for feat, score in mi_series.head(15).items():
        print(f"    {feat:<50s}  MI={score:.4f}")
    return selected


# ── Jamming feature set ────────────────────────────────────────────────────────

# Candidates: base + all temporal jamming features built in Cell 4


# ── DoS/DDoS feature set ───────────────────────────────────────────────────────


# ── Hybrid / cross-layer feature set ──────────────────────────────────────────


class OnlineEWMA:
    """
    Stateful per-metric EWMA anomaly detector.
    Suitable for deployment on resource-constrained UAV/UGV nodes.

    Parameters
    ----------
    alpha       : smoothing factor (0 < alpha <= 1); higher = more reactive
    k_sigma     : alarm threshold in units of running std
    warmup      : number of windows before alarming (initialisation phase)
    """

    def __init__(self, alpha: float = 0.3, k_sigma: float = 3.0, warmup: int = 10):
        self.alpha    = alpha
        self.k_sigma  = k_sigma
        self.warmup   = warmup
        self._ewma    : dict = {}   # running EWMA per feature
        self._ewma_sq : dict = {}   # running EWMA of x^2 (for variance)
        self._n       : dict = {}   # window count per feature

    def _running_std(self, feat: str) -> float:
        """Variance from E[X^2] - E[X]^2, numerically stable."""
        var = max(0.0, self._ewma_sq.get(feat, 0.0) - self._ewma.get(feat, 0.0) ** 2)
        return math.sqrt(var) + 1e-9

    def update(self, feat: str, value: float) -> dict:
        """
        Update EWMA state with one new observation.
        Returns a result dict with current stats and alarm flag.
        """
        a = self.alpha
        if feat not in self._ewma:
            self._ewma[feat]    = value
            self._ewma_sq[feat] = value ** 2
            self._n[feat]       = 1
            return {"ewma": value, "std": 0.0, "z_score": 0.0, "alarm": 0}

        self._ewma[feat]    = a * value + (1 - a) * self._ewma[feat]
        self._ewma_sq[feat] = a * value ** 2 + (1 - a) * self._ewma_sq[feat]
        self._n[feat]      += 1

        std   = self._running_std(feat)
        z     = abs(value - self._ewma[feat]) / std
        alarm = int(z > self.k_sigma and self._n[feat] >= self.warmup)

        return {
            "ewma"   : self._ewma[feat],
            "std"    : std,
            "z_score": round(z, 4),
            "alarm"  : alarm,
        }

    def batch_update(self, window: dict) -> dict:
        """Process a full feature window; returns per-feature results and a combined alarm."""
        results  = {}
        n_alarms = 0
        for feat, val in window.items():
            if not isinstance(val, (int, float)) or math.isnan(float(val)):
                continue
            res = self.update(feat, float(val))
            results[feat]  = res
            n_alarms      += res["alarm"]
        results["_n_alarms"] = n_alarms
        results["_alarm"]    = int(n_alarms >= 2)   # require >= 2 features alarming
        return results

    def reset(self):
        """Reset all state (e.g. after confirmed attack / node reboot)."""
        self._ewma.clear(); self._ewma_sq.clear(); self._n.clear()


class OnlineCUSUM:
    """
    Stateful per-metric CUSUM change-point detector.

    Calibrates mu and sigma from the first `warmup` observations.
    After calibration, accumulates S+ (upward) or S- (downward) depending
    on the monitored direction, and fires when |S| > H = h_factor * sigma.
    Resets S to 0 after each alarm to detect subsequent changes.
    """

    def __init__(self, k_factor: float = 0.5, h_factor: float = 5.0,
                 warmup: int = 10, direction: str = "both"):
        self.k_factor  = k_factor
        self.h_factor  = h_factor
        self.warmup    = warmup
        self.direction = direction   # "up" | "down" | "both"
        self._buf    : dict = {}
        self._mu     : dict = {}
        self._sigma  : dict = {}
        self._k      : dict = {}
        self._H      : dict = {}
        self._S_up   : dict = {}
        self._S_dn   : dict = {}
        self._n      : dict = {}

    def _calibrate(self, feat: str):
        vals = self._buf[feat]
        self._mu[feat]    = float(np.mean(vals))
        self._sigma[feat] = float(np.std(vals)) + 1e-6
        self._k[feat]     = self.k_factor * self._sigma[feat]
        self._H[feat]     = self.h_factor * self._sigma[feat]
        self._S_up[feat]  = 0.0
        self._S_dn[feat]  = 0.0

    def update(self, feat: str, value: float) -> dict:
        """Update CUSUM state; returns S values and alarm flag."""
        val = float(value)
        if feat not in self._n:
            self._n[feat] = 0; self._buf[feat] = []
        if self._n[feat] < self.warmup:
            self._buf[feat].append(val)
            self._n[feat] += 1
            if self._n[feat] == self.warmup:
                self._calibrate(feat)
            return {"S_up": 0.0, "S_dn": 0.0, "alarm": 0, "phase": "warmup"}

        mu = self._mu[feat]; k = self._k[feat]; H = self._H[feat]

        if self.direction in ("up", "both"):
            self._S_up[feat] = max(0.0, self._S_up[feat] + val - mu - k)
        if self.direction in ("down", "both"):
            self._S_dn[feat] = min(0.0, self._S_dn[feat] + val - mu + k)

        S_up  = self._S_up[feat]
        S_dn  = self._S_dn[feat]
        alarm = int(S_up > H or abs(S_dn) > H)

        if alarm:
            self._S_up[feat] = 0.0
            self._S_dn[feat] = 0.0

        return {"S_up": round(S_up, 4), "S_dn": round(S_dn, 4),
                "H": round(H, 4), "alarm": alarm, "phase": "active"}

    def batch_update(self, window: dict) -> dict:
        """Process a full feature window."""
        results = {}; n_alarms = 0
        for feat, val in window.items():
            if not isinstance(val, (int, float)) or math.isnan(float(val)):
                continue
            res = self.update(feat, float(val))
            results[feat] = res; n_alarms += res["alarm"]
        results["_n_alarms"] = n_alarms
        results["_alarm"]    = int(n_alarms >= 1)
        return results

    def reset(self, feat: str = None):
        if feat:
            self._S_up[feat] = self._S_dn[feat] = 0.0
        else:
            self._S_up.clear(); self._S_dn.clear()
            self._buf.clear();  self._n.clear()


class EdgeLightweightDetector:
    """
    Combined EWMA + CUSUM detector for deployment on a single UAV/UGV node.
    Processes one 1-second feature window at a time.

    EWMA monitors : RSSI, SNR, SINR, PDR (signal quality metrics)
    CUSUM monitors: SNR (downward), mac_retry_count (upward), cca_failure_count (upward)

    Jamming alarm is raised when:
      - EWMA fires on >= 2 signal metrics, OR
      - CUSUM fires on >= 1 metric AND EWMA fires on >= 1 metric (dual confirmation)
    """

    EWMA_FEATS  = ["rssi_dbm", "snr_db", "sinr_db",
                   "packet_delivery_ratio", "channel_occupancy_pct", "bit_error_rate"]
    CUSUM_FEATS = {
        "snr_db"               : "down",
        "sinr_db"              : "down",
        "mac_retry_count"      : "up",
        "cca_failure_count"    : "up",
        "channel_occupancy_pct": "up",
    }

    def __init__(self, node_id: str, ewma_alpha: float = 0.3,
                 ewma_k: float = 3.0, cusum_k: float = 0.5,
                 cusum_h: float = 5.0, warmup: int = 10):
        self.node_id  = node_id
        self.ewma     = OnlineEWMA(alpha=ewma_alpha, k_sigma=ewma_k, warmup=warmup)
        self.cusums   = {
            feat: OnlineCUSUM(k_factor=cusum_k, h_factor=cusum_h,
                              warmup=warmup, direction=direction)
            for feat, direction in self.CUSUM_FEATS.items()
        }
        self._window_count  = 0
        self._alarm_history = deque(maxlen=30)

    def process_window(self, window: dict) -> dict:
        """
        Process one feature window and return a detection result.
        Parameters : window — dict of feature name to scalar value
        Returns    : dict with node_id, window_id, ewma_alarm, cusum_alarm,
                     jam_alarm, consecutive_alarms, confidence, severity
        """
        self._window_count += 1
        ts = window.get("timestamp", datetime.now(timezone.utc).isoformat())

        # EWMA pass
        ewma_input  = {f: window[f] for f in self.EWMA_FEATS if f in window}
        ewma_result = self.ewma.batch_update(ewma_input)
        ewma_alarm  = ewma_result.get("_alarm", 0)

        # CUSUM pass
        cusum_alarms = 0; cusum_detail = {}
        for feat, cusum in self.cusums.items():
            if feat not in window:
                continue
            res = cusum.update(feat, window[feat])
            cusum_detail[feat] = res
            cusum_alarms      += res["alarm"]
        cusum_alarm = int(cusum_alarms >= 1)

        # Combined jamming alarm
        jam_alarm = int(
            ewma_alarm == 1 or
            (cusum_alarm == 1 and ewma_result.get("_n_alarms", 0) >= 1)
        )

        self._alarm_history.append(jam_alarm)
        consecutive = int(sum(self._alarm_history))

        total_metrics   = len(self.EWMA_FEATS) + len(self.CUSUM_FEATS)
        alarmed_metrics = ewma_result.get("_n_alarms", 0) + cusum_alarms
        confidence = round(min(alarmed_metrics / max(total_metrics, 1), 1.0), 4)

        return {
            "node_id"            : self.node_id,
            "window_id"          : self._window_count,
            "timestamp"          : ts,
            "ewma_alarm"         : ewma_alarm,
            "cusum_alarm"        : cusum_alarm,
            "jam_alarm"          : jam_alarm,
            "consecutive_alarms" : consecutive,
            "confidence"         : confidence,
            "severity"           : ("HIGH" if consecutive >= 10 else
                                    "MEDIUM" if consecutive >= 3 else "LOW")
                                    if jam_alarm else "NONE",
            "ewma_details"       : {k: v for k, v in ewma_result.items()
                                    if not k.startswith("_")},
            "cusum_details"      : cusum_detail,
        }


# -- Demonstration on edge_feat_df first 50 windows --


def jitter_sequences(X: np.ndarray, noise_std: float = 0.02) -> np.ndarray:
    """Add small Gaussian noise to sequence arrays for augmentation."""
    return (X + np.random.normal(0, noise_std, X.shape)).astype(np.float32).clip(0, 1)


def interpolate_sequences(X_a: np.ndarray, X_b: np.ndarray,
                           alpha: float = None) -> np.ndarray:
    """Linear interpolation between two sequence batches."""
    if alpha is None:
        alpha = np.random.uniform(0.3, 0.7)
    return (alpha * X_a + (1 - alpha) * X_b).astype(np.float32)


def prepare_feature_dataframe(
    cleaned_df: pd.DataFrame,
    fe_state: dict = None,
    fit_scalers: bool = False,
    preview: bool = False,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline for a cleaned input DataFrame.
    Steps (mirrors prepare_detection_dataframe from original):
      1. Parse timestamps, ensure node_id present
      2. TemporalJammingFeatureBuilder  (edge rows)
      3. TemporalDoSFeatureBuilder      (gcc rows)
      4. Merge temporal features
      5. CrossLayerFeatureBuilder       (all rows)
      6. SwarmConsensusFeatureBuilder   (all rows)
      7. Feature selection (from fe_state if provided)
      8. Apply scalers (fit if fit_scalers=True, else transform)
      9. Add source_row_id for sequence construction
    """
    import warnings; warnings.filterwarnings("ignore")
    df = cleaned_df.copy()

    # Step 1: basic prep
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    if "node_id"          not in df.columns: df["node_id"]          = "node_001"
    if "collection_point" not in df.columns: df["collection_point"] = "edge"
    if "label"            not in df.columns: df["label"]            = "normal"
    df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)
    df["source_row_id"] = df.index.astype(int)
    if preview: print(f"[FE] Input: {len(df)} rows, {df.shape[1]} cols")

    # Steps 2+3: temporal features split by collection point
    _jam_bld = TemporalJammingFeatureBuilder()
    _dos_bld = TemporalDoSFeatureBuilder()
    edge_mask = df["collection_point"].isin(["edge", "edge_gcc"])
    gcc_mask  = df["collection_point"].isin(["gcc",  "edge_gcc"])

    if edge_mask.any():
        edge_sub = _jam_bld.build(df[edge_mask].copy())
        edge_new = [c for c in edge_sub.columns if c not in df.columns]
        for c in edge_new: df[c] = np.nan
        df.loc[edge_mask, edge_new] = edge_sub[edge_new].values
        if preview: print(f"[FE] Temporal jamming features: {len(edge_new)}")

    if gcc_mask.any():
        gcc_sub = _dos_bld.build(df[gcc_mask].copy())
        gcc_new = [c for c in gcc_sub.columns if c not in df.columns]
        for c in gcc_new:
            if c not in df.columns: df[c] = np.nan
        df.loc[gcc_mask, gcc_new] = gcc_sub[gcc_new].values
        if preview: print(f"[FE] Temporal DoS features: {len(gcc_new)}")

    df = df.fillna(0.0)

    # Steps 4+5: cross-layer and swarm
    df = CrossLayerFeatureBuilder().build(df)
    df = SwarmConsensusFeatureBuilder().build(df)
    if preview: print(f"[FE] After cross-layer + swarm: {df.shape[1]} cols")

    # Steps 6+7: feature selection & scaling
    # When fe_state lacks the selected feature lists (standalone use without a
    # prior fit), fall back to all available engineered jamming / DoS columns.
    _all_jam = [c for c in df.columns if any(f in c for f in JAM_FEAT_COLS)]
    _all_dos = [c for c in df.columns if any(f in c for f in DOS_FEAT_COLS)]
    _jf = (fe_state or {}).get("JAM_FEAT_FINAL",    _all_jam)
    _df = (fe_state or {}).get("DOS_FEAT_FINAL",    _all_dos)
    _hf = (fe_state or {}).get("HYBRID_FEAT_FINAL", list(set(_all_jam + _all_dos)))
    _sj = (fe_state or {}).get("scaler_jam")
    _sd = (fe_state or {}).get("scaler_dos")
    _sh = (fe_state or {}).get("scaler_hybrid")

    for col in _hf:
        if col not in df.columns: df[col] = 0.0
    df = safe_numeric_fill(df, _hf)

    if fit_scalers:
        _sj = fit_scaler_on_normal(df, _jf)
        _sd = fit_scaler_on_normal(df, _df)
        _sh = fit_scaler_on_normal(df, _hf)

    if _sj is not None and _jf: df = apply_scaler(df, _jf, _sj)
    if _sd is not None and _df: df = apply_scaler(df, _df, _sd)
    if _sh is not None and _hf: df = apply_scaler(df, _hf, _sh)

    if preview:
        print(f"[FE] Scaling applied. Final shape: {df.shape}")
        print(f"[FE] Jamming cols : {len(_jf)} | DoS cols: {len(_df)} | Hybrid cols: {len(_hf)}")
    return df


def run_feature_engineering_agent(
    cleaned_df: pd.DataFrame = None,
    csv_path: str = None,
    fe_state_path: str = "feature_engineering_state.pkl",
    fit_scalers: bool = False,
    save_outputs: bool = False,
    preview: bool = True,
) -> tuple:
    """
    Agent-level wrapper for the feature engineering pipeline.
    Mirrors run_collection_cleaning_agent() from the original.
    Returns (feat_df, jam_sequences, dos_sequences, hybrid_sequences, fe_state).
    """
    _fe = None
    if os.path.exists(fe_state_path):
        with open(fe_state_path, "rb") as f:
            _fe = pickle.load(f)
        if preview: print(f"[FE Agent] Loaded state from {fe_state_path}")

    if cleaned_df is None:
        if csv_path is None:
            raise ValueError("Either cleaned_df or csv_path must be provided")
        cleaned_df = pd.read_csv(csv_path)
        if preview: print(f"[FE Agent] Loaded {len(cleaned_df)} rows")

    feat_df = prepare_feature_dataframe(
        cleaned_df, fe_state=_fe, fit_scalers=fit_scalers, preview=preview
    )

    _all_jam = [c for c in feat_df.columns if any(f in c for f in JAM_FEAT_COLS)]
    _all_dos = [c for c in feat_df.columns if any(f in c for f in DOS_FEAT_COLS)]
    _jf = (_fe or {}).get("JAM_FEAT_FINAL",    _all_jam)
    _df = (_fe or {}).get("DOS_FEAT_FINAL",    _all_dos)
    _hf = (_fe or {}).get("HYBRID_FEAT_FINAL", list(set(_all_jam + _all_dos)))
    _sl = (_fe or {}).get("seq_len",            SEQ_LEN)

    jam_seqs = create_sequences(feat_df, [c for c in _jf if c in feat_df.columns], _sl)
    dos_seqs = create_sequences(feat_df, [c for c in _df if c in feat_df.columns], _sl)
    hyb_seqs = create_sequences(feat_df, [c for c in _hf if c in feat_df.columns], _sl)

    if save_outputs:
        feat_df.to_csv("feat_engineered_output.csv", index=False)
        np.save("X_jam_inference.npy",    jam_seqs[0])
        np.save("X_dos_inference.npy",    dos_seqs[0])
        np.save("X_hybrid_inference.npy", hyb_seqs[0])
        if preview: print("[FE Agent] Saved inference arrays + CSV.")

    return feat_df, jam_seqs, dos_seqs, hyb_seqs, _fe


