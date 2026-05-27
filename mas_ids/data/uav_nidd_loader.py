"""
mas_ids.data.uav_nidd_loader
==============================
Drop-in replacement for TrafficDataGenerator that loads the UAV-NIDD dataset.

UAV-NIDD reference:
  Hassan Jalil Hadi, "UAV-NIDD: A Dynamic Dataset for Cybersecurity and
  Intrusion Detection in UAV Networks," IEEE Trans. Network Science and
  Engineering, 2025. DOI: 10.1109/TNSE.2025.3553442
  https://zenodo.org/records/15125851

Usage (replaces TrafficDataGenerator in your pipeline driver):
-----------------------------------------------------------------
  from mas_ids.data.uav_nidd_loader import load_uav_nidd

  cleaned_df = load_uav_nidd(
      csv_path="/kaggle/input/uav-nidd/Normal_Traffic.csv",   # one file
      # -- or --
      csv_dir="/kaggle/input/uav-nidd/",                      # whole folder
      label_col="Label",
      window_seconds=1.0,
  )
  # cleaned_df has the same schema as DataCleaner output:
  #   JAMMING_FEATURES + DOS_FEATURES + metadata cols + 'label'

Then pass cleaned_df directly to the Feature Engineering agent:
  from mas_ids.agents.feature_agent import prepare_feature_dataframe
  feat_df = prepare_feature_dataframe(cleaned_df, fit_scalers=True)

Label mapping from UAV-NIDD → mas_ids schema
--------------------------------------------
  Normal / Benign              → normal
  GPS_Jamming / Jamming        → jamming
  DoS / Flooding               → dos
  DDoS                         → ddos
  GPS_Spoofing                 → jamming   (physical layer attack, same branch)
  Scanning / Reconnaissance    → suspicious (not a separate class in mas_ids)
  MITM / Replay / Evil_Twin    → suspicious
  Brute_Force / Fake_Landing   → suspicious
"""

import os
import glob
import math
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict

# ── Label mapping ──────────────────────────────────────────────────────────────
# UAV-NIDD label strings → mas_ids 5-class schema
# Adjust the keys here if your CSV uses different capitalisation.
LABEL_MAP = {
    # Normal
    "normal": "normal", "benign": "normal", "background": "normal", "": "normal",

    # Jamming / physical-layer attacks
    "gps_jamming": "jamming", "jamming": "jamming",
    "gps_spoofing": "jamming",       # physical-layer → jamming branch
    "wlan_jamming": "jamming",

    # DoS / single-source floods
    "dos": "dos", "flooding": "dos", "dos_flooding": "dos",
    "icmp_flood": "dos", "udp_flood": "dos", "syn_flood": "dos",
    "tcp_flood": "dos",

    # DDoS / coordinated floods
    "ddos": "ddos", "ddos_flooding": "ddos", "distributed_flooding": "ddos",

    # Everything else → suspicious (keep as a valid label; the detection
    # agent will score it accordingly)
    "scanning": "suspicious", "reconnaissance": "suspicious",
    "mitm": "suspicious", "replay": "suspicious", "evil_twin": "suspicious",
    "brute_force": "suspicious", "fake_landing": "suspicious",
    "mitm_attack": "suspicious", "replay_attack": "suspicious",
}


def _norm_label(raw: str) -> str:
    """Normalise a raw Label string to the 5-class mas_ids schema."""
    cleaned = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    return LABEL_MAP.get(cleaned, "suspicious")   # unknown → suspicious


# ── Per-window feature derivation ─────────────────────────────────────────────

def _shannon_entropy(series: pd.Series) -> float:
    counts = series.dropna().value_counts()
    total  = counts.sum()
    if total == 0: return 0.0
    p = counts / total
    return float(-(p * np.log2(p + 1e-12)).sum())


def _derive_window_features(win: pd.DataFrame) -> dict:
    """
    Convert a 1-second packet-level window into one row of derived features
    matching the mas_ids JAMMING_FEATURES + DOS_FEATURES schema.
    """
    n = len(win)
    if n == 0:
        return {}

    # ── Timing ────────────────────────────────────────────────────────────────
    duration = float(win["frame.time_relative"].max() -
                     win["frame.time_relative"].min()) if n > 1 else 1.0
    duration = max(duration, 1e-6)

    # ── Physical layer (jamming features) ─────────────────────────────────────
    rssi_vals = pd.to_numeric(win["radiotap.dbm_antsignal"], errors="coerce").dropna()
    snr_vals  = pd.to_numeric(win["wlan_radio.signal_dbm"],  errors="coerce").dropna()
    rate_vals = pd.to_numeric(win["radiotap.datarate"],       errors="coerce").dropna()
    retry_arr = pd.to_numeric(win["wlan.fc.retry"],           errors="coerce").fillna(0)
    fcs_bad   = pd.to_numeric(win["wlan.fcs.bad_checksum"],   errors="coerce").fillna(0)
    duration_arr = pd.to_numeric(win.get("wlan.duration", pd.Series([0]*n)),
                                 errors="coerce").fillna(0)

    rssi_mean   = float(rssi_vals.mean()) if len(rssi_vals) else -90.0
    snr_mean    = float(snr_vals.mean())  if len(snr_vals)  else 0.0
    rssi_std    = float(rssi_vals.std())  if len(rssi_vals) > 1 else 0.0
    data_rate   = float(rate_vals.mean()) if len(rate_vals)  else 0.0
    mac_retries = float(retry_arr.sum())
    bad_fcs_rate= float(fcs_bad.mean())    # 0–1 fraction of bad frames

    # Channel occupancy proxy: total wlan.duration / window duration (µs → fraction)
    ch_occ = float(duration_arr.sum()) / (duration * 1e6 + 1e-9)
    ch_occ = min(ch_occ, 1.0)

    # BER proxy: bad FCS rate (true BER needs PHY info not captured)
    ber_proxy = bad_fcs_rate

    # ── Network layer (DoS/DDoS features) ─────────────────────────────────────
    frame_lens  = pd.to_numeric(win["frame.len"], errors="coerce").dropna()
    bytes_total = float(frame_lens.sum())
    pps         = n / duration
    bps         = bytes_total / duration
    mean_pkt_sz = float(frame_lens.mean()) if len(frame_lens) else 0.0

    # Source IP entropy
    src_ip_entropy = _shannon_entropy(win["ip.src"].fillna("0.0.0.0"))

    # TTL stats
    ttl_vals   = pd.to_numeric(win["ip.ttl"], errors="coerce").dropna()
    ttl_mean   = float(ttl_vals.mean()) if len(ttl_vals) else 64.0
    ttl_std    = float(ttl_vals.std())  if len(ttl_vals) > 1 else 0.0

    # ARP ratio (ARP packets ÷ total: high ARP = potential flood)
    arp_flag   = pd.to_numeric(win.get("arp", pd.Series([0]*n)), errors="coerce").fillna(0)
    arp_ratio  = float(arp_flag.mean())

    # UDP features
    udp_lens   = pd.to_numeric(win.get("udp.length", pd.Series([0]*n)), errors="coerce").fillna(0)
    udp_pkt_ratio = float((udp_lens > 0).mean())

    # TCP ACK (crude TCP activity indicator; no SYN flag captured)
    tcp_ack   = pd.to_numeric(win.get("tcp.ack", pd.Series([0]*n)), errors="coerce").fillna(0)
    tcp_rate  = float((tcp_ack > 0).sum()) / duration

    # Port entropy (diversity of destination ports → DDoS signature)
    dst_port_entropy = _shannon_entropy(
        win.get("udp.dstport", pd.Series(dtype=object)).fillna("0")
    )

    # Inter-arrival time
    iat_vals = pd.to_numeric(win.get("frame.time_delta_displayed",
                                     pd.Series([0.0]*n)), errors="coerce").dropna()
    iat_mean = float(iat_vals.mean()) if len(iat_vals) else 0.0
    iat_std  = float(iat_vals.std())  if len(iat_vals) > 1 else 0.0

    # seq number variance (replay / jam detection)
    seq_vals = pd.to_numeric(win.get("wlan.seq", pd.Series([0]*n)), errors="coerce").dropna()
    seq_std  = float(seq_vals.std())  if len(seq_vals) > 1 else 0.0

    # Compute a derived "unique source count" and dst-IP entropy from real IPs
    unique_src = int(win["ip.src"].nunique())
    dst_ip_entropy = _shannon_entropy(win.get("ip.dst", pd.Series(dtype=object)).fillna("0.0.0.0"))
    # ICMP rate: ip.proto == 1 is ICMP
    proto_vals = pd.to_numeric(win.get("ip.proto", pd.Series([0]*n)), errors="coerce").fillna(0)
    icmp_rate  = float((proto_vals == 1).sum()) / duration
    udp_rate   = float((udp_lens > 0).sum()) / duration

    return {
        # ════════════════════════════════════════════════════════════════════
        # JAMMING / PHYSICAL+MAC FEATURES — mapped onto schema names so the
        # mutual-information feature selector picks them up.
        # ════════════════════════════════════════════════════════════════════
        "rssi_dbm"               : rssi_mean,
        "snr_db"                 : snr_mean,
        "sinr_db"                : snr_mean,                       # no SINR in capture -> SNR proxy
        "noise_floor_dbm"        : (rssi_mean - snr_mean) if snr_mean != 0 else -90.0,
        "bit_error_rate"         : ber_proxy,                      # from bad-FCS rate
        "bad_packet_ratio"       : bad_fcs_rate,
        "packet_delivery_ratio"  : 1.0 - bad_fcs_rate,
        "mac_retry_count"        : mac_retries,
        "retransmission_count"   : mac_retries,                   # retries == retransmissions in 802.11 capture
        "channel_occupancy_pct"  : ch_occ * 100.0,
        # CCA/backoff are not in WiFi capture; approximate from retry pressure & seq gaps
        "cca_failure_count"      : mac_retries * float(bad_fcs_rate > 0.1),
        "backoff_count"          : seq_std,                        # seq-number variance ~ contention/backoff proxy

        # ════════════════════════════════════════════════════════════════════
        # DoS / DDoS NETWORK+TRANSPORT FEATURES — mapped onto schema names.
        # ════════════════════════════════════════════════════════════════════
        "packets_per_second"     : pps,
        "bytes_per_second"       : bps,
        "unique_source_count"    : float(unique_src),
        "src_ip_entropy"         : src_ip_entropy,
        "dst_ip_entropy"         : dst_ip_entropy,
        "icmp_packet_rate"       : icmp_rate,
        "connection_attempts_per_sec": tcp_rate,                  # TCP activity rate (no SYN flag captured)
        "half_open_connections"  : tcp_rate / (pps + 1e-9) * n,   # best proxy without TCP flags
        "tcp_retransmission_rate": tcp_rate * float(bad_fcs_rate > 0.05),
        "udp_packet_rate"        : udp_rate,
        "syn_packet_rate"        : tcp_rate,                      # TCP-ack rate proxies TCP control traffic
        "port_scan_score"        : dst_port_entropy,              # port diversity ~ scan/DDoS spread

        # Application-layer (L7) features are not present in WiFi-frame capture.
        # They are intentionally omitted (the FE selector will simply not see them),
        # rather than injected as misleading zeros.

        # ════════════════════════════════════════════════════════════════════
        # EXTRA real signals kept under names that the cross-layer / swarm
        # builders and the selector also use (substring-matched on schema names).
        # ════════════════════════════════════════════════════════════════════
        "rssi_dbm_std"           : rssi_std,                      # matches 'rssi_dbm' prefix -> selected
        "data_rate_mbps"         : data_rate,
        "mean_packet_size_bytes" : mean_pkt_sz,
        "ip_ttl_mean"            : ttl_mean,
        "ip_ttl_std"             : ttl_std,
        "iat_mean_s"             : iat_mean,
        "iat_std_s"              : iat_std,
        "arp_ratio"              : arp_ratio,
        "modulation_ofdm_flag"   : float(
            pd.to_numeric(win.get("radiotap.channel.flags.ofdm",
                                  pd.Series([0]*n)), errors="coerce").fillna(0).mean() > 0),
        "modulation_cck_flag"    : float(
            pd.to_numeric(win.get("radiotap.channel.flags.cck",
                                  pd.Series([0]*n)), errors="coerce").fillna(0).mean() > 0),
    }


# ── Main loader ───────────────────────────────────────────────────────────────

def load_uav_nidd(
    csv_path: str   = None,
    csv_dir:  str   = None,
    label_col: str  = "Label",
    window_seconds: float = 1.0,
    node_col: str   = "wlan.bssid",
    max_rows: int   = None,
    attack_frac: float = 0.20,
    verbose: bool   = True,
) -> pd.DataFrame:
    """
    Load UAV-NIDD CSV file(s) and return a DataFrame in mas_ids cleaned_df format.

    Parameters
    ----------
    csv_path       : path to a single CSV file (use this OR csv_dir, not both)
    csv_dir        : directory containing multiple UAV-NIDD CSV files
    label_col      : name of the label column (default "Label")
    window_seconds : aggregation window in seconds (default 1.0, matches SEQ_LEN)
    node_col       : column to use as node_id (default "wlan.bssid")
    max_rows       : optional row limit for quick testing
    attack_frac    : a window is labeled as an attack if >= this fraction of
                     its packets are attack packets (default 0.20). Prevents
                     majority-voting from erasing minority attack classes.
    verbose        : print progress

    Returns
    -------
    pd.DataFrame with the same schema as DataCleaner output:
      - columns matching JAMMING_FEATURES + DOS_FEATURES subsets
      - metadata columns: node_id, node_type, timestamp, label,
                          collection_point, window_id, source_row_id
    """
    # ── Load raw CSV(s) ────────────────────────────────────────────────────────
    if csv_path is not None:
        raw = pd.read_csv(csv_path, low_memory=False, nrows=max_rows)
        if verbose: print(f"[UAV-NIDD] Loaded {len(raw)} rows from {csv_path}")
    elif csv_dir is not None:
        files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {csv_dir}")
        parts = [pd.read_csv(f, low_memory=False) for f in files]
        raw = pd.concat(parts, ignore_index=True)
        if max_rows:
            raw = raw.head(max_rows)
        if verbose: print(f"[UAV-NIDD] Loaded {len(raw)} rows from {len(files)} files in {csv_dir}")
    else:
        raise ValueError("Provide either csv_path or csv_dir")

    # ── Normalise timestamps ───────────────────────────────────────────────────
    raw["frame.time_epoch"]    = pd.to_numeric(raw.get("frame.time_epoch", 0),    errors="coerce").fillna(0)
    raw["frame.time_relative"] = pd.to_numeric(raw.get("frame.time_relative", 0), errors="coerce").fillna(0)
    raw["timestamp"] = pd.to_datetime(raw["frame.time_epoch"], unit="s", utc=True)

    # ── Normalise node_id ──────────────────────────────────────────────────────
    if node_col in raw.columns:
        raw["node_id"] = raw[node_col].astype(str).str.strip().replace("nan", "unknown")
    else:
        raw["node_id"] = "uav_001"

    # ── Map labels ─────────────────────────────────────────────────────────────
    if label_col not in raw.columns:
        raise ValueError(f"Label column '{label_col}' not found. "
                         f"Available: {list(raw.columns)}")
    raw["label_raw"] = raw[label_col].astype(str)
    raw["label"]     = raw["label_raw"].apply(_norm_label)

    label_dist = raw["label"].value_counts().to_dict()
    if verbose:
        print(f"[UAV-NIDD] Label distribution (mapped): {label_dist}")

    # ── Determine collection_point from attack type ────────────────────────────
    # UAV-NIDD has 3 scenarios: compromised UAV, compromised AP, compromised GCS
    # We approximate: jamming rows = edge collection, network rows = gcc collection
    def _collection_point(lbl):
        if lbl in ("jamming",):
            return "edge"          # physical-layer attacks detected at Edge
        elif lbl in ("dos", "ddos"):
            return "gcc"           # network floods detected at GCC
        else:
            return "edge_gcc"      # normal + suspicious captured at both
    raw["collection_point"] = raw["label"].apply(_collection_point)

    # ── Aggregate packets into 1-second windows per node ──────────────────────
    if verbose:
        print(f"[UAV-NIDD] Aggregating into {window_seconds}s windows per node ...")

    raw = raw.sort_values(["node_id", "frame.time_epoch"]).reset_index(drop=True)

    # Assign window bucket: floor(time_epoch / window_seconds)
    raw["_win_bucket"] = (raw["frame.time_epoch"] // window_seconds).astype(int)

    rows = []
    win_id = 0

    for (node_id, bucket), win in raw.groupby(["node_id", "_win_bucket"], sort=False):
        feats = _derive_window_features(win)
        if not feats:
            continue

        # Window label: attack-priority rule (NOT simple majority).
        # Rationale: attacks are often interleaved with normal traffic; a window
        # that contains a meaningful fraction of attack packets is an ATTACK
        # window. Simple majority voting would drown out minority attack classes
        # and starve the detector of positive examples. We label the window with
        # the most frequent ATTACK class if attack packets exceed ATTACK_FRAC of
        # the window; otherwise normal.
        ATTACK_FRAC = attack_frac
        win_labels = win["label"]
        attack_mask = ~win_labels.isin(["normal"])
        attack_frac = float(attack_mask.mean())
        if attack_frac >= ATTACK_FRAC:
            # most common non-normal label in this window
            attack_counts = win_labels[attack_mask].value_counts()
            majority_label = attack_counts.idxmax() if len(attack_counts) else "normal"
        else:
            majority_label = "normal"

        # Representative timestamp = midpoint of window
        t_mid = pd.Timestamp(bucket * window_seconds + window_seconds / 2,
                             unit="s", tz="UTC")

        row = {
            "node_id"         : node_id,
            "node_type"       : "uav",       # UAV-NIDD is UAV-network data
            "timestamp"       : t_mid,
            "label"           : majority_label,
            "collection_point": win["collection_point"].value_counts().idxmax(),
            "window_id"       : win_id,
            "source_row_id"   : win_id,
        }
        row.update(feats)
        rows.append(row)
        win_id += 1

    df = pd.DataFrame(rows)

    # ── Fill any remaining NaNs and clip extreme values ────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = (df[num_cols]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0))

    # Clip extreme outliers (e.g. very high PPS from flood attacks)
    CLIP = {
        "rssi_dbm": (-120, 0),
        "snr_db":   (-30,  50),
        "packets_per_second": (0, 1_000_000),
        "bytes_per_second":   (0, 1e9),
        "channel_occupancy_pct": (0, 100),
        "packet_delivery_ratio": (0, 1),
        "bit_error_rate":        (0, 1),
        "bad_packet_ratio":      (0, 1),
    }
    for col, (lo, hi) in CLIP.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    if verbose:
        print(f"[UAV-NIDD] Output: {df.shape[0]} windows × {df.shape[1]} columns")
        print(f"           Window label dist: {df['label'].value_counts().to_dict()}")

    return df


# ── Convenience: build DataCleaner-compatible split ───────────────────────────

def load_uav_nidd_split(
    csv_path: str = None,
    csv_dir:  str = None,
    label_col: str = "Label",
    window_seconds: float = 1.0,
    max_rows: int = None,
    attack_frac: float = 0.20,
    verbose: bool = True,
):
    """
    Load UAV-NIDD and return (cleaned_df, edge_df, gcc_df, all_df) —
    the exact four variables the pipeline expects after DataCleaner.
    """
    df = load_uav_nidd(csv_path=csv_path, csv_dir=csv_dir,
                       label_col=label_col, window_seconds=window_seconds,
                       max_rows=max_rows, attack_frac=attack_frac, verbose=verbose)
    edge_df = df[df["collection_point"].isin(["edge",     "edge_gcc"])].copy()
    gcc_df  = df[df["collection_point"].isin(["gcc",      "edge_gcc"])].copy()
    return df, edge_df, gcc_df, df.copy()


# ══════════════════════════════════════════════════════════════════════════════
# PER-PACKET LOADER — for short-burst capture datasets where time-windowing
# yields too few samples (e.g. UAV-NIDD DDoS/Normal classes).
#
# Instead of aggregating into 1s windows, each packet becomes one sample. Rolling
# per-node context (last W packets) is added so temporal/EWMA-style features still
# carry signal. Returns the same (cleaned_df, edge_df, gcc_df, all_df) tuple as
# load_uav_nidd_split, so it is drop-in compatible with the pipeline.
# ══════════════════════════════════════════════════════════════════════════════

def load_uav_nidd_perpacket(
    csv_path: str,
    label_col: str = "Label",
    roll_window: int = 20,
    max_rows: int = None,
    verbose: bool = True,
):
    """
    Load a pre-cleaned UAV-NIDD CSV as PER-PACKET samples (no time-windowing).

    Parameters
    ----------
    csv_path    : path to the cleaned CSV (raw packet rows + Label)
    label_col   : label column name (default "Label"). If labels are already in
                  mas_ids schema (normal/jamming/dos/ddos/suspicious) they are kept;
                  otherwise they are mapped via the module LABEL_MAP.
    roll_window : number of preceding packets per node used for rolling features
    max_rows    : optional row cap
    verbose     : print progress

    Returns
    -------
    (cleaned_df, edge_df, gcc_df, all_df) — same contract as load_uav_nidd_split.
    Each row is one packet with derived per-packet + rolling-context features
    matching the mas_ids schema.
    """
    raw = pd.read_csv(csv_path, low_memory=False, nrows=max_rows)
    if verbose:
        print(f"[UAV-NIDD/per-packet] Loaded {len(raw):,} rows from {csv_path}")

    # ── Label normalisation ────────────────────────────────────────────────────
    raw[label_col] = raw[label_col].astype(str).str.strip()
    schema_labels = {"normal", "jamming", "dos", "ddos", "suspicious"}
    if set(raw[label_col].str.lower().unique()) <= schema_labels:
        raw["label"] = raw[label_col].str.lower()
    else:
        raw["label"] = raw[label_col].apply(_norm_label)

    # ── Numeric coercion helpers ───────────────────────────────────────────────
    def num(col):
        return (pd.to_numeric(raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
                if col in raw.columns else pd.Series(0.0, index=raw.index))

    # node_id = bssid if present else single node
    node = raw["wlan.bssid"].astype(str) if "wlan.bssid" in raw.columns else pd.Series("n0", index=raw.index)
    raw["node_id"] = node.fillna("n0").replace("nan", "n0")

    # ── Per-packet base features ────────────────────────────────────────────────
    out = pd.DataFrame(index=raw.index)
    out["node_id"]   = raw["node_id"]
    out["node_type"] = "uav"
    out["timestamp"] = pd.to_datetime(num("frame.time_epoch"), unit="s", utc=True, errors="coerce")
    out["label"]     = raw["label"]

    out["rssi_dbm"]               = num("radiotap.dbm_antsignal")
    out["snr_db"]                 = num("wlan_radio.signal_dbm") if "wlan_radio.signal_dbm" in raw.columns else num("radiotap.dbm_antsignal")
    out["noise_floor_dbm"]        = out["rssi_dbm"] - out["snr_db"]
    out["bad_packet_ratio"]       = num("wlan.fcs.bad_checksum")
    out["bit_error_rate"]         = num("wlan.fcs.bad_checksum")
    out["packet_delivery_ratio"]  = 1.0 - num("wlan.fcs.bad_checksum")
    out["mac_retry_count"]        = num("wlan.fc.retry")
    out["retransmission_count"]   = num("wlan.fc.retry")
    out["data_rate_mbps"]         = num("radiotap.datarate")
    out["channel_occupancy_pct"]  = (num("wlan.duration") / 1000.0).clip(0, 100)
    out["frame_len"]              = num("frame.len")
    out["udp_length"]             = num("udp.length")
    out["ip_ttl"]                 = num("ip.ttl")
    out["iat_s"]                  = num("frame.time_delta_displayed")
    out["arp_flag"]               = num("arp")
    out["modulation_ofdm_flag"]   = (num("radiotap.channel.flags.ofdm") > 0).astype(float)
    out["modulation_cck_flag"]    = (num("radiotap.channel.flags.cck") > 0).astype(float)
    out["is_udp"]                 = (num("udp.length") > 0).astype(float)
    out["is_tcp"]                 = (num("tcp.ack") > 0).astype(float) if "tcp.ack" in raw.columns else 0.0
    out["is_icmp"]                = (pd.to_numeric(raw.get("ip.proto", 0), errors="coerce").fillna(0) == 1).astype(float)

    # ── Rolling per-node context (gives temporal models signal) ─────────────────
    out = out.sort_values(["node_id", "timestamp"]).reset_index(drop=True)
    grp = out.groupby("node_id", sort=False)
    r = roll_window
    out["packets_per_second"]   = grp["iat_s"].transform(lambda s: 1.0 / (s.rolling(r, min_periods=1).mean() + 1e-6))
    out["bytes_per_second"]     = out["packets_per_second"] * grp["frame_len"].transform(lambda s: s.rolling(r, min_periods=1).mean())
    out["mean_packet_size_bytes"] = grp["frame_len"].transform(lambda s: s.rolling(r, min_periods=1).mean())
    out["rssi_roll_std"]        = grp["rssi_dbm"].transform(lambda s: s.rolling(r, min_periods=1).std()).fillna(0.0)
    out["retry_roll_sum"]       = grp["mac_retry_count"].transform(lambda s: s.rolling(r, min_periods=1).sum())
    out["badfcs_roll_mean"]     = grp["bad_packet_ratio"].transform(lambda s: s.rolling(r, min_periods=1).mean())
    out["udp_roll_rate"]        = grp["is_udp"].transform(lambda s: s.rolling(r, min_periods=1).mean())
    out["icmp_roll_rate"]       = grp["is_icmp"].transform(lambda s: s.rolling(r, min_periods=1).mean())
    out["arp_roll_rate"]        = grp["arp_flag"].transform(lambda s: s.rolling(r, min_periods=1).mean())
    out["ttl_roll_std"]         = grp["ip_ttl"].transform(lambda s: s.rolling(r, min_periods=1).std()).fillna(0.0)

    # rolling source-IP diversity (DDoS signal) if ip.src present
    if "ip.src" in raw.columns:
        raw_sorted_src = raw.loc[out.index.map(lambda i: i), "ip.src"] if False else None
    out["unique_source_count"] = 1.0  # per-packet; rolling diversity handled below
    if "ip.src" in raw.columns:
        # Factorize IP strings to integer codes so rolling() can aggregate them,
        # then count distinct codes in each rolling window (nunique proxy).
        src_codes = pd.factorize(raw["ip.src"].astype(str))[0].astype(float)
        tmp = out[["node_id"]].copy()
        tmp["src_code"] = src_codes[:len(tmp)] if len(src_codes) >= len(tmp) else 0.0
        out["unique_source_count"] = (
            tmp.groupby("node_id")["src_code"]
               .transform(lambda s: s.rolling(r, min_periods=1)
                          .apply(lambda x: float(len(np.unique(x))), raw=True))
        ).fillna(1.0)

    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Drop feature columns that are constant-zero for this capture (no signal):
    # noise_floor (signal==noise here), TCP/ICMP flags (not populated in AP capture).
    for _dead in ["noise_floor_dbm", "is_tcp", "is_icmp", "icmp_roll_rate"]:
        if _dead in out.columns and (out[_dead] == 0).all():
            out = out.drop(columns=[_dead])

    # ── collection_point seam (same convention as windowed loader) ──────────────
    out["collection_point"] = out["label"].apply(
        lambda l: "edge" if l == "jamming" else ("gcc" if l in ("dos", "ddos") else "edge_gcc"))
    out["window_id"]   = np.arange(len(out))
    out["source_row_id"] = out["window_id"]

    if verbose:
        print(f"[UAV-NIDD/per-packet] Output: {out.shape[0]:,} packet-samples × {out.shape[1]} cols")
        print(f"           Label dist: {out['label'].value_counts().to_dict()}")

    edge_df = out[out["collection_point"].isin(["edge", "edge_gcc"])].copy()
    gcc_df  = out[out["collection_point"].isin(["gcc",  "edge_gcc"])].copy()
    return out, edge_df, gcc_df, out.copy()
