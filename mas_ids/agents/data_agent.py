"""
mas_ids.agents.data_agent
=========================
Agent 1 — Data Collection & Cleaning.

Defines the feature schema, validity bounds, the synthetic TrafficDataGenerator,
two-tier collectors (Edge UAV/UGV + Gateway/GCC), quality checker, cleaner, and
feature-matrix builder.

Public API:
  JAMMING_FEATURES, DOS_FEATURES, ALL_FEATURES, VALIDITY_BOUNDS
  TrafficDataGenerator, EdgeCollector, GCCCollector,
  DataQualityChecker, DataCleaner, FeatureMatrixBuilder
"""
from datetime import datetime, timezone
from collections import deque, defaultdict
from sklearn.utils import shuffle
import json
import math
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..config import LABEL_MAP, SEQ_LEN, GLOBAL_SEED
from ..utils import shannon_entropy


# ═══════════════════════════════════════════════════════════════════════════════
#  JAMMING DETECTION — Edge (UAV/UGV) Features
# ═══════════════════════════════════════════════════════════════════════════════

# Physical Layer (L1) — raw signal metrics
JAMMING_PHY_FEATURES = [
    "rssi_dbm",              # Received Signal Strength Indicator (dBm)
    "snr_db",                # Signal-to-Noise Ratio (dB)
    "sinr_db",               # Signal-to-Interference-plus-Noise Ratio (dB)
    "noise_floor_dbm",       # Ambient noise / interference floor (dBm)
    "bit_error_rate",        # Bit Error Rate (fraction 0–1)
    "bad_packet_ratio",      # Fraction of corrupted received packets
]

# Link / MAC Layer (L2) — channel access and reliability metrics
JAMMING_MAC_FEATURES = [
    "packet_delivery_ratio",       # Fraction of packets successfully delivered
    "mac_retry_count",             # MAC-layer retransmission attempts per window
    "retransmission_count",        # Total retransmissions per window
    "channel_occupancy_pct",       # Channel busy fraction (%)
    "cca_failure_count",           # Clear Channel Assessment failures per window
    "backoff_count",               # MAC backoff events per window
]

# All jamming features (edge-collected)
JAMMING_FEATURES = JAMMING_PHY_FEATURES + JAMMING_MAC_FEATURES

# ═══════════════════════════════════════════════════════════════════════════════
#  DoS / DDoS DETECTION — Gateway / GCC Features
# ═══════════════════════════════════════════════════════════════════════════════

# Network Layer (L3) — IP-level traffic features
DOS_NET_FEATURES = [
    "packets_per_second",      # Total inbound packets per second
    "bytes_per_second",        # Total inbound bytes per second
    "unique_source_count",     # Number of distinct source IPs per window
    "src_ip_entropy",          # Shannon entropy of source IP distribution
    "dst_ip_entropy",          # Shannon entropy of destination IP distribution
    "icmp_packet_rate",        # ICMP packets per second (ping floods)
]

# Transport Layer (L4) — connection-level features
DOS_TRANSPORT_FEATURES = [
    "connection_attempts_per_sec",  # New connection attempts per second
    "half_open_connections",        # SYN-sent without ACK (TCP SYN flood indicator)
    "tcp_retransmission_rate",      # TCP retransmissions per second
    "udp_packet_rate",              # UDP packets per second
    "syn_packet_rate",              # SYN packets per second
    "fin_rst_rate",                 # FIN+RST packets per second
    "port_scan_score",              # Distinct destination ports targeted per source
]

# Application Layer (L7) — optional, UAV control-channel metrics
DOS_APP_FEATURES = [
    "api_request_rate",            # Requests/s to control or API endpoints
    "error_response_rate",         # Error / failed response rate (4xx, 5xx)
    "command_frequency",           # UAV command messages per second
]

# All DoS/DDoS features (gateway-collected)
DOS_FEATURES = DOS_NET_FEATURES + DOS_TRANSPORT_FEATURES + DOS_APP_FEATURES

# ═══════════════════════════════════════════════════════════════════════════════
#  COMBINED FEATURE SET (used for hybrid / cross-layer detection)
# ═══════════════════════════════════════════════════════════════════════════════
ALL_FEATURES = JAMMING_FEATURES + DOS_FEATURES

# Metadata columns (not used as ML features)
METADATA_COLS = [
    "timestamp",
    "node_id",
    "node_type",          # "uav" | "ugv" | "gateway"
    "collection_point",   # "edge" | "gcc"
    "window_id",
    "label",              # ground-truth: "normal" | "jamming" | "dos" | "ddos"
]


# ── Validity bounds: (min, max) ───────────────────────────────────────────────
# Physical layer bounds are grounded in 802.11/LTE/5.8 GHz hardware specs
# Network layer bounds reflect realistic UAV swarm traffic at max stress

FEATURE_BOUNDS = {
    # Physical Layer (L1)
    "rssi_dbm"                   : (-120.0, -10.0),   # Hardware limit: most radios -120 to -10 dBm
    "snr_db"                     : (-30.0,   60.0),   # SNR below -30 unphysical, above 60 ideal
    "sinr_db"                    : (-20.0,   50.0),   # SINR spec: -20 dB lower bound per 3GPP
    "noise_floor_dbm"            : (-130.0, -50.0),   # Thermal noise floor range
    "bit_error_rate"             : (0.0,     1.0),    # Probability [0, 1]
    "bad_packet_ratio"           : (0.0,     1.0),    # Fraction [0, 1]

    # Link / MAC Layer (L2)
    "packet_delivery_ratio"      : (0.0,     1.0),    # Fraction [0, 1]
    "mac_retry_count"            : (0.0,  1000.0),    # Per window; >1000 is physically impossible
    "retransmission_count"       : (0.0,  5000.0),    # Per window
    "channel_occupancy_pct"      : (0.0,   100.0),    # Percentage [0, 100]
    "cca_failure_count"          : (0.0,  2000.0),    # Per window
    "backoff_count"              : (0.0,  2000.0),    # Per window

    # Network Layer (L3)
    "packets_per_second"         : (0.0, 1_000_000.0),  # Realistic swarm max ~1 Mpps
    "bytes_per_second"           : (0.0, 1_250_000_000.0),  # Max 10 Gbps
    "unique_source_count"        : (0.0,    65535.0),   # IPv4 practical upper bound per /16
    "src_ip_entropy"             : (0.0,       16.0),   # Shannon entropy bits
    "dst_ip_entropy"             : (0.0,       16.0),
    "icmp_packet_rate"           : (0.0,  500_000.0),

    # Transport Layer (L4)
    "connection_attempts_per_sec": (0.0, 500_000.0),
    "half_open_connections"      : (0.0,  100_000.0),
    "tcp_retransmission_rate"    : (0.0,   50_000.0),
    "udp_packet_rate"            : (0.0,  500_000.0),
    "syn_packet_rate"            : (0.0,  500_000.0),
    "fin_rst_rate"               : (0.0,  500_000.0),
    "port_scan_score"            : (0.0,   65535.0),

    # Application Layer (L7)
    "api_request_rate"           : (0.0,   10_000.0),
    "error_response_rate"        : (0.0,       1.0),   # Fraction [0, 1]
    "command_frequency"          : (0.0,    1_000.0),
}

# ── Consistency rules: (metric_a, operator, metric_b) ─────────────────────────
# Flag rows where logical relationships between metrics are violated
CONSISTENCY_RULES = [
    # MAC retries cannot exceed total packets sent (approximated via PDR)
    # retransmission_count should not exceed mac_retry_count * some slack
    ("retransmission_count",    "<=", "mac_retry_count",      2.0,
     "RETX_EXCEEDS_RETRIES"),
    # half-open connections should not exceed total connection attempts
    ("half_open_connections",   "<=", "connection_attempts_per_sec", 1.0,
     "HALF_OPEN_EXCEEDS_ATTEMPTS"),
    # SYN rate should not exceed total connection attempts (SYN is a subset)
    ("syn_packet_rate",         "<=", "connection_attempts_per_sec", 1.5,
     "SYN_EXCEEDS_CONN_ATTEMPTS"),
]


class TrafficDataGenerator:
    """
    Synthetic data generator for UAV/UGV network traffic.
    Produces per-window feature vectors for Normal, Jamming, DoS, and DDoS labels.

    Window length: 1 second (all rates are per-second values).
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

    # ──────────────────────────────────────────────────────────────────────────
    #  NORMAL baseline — benign UAV swarm mission traffic
    # ──────────────────────────────────────────────────────────────────────────
    def _normal_row(self) -> dict:
        return {
            # Physical Layer — healthy signal in urban environment
            "rssi_dbm"                   : np.random.uniform(-75, -45),
            "snr_db"                     : np.random.uniform(15, 35),
            "sinr_db"                    : np.random.uniform(10, 30),
            "noise_floor_dbm"            : np.random.uniform(-100, -85),
            "bit_error_rate"             : np.random.uniform(0.0001, 0.002),
            "bad_packet_ratio"           : np.random.uniform(0.0, 0.02),
            # MAC Layer — low contention, reliable delivery
            "packet_delivery_ratio"      : np.random.uniform(0.92, 1.0),
            "mac_retry_count"            : np.random.randint(0, 10),
            "retransmission_count"       : np.random.randint(0, 8),
            "channel_occupancy_pct"      : np.random.uniform(5, 30),
            "cca_failure_count"          : np.random.randint(0, 5),
            "backoff_count"              : np.random.randint(0, 5),
            # Network Layer — mission telemetry traffic
            "packets_per_second"         : np.random.uniform(10, 200),
            "bytes_per_second"           : np.random.uniform(1_000, 50_000),
            "unique_source_count"        : np.random.randint(2, 15),
            "src_ip_entropy"             : np.random.uniform(1.0, 3.5),
            "dst_ip_entropy"             : np.random.uniform(0.5, 2.0),
            "icmp_packet_rate"           : np.random.uniform(0, 5),
            # Transport Layer — normal connections
            "connection_attempts_per_sec": np.random.uniform(0.5, 10),
            "half_open_connections"      : np.random.randint(0, 5),
            "tcp_retransmission_rate"    : np.random.uniform(0, 3),
            "udp_packet_rate"            : np.random.uniform(5, 100),
            "syn_packet_rate"            : np.random.uniform(0.5, 8),
            "fin_rst_rate"               : np.random.uniform(0.5, 8),
            "port_scan_score"            : np.random.randint(1, 5),
            # Application Layer — normal command traffic
            "api_request_rate"           : np.random.uniform(1, 20),
            "error_response_rate"        : np.random.uniform(0.0, 0.03),
            "command_frequency"          : np.random.uniform(2, 30),
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  JAMMING — RF interference (constant or reactive jammer)
    #  Scenario 1: Urban disaster zone, portable RF jammer
    # ──────────────────────────────────────────────────────────────────────────
    def _jamming_row(self, severity: float = 1.0) -> dict:
        """
        severity in [0.5, 1.5]: scales attack intensity.
        Physical / MAC layer metrics degrade; network layer stays near normal
        (jammer affects the radio, not the IP stack directly).
        """
        row = self._normal_row()   # start from normal baseline
        s = np.clip(severity, 0.5, 1.5)

        # Physical layer — signal degradation
        row["rssi_dbm"]          = np.random.uniform(-110, -80) * s
        row["rssi_dbm"]          = np.clip(row["rssi_dbm"], -120, -10)
        row["snr_db"]            = np.random.uniform(-10, 8) / s
        row["sinr_db"]           = np.random.uniform(-15, 5) / s
        row["noise_floor_dbm"]   = np.random.uniform(-80, -60) * s
        row["noise_floor_dbm"]   = np.clip(row["noise_floor_dbm"], -130, -50)
        row["bit_error_rate"]    = np.random.uniform(0.05, 0.40) * s
        row["bit_error_rate"]    = np.clip(row["bit_error_rate"], 0, 1)
        row["bad_packet_ratio"]  = np.random.uniform(0.10, 0.60) * s
        row["bad_packet_ratio"]  = np.clip(row["bad_packet_ratio"], 0, 1)

        # MAC layer — increased contention, retransmissions, CCA failures
        row["packet_delivery_ratio"] = np.random.uniform(0.30, 0.70) / s
        row["packet_delivery_ratio"] = np.clip(row["packet_delivery_ratio"], 0, 1)
        row["mac_retry_count"]       = int(np.random.randint(30, 150) * s)
        row["retransmission_count"]  = int(np.random.randint(25, 120) * s)
        row["channel_occupancy_pct"] = np.random.uniform(55, 95) * s
        row["channel_occupancy_pct"] = np.clip(row["channel_occupancy_pct"], 0, 100)
        row["cca_failure_count"]     = int(np.random.randint(80, 400) * s)
        row["backoff_count"]         = int(np.random.randint(80, 400) * s)

        # Network/transport layer: degraded due to physical loss, not direct attack
        row["packets_per_second"]    = np.random.uniform(5, 80)    # drops due to losses
        row["tcp_retransmission_rate"] = np.random.uniform(5, 30)  # elevated TCP retx

        return row

    # ──────────────────────────────────────────────────────────────────────────
    #  DoS — single-source flooding
    #  Scenario 2 (single attacker), Scenario 3 (hybrid)
    # ──────────────────────────────────────────────────────────────────────────
    def _dos_row(self, severity: float = 1.0) -> dict:
        """
        Single-source high-rate flooding targeting relay UAV control ports.
        Physical/MAC layer remains clean (attacker is on the IP network).
        """
        row = self._normal_row()
        s = np.clip(severity, 0.5, 1.5)

        # Network layer — traffic volume spike from one source
        row["packets_per_second"]          = np.random.uniform(5_000, 50_000) * s
        row["bytes_per_second"]            = np.random.uniform(5e6, 80e6) * s
        row["unique_source_count"]         = np.random.randint(1, 4)     # single source
        row["src_ip_entropy"]              = np.random.uniform(0.0, 0.5) # very low entropy
        row["dst_ip_entropy"]              = np.random.uniform(0.0, 0.8)
        row["icmp_packet_rate"]            = np.random.uniform(500, 5_000) * s

        # Transport layer — connection exhaustion
        row["connection_attempts_per_sec"] = np.random.uniform(1_000, 20_000) * s
        row["half_open_connections"]       = np.random.randint(500, 5_000)
        row["tcp_retransmission_rate"]     = np.random.uniform(100, 2_000) * s
        row["syn_packet_rate"]             = np.random.uniform(1_000, 20_000) * s
        row["fin_rst_rate"]                = np.random.uniform(200, 5_000) * s
        row["port_scan_score"]             = np.random.randint(1, 10)     # focused port

        # Application layer — control endpoint saturation
        row["api_request_rate"]            = np.random.uniform(500, 5_000) * s
        row["error_response_rate"]         = np.random.uniform(0.60, 1.0)
        row["command_frequency"]           = np.random.uniform(200, 1_000) * s

        return row

    # ──────────────────────────────────────────────────────────────────────────
    #  DDoS — coordinated multi-source flooding
    #  Scenario 2: multiple attacker devices
    # ──────────────────────────────────────────────────────────────────────────
    def _ddos_row(self, severity: float = 1.0) -> dict:
        """
        Multi-source coordinated flooding — high source diversity, high volume.
        Key distinguisher from DoS: high unique_source_count and src_ip_entropy.
        """
        row = self._dos_row(severity)
        s = np.clip(severity, 0.5, 1.5)

        # Override source diversity (key DDoS signature)
        row["unique_source_count"] = int(np.random.randint(50, 500) * s)
        row["src_ip_entropy"]      = np.random.uniform(5.0, 9.0)   # high entropy
        row["dst_ip_entropy"]      = np.random.uniform(0.5, 2.5)   # focused target

        # Even higher volume due to multiple sources
        row["packets_per_second"]  = np.random.uniform(20_000, 200_000) * s
        row["bytes_per_second"]    = np.random.uniform(20e6, 200e6) * s
        row["udp_packet_rate"]     = np.random.uniform(5_000, 100_000) * s

        return row

    # ──────────────────────────────────────────────────────────────────────────
    #  HYBRID — simultaneous jamming + DoS/DDoS (Scenario 3)
    # ──────────────────────────────────────────────────────────────────────────
    def _hybrid_row(self, severity: float = 1.0) -> dict:
        """
        Combines physical-layer jamming signatures with network-layer flooding.
        Most demanding scenario: cross-layer attack.
        """
        jam = self._jamming_row(severity)
        dos = self._ddos_row(severity)
        # Merge: physical/MAC from jamming row, network/transport/app from DDoS row
        row = {}
        for feat in JAMMING_FEATURES:
            row[feat] = jam[feat]
        for feat in DOS_FEATURES:
            row[feat] = dos[feat]
        return row

    # ──────────────────────────────────────────────────────────────────────────
    #  PUBLIC: generate full dataset
    # ──────────────────────────────────────────────────────────────────────────
    def generate(
        self,
        n_normal  : int = 2000,
        n_jamming : int = 800,
        n_dos     : int = 600,
        n_ddos    : int = 600,
        n_hybrid  : int = 400,
    ) -> pd.DataFrame:
        records = []
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        def make_meta(label, node_type, collection_point, idx):
            return {
                "timestamp"        : (t0 + pd.Timedelta(seconds=idx)).isoformat(),
                "node_id"          : f"{node_type}_{(idx % 10) + 1:03d}",
                "node_type"        : node_type,
                "collection_point" : collection_point,
                "window_id"        : idx,
                "label"            : label,
            }

        idx = 0

        for _ in range(n_normal):
            r = self._normal_row()
            r.update(make_meta("normal", "uav", "edge", idx)); idx += 1
            records.append(r)

        for _ in range(n_jamming):
            sev = np.random.uniform(0.7, 1.4)
            r = self._jamming_row(sev)
            r.update(make_meta("jamming", "uav", "edge", idx)); idx += 1
            records.append(r)

        for _ in range(n_dos):
            sev = np.random.uniform(0.7, 1.4)
            r = self._dos_row(sev)
            r.update(make_meta("dos", "gateway", "gcc", idx)); idx += 1
            records.append(r)

        for _ in range(n_ddos):
            sev = np.random.uniform(0.8, 1.4)
            r = self._ddos_row(sev)
            r.update(make_meta("ddos", "gateway", "gcc", idx)); idx += 1
            records.append(r)

        for _ in range(n_hybrid):
            sev = np.random.uniform(0.9, 1.5)
            r = self._hybrid_row(sev)
            r.update(make_meta("hybrid", "uav", "edge_gcc", idx)); idx += 1
            records.append(r)

        df = pd.DataFrame(records)
        df = shuffle(df, random_state=42).reset_index(drop=True)
        return df


# ── Instantiate and generate ──────────────────────────────────────────────────


class EdgeCollector:
    """
    Edge Agent — Deployed on UAV/UGV nodes.
    Collects Physical (L1) and MAC (L2) metrics for jamming detection.
    Operates in real-time with a 1-second aggregation window.
    """

    def __init__(self, node_id: str, node_type: str = "uav"):
        self.node_id   = node_id
        self.node_type = node_type
        self._window_buffer: list[dict] = []
        self._window_id = 0

    def ingest(self, raw_sample: dict) -> None:
        """Buffer a raw sample (sub-second granularity)."""
        self._window_buffer.append(raw_sample)

    def flush_window(self) -> dict | None:
        """
        Aggregate buffered samples into a 1-second feature window.
        Returns a record dict ready for the cleaning pipeline.
        """
        if not self._window_buffer:
            return None

        buf = pd.DataFrame(self._window_buffer)
        record = {
            "timestamp"        : datetime.now(timezone.utc).isoformat(),
            "node_id"          : self.node_id,
            "node_type"        : self.node_type,
            "collection_point" : "edge",
            "window_id"        : self._window_id,
        }

        # Aggregate each jamming feature — mean over the window
        for feat in JAMMING_FEATURES:
            if feat in buf.columns:
                record[feat] = float(buf[feat].mean())
            else:
                record[feat] = np.nan   # missing → flagged by cleaner

        # DoS features are not collected at edge; fill with NaN
        for feat in DOS_FEATURES:
            record[feat] = np.nan

        self._window_buffer.clear()
        self._window_id += 1
        return record

    def collect_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method: collect all rows from a pre-built DataFrame
        (used in simulation / offline processing).
        """
        records = []
        for _, row in df.iterrows():
            rec = row.to_dict()
            rec["collection_point"] = "edge"
            rec["node_id"]          = self.node_id
            rec["node_type"]        = self.node_type
            records.append(rec)
        return pd.DataFrame(records)


class GCCCollector:
    """
    Gateway / GCC Agent — Deployed at the Ground Control Center.
    Collects Network (L3), Transport (L4), and Application (L7) metrics
    for DoS/DDoS detection.
    Performs packet-level aggregation into per-second feature windows.
    """

    def __init__(self, gateway_id: str = "gcc_001"):
        self.gateway_id = gateway_id
        self._window_id = 0
        self._flow_table: dict[str, dict] = {}    # src_ip → stats
        self._half_open : set[str]        = set() # unacked SYN keys

    def _compute_entropy(self, counts: list[int]) -> float:
        """Shannon entropy (bits) of a frequency distribution."""
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        return float(-sum(p * math.log2(p) for p in probs))

    def aggregate_window(self, packet_records: list[dict]) -> dict | None:
        """
        Aggregate a list of packet-level dicts captured in a 1-second window
        into a feature vector.

        Expected packet fields: src_ip, dst_ip, src_port, dst_port,
                                 protocol, length, tcp_flags
        """
        if not packet_records:
            return None

        pps   = len(packet_records)
        bps   = sum(p.get("length", 0) for p in packet_records)

        src_counts = defaultdict(int)
        dst_counts = defaultdict(int)
        syn_count = fin_rst_count = icmp_count = udp_count = 0
        conn_attempts = 0

        for pkt in packet_records:
            src_counts[pkt.get("src_ip", "")] += 1
            dst_counts[pkt.get("dst_ip", "")] += 1
            proto = pkt.get("protocol", "").upper()
            flags = pkt.get("tcp_flags", "").upper()
            if proto == "ICMP":
                icmp_count += 1
            if proto == "UDP":
                udp_count += 1
            if "SYN" in flags and "ACK" not in flags:
                syn_count += 1
                conn_attempts += 1
            if "FIN" in flags or "RST" in flags:
                fin_rst_count += 1

        record = {
            "timestamp"                  : datetime.now(timezone.utc).isoformat(),
            "node_id"                    : self.gateway_id,
            "node_type"                  : "gateway",
            "collection_point"           : "gcc",
            "window_id"                  : self._window_id,
            # L3
            "packets_per_second"         : pps,
            "bytes_per_second"           : bps,
            "unique_source_count"        : len(src_counts),
            "src_ip_entropy"             : self._compute_entropy(list(src_counts.values())),
            "dst_ip_entropy"             : self._compute_entropy(list(dst_counts.values())),
            "icmp_packet_rate"           : icmp_count,
            # L4
            "connection_attempts_per_sec": conn_attempts,
            "half_open_connections"      : len(self._half_open),
            "tcp_retransmission_rate"    : 0,   # needs full TCP state; 0 as placeholder
            "udp_packet_rate"            : udp_count,
            "syn_packet_rate"            : syn_count,
            "fin_rst_rate"               : fin_rst_count,
            "port_scan_score"            : 0,   # computed separately
            # L7 — filled with NaN if unavailable
            "api_request_rate"           : np.nan,
            "error_response_rate"        : np.nan,
            "command_frequency"          : np.nan,
        }

        # Jamming features not collected at GCC — NaN
        for feat in JAMMING_FEATURES:
            record[feat] = np.nan

        self._window_id += 1
        return record

    def collect_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience offline collection from pre-built DataFrame.
        """
        records = []
        for _, row in df.iterrows():
            rec = row.to_dict()
            rec["collection_point"] = "gcc"
            rec["node_id"]          = self.gateway_id
            rec["node_type"]        = "gateway"
            records.append(rec)
        return pd.DataFrame(records)


# ── Simulate collection: split raw_df by collection point ─────────────────────

edge_collector = EdgeCollector(node_id="uav_001", node_type="uav")
gcc_collector  = GCCCollector(gateway_id="gcc_001")


class DataQualityChecker:
    """
    Stateless data quality checker.
    Annotates each row with quality flags and a composite quality score.
    Does NOT drop or modify values — remediation is handled by DataCleaner.
    """

    def __init__(
        self,
        bounds          : dict = FEATURE_BOUNDS,
        consistency_rules: list = CONSISTENCY_RULES,
        spike_z_thresh  : float = 4.0,
        min_required_phy: list  = None,
        min_required_net: list  = None,
    ):
        self.bounds            = bounds
        self.rules             = consistency_rules
        self.spike_z           = spike_z_thresh
        # Essential fields that must be non-null per collection point
        self.min_required_phy  = min_required_phy or [
            "rssi_dbm", "snr_db", "packet_delivery_ratio", "mac_retry_count"
        ]
        self.min_required_net  = min_required_net or [
            "packets_per_second", "bytes_per_second",
            "unique_source_count", "connection_attempts_per_sec"
        ]

    # ── 1. Missing fields ──────────────────────────────────────────────────────
    def _check_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag rows missing essential fields for their collection point.
        Adds: missing_essential_flag, missing_field_list
        """
        flags = []
        for _, row in df.iterrows():
            cp = row.get("collection_point", "")
            required = (
                self.min_required_phy if cp == "edge"
                else self.min_required_net if cp == "gcc"
                else self.min_required_phy + self.min_required_net
            )
            missing = [f for f in required if f not in row.index or pd.isna(row[f])]
            flags.append({"missing_essential_flag": int(len(missing) > 0),
                          "missing_field_list"     : json.dumps(missing)})

        return df.assign(**pd.DataFrame(flags, index=df.index))

    # ── 2. Out-of-range values ─────────────────────────────────────────────────
    def _check_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag rows containing any value outside defined physical/logical bounds.
        Adds: out_of_range_flag, out_of_range_fields
        """
        flags = []
        for _, row in df.iterrows():
            bad = []
            for feat, (lo, hi) in self.bounds.items():
                if feat in row.index and pd.notna(row[feat]):
                    val = float(row[feat])
                    if not (lo <= val <= hi):
                        bad.append(f"{feat}={val:.4g}")
            flags.append({"out_of_range_flag"  : int(len(bad) > 0),
                          "out_of_range_fields" : json.dumps(bad)})

        return df.assign(**pd.DataFrame(flags, index=df.index))

    # ── 3. Consistency checks ──────────────────────────────────────────────────
    def _check_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag rows violating cross-metric logical relationships.
        Adds: consistency_flag, consistency_violations
        """
        flags = []
        for _, row in df.iterrows():
            viols = []
            for (a, op, b, slack, tag) in self.rules:
                if a in row.index and b in row.index:
                    va = row.get(a); vb = row.get(b)
                    if pd.notna(va) and pd.notna(vb):
                        if op == "<=":
                            if float(va) > float(vb) * slack:
                                viols.append(tag)
                        elif op == ">=":
                            if float(va) < float(vb) * slack:
                                viols.append(tag)
            flags.append({"consistency_flag"       : int(len(viols) > 0),
                          "consistency_violations"  : json.dumps(viols)})

        return df.assign(**pd.DataFrame(flags, index=df.index))

    # ── 4. Duplicate timestamps ────────────────────────────────────────────────
    def _check_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag rows with a duplicate (timestamp, node_id) key.
        Adds: duplicate_flag
        """
        dup_mask = df.duplicated(subset=["timestamp", "node_id"], keep="first")
        return df.assign(duplicate_flag=dup_mask.astype(int))

    # ── 5. Statistical spike / drop detection ─────────────────────────────────
    def _check_spikes(self, df: pd.DataFrame, numeric_feats: list) -> pd.DataFrame:
        """
        Flag rows where a feature value is more than spike_z standard deviations
        from the rolling column mean. Uses global stats (suitable for batch).
        Adds: spike_flag, spike_fields
        """
        numeric_df = df[numeric_feats].apply(pd.to_numeric, errors="coerce")
        col_means  = numeric_df.mean()
        col_stds   = numeric_df.std().replace(0, np.nan)
        z_scores   = (numeric_df - col_means).abs() / col_stds

        flags = []
        for i in range(len(df)):
            spike_cols = list(
                z_scores.columns[z_scores.iloc[i].fillna(0) > self.spike_z]
            )
            flags.append({"spike_flag"  : int(len(spike_cols) > 0),
                          "spike_fields": json.dumps(spike_cols)})

        return df.assign(**pd.DataFrame(flags, index=df.index))

    # ── Composite quality score ────────────────────────────────────────────────
    def _quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Single composite quality score [0.0 = clean, 1.0 = many issues].
        Weighted sum of individual flags.
        """
        weights = {
            "missing_essential_flag" : 0.40,
            "out_of_range_flag"      : 0.25,
            "consistency_flag"       : 0.15,
            "duplicate_flag"         : 0.10,
            "spike_flag"             : 0.10,
        }
        score = pd.Series(0.0, index=df.index)
        for col, w in weights.items():
            if col in df.columns:
                score += df[col].fillna(0).astype(float) * w
        return df.assign(quality_issue_score=score.clip(0.0, 1.0))

    # ── Public API ─────────────────────────────────────────────────────────────
    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all checks and return annotated DataFrame."""
        numeric_feats = [c for c in ALL_FEATURES if c in df.columns]
        df = self._check_missing(df)
        df = self._check_bounds(df)
        df = self._check_consistency(df)
        df = self._check_duplicates(df)
        df = self._check_spikes(df, numeric_feats)
        df = self._quality_score(df)
        return df


# ── Run quality checks on both collections ────────────────────────────────────


quality_cols = [
    "missing_essential_flag", "out_of_range_flag",
    "consistency_flag", "duplicate_flag",
    "spike_flag", "quality_issue_score"
]


class DataCleaner:
    """
    Cleans and normalises a quality-checked DataFrame.

    Remediation strategy (in order):
      1. Drop confirmed duplicates
      2. Clip out-of-range values to their valid bounds
      3. Impute remaining NaNs (median for numeric, 0 for count features)
      4. Add derived ratio features
      5. Fit + apply MinMaxScaler per feature group
    """

    def __init__(
        self,
        bounds     : dict = FEATURE_BOUNDS,
        drop_thresh: float = 0.5,     # drop rows with quality_issue_score >= thresh
        scale      : bool = True,
    ):
        self.bounds      = bounds
        self.drop_thresh = drop_thresh
        self.scale       = scale
        self._scaler_jam : MinMaxScaler | None = None
        self._scaler_dos : MinMaxScaler | None = None

    # ── Step 1: drop high-severity quality issues ──────────────────────────────
    def _drop_poor_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        if "quality_issue_score" in df.columns:
            n_before = len(df)
            df = df[df["quality_issue_score"] < self.drop_thresh].copy()
            dropped = n_before - len(df)
            if dropped > 0:
                print(f"  [Cleaner] Dropped {dropped} rows (quality_issue_score ≥ {self.drop_thresh})")
        # Always remove exact duplicates
        df = df.drop_duplicates(subset=["timestamp", "node_id"], keep="first")
        return df

    # ── Step 2: clip out-of-range values ──────────────────────────────────────
    def _clip_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        for feat, (lo, hi) in self.bounds.items():
            if feat in df.columns:
                df[feat] = pd.to_numeric(df[feat], errors="coerce").clip(lower=lo, upper=hi)
        return df

    # ── Step 3: impute missing values ─────────────────────────────────────────
    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        count_feats = [
            "mac_retry_count", "retransmission_count", "cca_failure_count",
            "backoff_count", "half_open_connections", "port_scan_score"
        ]
        for feat in ALL_FEATURES:
            if feat not in df.columns:
                df[feat] = 0.0
                continue
            if df[feat].isna().any():
                if feat in count_feats:
                    df[feat] = df[feat].fillna(0)
                else:
                    median_val = df[feat].median()
                    df[feat]   = df[feat].fillna(median_val if pd.notna(median_val) else 0.0)
        return df

    # ── Step 4: derived ratio features ────────────────────────────────────────
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add normalised ratio features to improve ML separability.
        All ratios are computed safely (denominator guarded against zero).
        """
        eps = 1e-9

        # Retransmission ratio (MAC retries per packet)
        df["retx_per_retry"] = (
            df["retransmission_count"] / (df["mac_retry_count"] + eps)
        ).clip(0, 1)

        # Channel busy fraction (alias for normalisation clarity)
        df["channel_busy_frac"] = df["channel_occupancy_pct"] / 100.0

        # SYN flood ratio: SYN packets vs total connection attempts
        df["syn_flood_ratio"] = (
            df["syn_packet_rate"] / (df["connection_attempts_per_sec"] + eps)
        ).clip(0, 5)

        # Half-open ratio: stale half-open vs total attempts
        df["half_open_ratio"] = (
            df["half_open_connections"] / (df["connection_attempts_per_sec"] + eps)
        ).clip(0, 100)

        # Source concentration: 1 / src_ip_entropy (high = concentrated = DoS)
        df["src_concentration"] = 1.0 / (df["src_ip_entropy"] + eps)
        df["src_concentration"] = df["src_concentration"].clip(0, 10)

        # Traffic load per source
        df["pps_per_source"] = (
            df["packets_per_second"] / (df["unique_source_count"] + eps)
        ).clip(0, 1_000_000)

        # SNR degradation indicator (negative SINR is strong jamming signal)
        df["sinr_negative_flag"] = (df["sinr_db"] < 0).astype(float)

        # BER severity tier
        df["ber_severe"] = (df["bit_error_rate"] > 0.1).astype(float)

        print("  [Cleaner] Derived features added:",
              ["retx_per_retry", "channel_busy_frac", "syn_flood_ratio",
               "half_open_ratio", "src_concentration", "pps_per_source",
               "sinr_negative_flag", "ber_severe"])
        return df

    # ── Step 5: scale per feature group ───────────────────────────────────────
    def _scale(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        MinMax scale jamming and DoS features independently.
        Avoids cross-group scale contamination.
        """
        derived = [
            "retx_per_retry", "channel_busy_frac", "syn_flood_ratio",
            "half_open_ratio", "src_concentration", "pps_per_source",
            "sinr_negative_flag", "ber_severe"
        ]
        jam_cols = [c for c in JAMMING_FEATURES + derived[:4] if c in df.columns]
        dos_cols = [c for c in DOS_FEATURES    + derived[4:]  if c in df.columns]

        df_scaled = df.copy()

        if fit:
            self._scaler_jam = MinMaxScaler(feature_range=(0, 1))
            self._scaler_dos = MinMaxScaler(feature_range=(0, 1))
            df_scaled[jam_cols] = self._scaler_jam.fit_transform(
                df[jam_cols].astype(float)
            )
            df_scaled[dos_cols] = self._scaler_dos.fit_transform(
                df[dos_cols].astype(float)
            )
        else:
            if self._scaler_jam:
                df_scaled[jam_cols] = self._scaler_jam.transform(
                    df[jam_cols].astype(float)
                )
            if self._scaler_dos:
                df_scaled[dos_cols] = self._scaler_dos.transform(
                    df[dos_cols].astype(float)
                )

        print(f"  [Cleaner] Scaled {len(jam_cols)} jamming features,"
              f" {len(dos_cols)} DoS/DDoS features")
        return df_scaled

    # ── Public API ─────────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: clean, derive, scale (fits scalers)."""
        print(f"[DataCleaner] Input: {len(df)} rows")
        df = self._drop_poor_quality(df)
        df = self._clip_bounds(df)
        df = self._impute(df)
        df = self._add_derived_features(df)
        if self.scale:
            df = self._scale(df, fit=True)
        df["cleaned"] = 1
        print(f"[DataCleaner] Output: {len(df)} rows")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-fitted pipeline to new data (inference-time)."""
        df = self._clip_bounds(df)
        df = self._impute(df)
        df = self._add_derived_features(df)
        if self.scale:
            df = self._scale(df, fit=False)
        df["cleaned"] = 1
        return df


# ── Merge edge + gcc checked data, then clean ─────────────────────────────────


class FeatureMatrixBuilder:
    """
    Constructs separate feature matrices for edge (jamming) and GCC (DoS/DDoS)
    detection pipelines, and builds fixed-length sliding-window sequences
    for sequential models.
    """

    DERIVED_JAM = [
        "retx_per_retry", "channel_busy_frac",
        "sinr_negative_flag", "ber_severe"
    ]
    DERIVED_DOS = [
        "syn_flood_ratio", "half_open_ratio",
        "src_concentration", "pps_per_source"
    ]

    def __init__(self, seq_len: int = 10):
        self.seq_len = seq_len

    # ── Feature column selection ───────────────────────────────────────────────
    def jamming_feature_cols(self, df: pd.DataFrame) -> list[str]:
        candidates = JAMMING_FEATURES + self.DERIVED_JAM
        return [c for c in candidates if c in df.columns]

    def dos_feature_cols(self, df: pd.DataFrame) -> list[str]:
        candidates = DOS_FEATURES + self.DERIVED_DOS
        return [c for c in candidates if c in df.columns]

    # ── Split by collection point ──────────────────────────────────────────────
    def split_by_collection(self, df: pd.DataFrame):
        """
        Returns (edge_df, gcc_df).
        Hybrid rows are included in both.
        """
        edge_df = df[df["collection_point"].isin(["edge", "edge_gcc"])].copy()
        gcc_df  = df[df["collection_point"].isin(["gcc",  "edge_gcc"])].copy()
        return edge_df, gcc_df

    # ── Build 2-D feature matrix ───────────────────────────────────────────────
    def build_matrix(self, df: pd.DataFrame, feature_cols: list[str]):
        """
        Returns (X, y, feature_cols) as numpy arrays.
        X shape: (n_samples, n_features)
        y shape: (n_samples,) — integer encoded labels
        """
        label_map = {"normal": 0, "jamming": 1, "dos": 2, "ddos": 3, "hybrid": 4}
        X = df[feature_cols].astype(float).values
        y = df["label"].map(label_map).fillna(-1).astype(int).values
        return X, y, feature_cols

    # ── Build 3-D sliding window sequences ────────────────────────────────────
    def build_sequences(self, df: pd.DataFrame, feature_cols: list[str]):
        """
        Creates overlapping sequences of length seq_len for temporal models.
        Groups by node_id and slides a window over chronological records.

        Returns:
            X_seq : (n_windows, seq_len, n_features)
            y_seq : (n_windows,)  label of the LAST step in the window
        """
        df_sorted = df.sort_values(["node_id", "timestamp"]).copy()
        X_list, y_list = [], []
        label_map = {"normal": 0, "jamming": 1, "dos": 2, "ddos": 3, "hybrid": 4}

        for node_id, group in df_sorted.groupby("node_id"):
            vals   = group[feature_cols].astype(float).values
            labels = group["label"].map(label_map).fillna(-1).astype(int).values
            n = len(vals)
            if n < self.seq_len:
                continue
            for i in range(n - self.seq_len + 1):
                X_list.append(vals[i : i + self.seq_len])
                y_list.append(labels[i + self.seq_len - 1])

        if not X_list:
            return np.empty((0, self.seq_len, len(feature_cols))), np.empty(0, dtype=int)

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ── Build matrices ─────────────────────────────────────────────────────────────


# ── Convenience aliases (used by Feature Engineering agent) ───────────────────
JAM_FEAT_COLS = JAMMING_FEATURES
DOS_FEAT_COLS = DOS_FEATURES
ALL_FEAT_COLS = ALL_FEATURES
