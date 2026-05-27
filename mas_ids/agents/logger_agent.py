"""
mas_ids.agents.logger_agent
===========================
Agent 6 — Logger.

SHA-256 integrity hashing per event, Merkle-tree batch indexing, Bloom filter
for DDoS source tracking, traffic sketches (PPS/entropy/SYN), physical-layer
sketches (SNR/BER/channel events), and tamper detection.

Public API:
  BloomFilter, TrafficSketchAggregator, PhysicalLayerSketch, LoggerAgent,
  verify_integrity, run_logger_agent, analyse_logs
"""
import os
import json
import math
import hashlib
import uuid
import struct
from copy import deepcopy
from datetime import datetime, timezone
from collections import defaultdict, deque

import numpy as np
import pandas as pd


class BloomFilter:
    """
    Space-efficient probabilistic membership data structure.
    Used for DDoS source IP tracking per the requirements spec:
    "efficient source membership tracking, approximate DDoS source
    reconstruction, low-memory footprint logging."

    Parameters
    ----------
    capacity    : expected number of elements
    error_rate  : desired false-positive probability
    """

    def __init__(self, capacity: int = 10_000, error_rate: float = 0.01):
        self.capacity   = capacity
        self.error_rate = error_rate
        # Optimal bit-array size and number of hash functions
        self.m = self._optimal_m(capacity, error_rate)  # bits
        self.k = self._optimal_k(capacity, self.m)      # hash functions
        self._bits  = bytearray(math.ceil(self.m / 8))
        self._count = 0

    @staticmethod
    def _optimal_m(n: int, p: float) -> int:
        return max(1, int(-n * math.log(p) / (math.log(2) ** 2)))

    @staticmethod
    def _optimal_k(n: int, m: int) -> int:
        return max(1, round((m / n) * math.log(2)))

    def _hashes(self, item: str) -> list[int]:
        """Generate k independent hash positions using double-hashing."""
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)  % self.m
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16) % self.m
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item: str) -> None:
        """Insert item into the filter."""
        for pos in self._hashes(str(item)):
            byte_idx, bit_idx = divmod(pos, 8)
            self._bits[byte_idx] |= (1 << bit_idx)
        self._count += 1

    def __contains__(self, item: str) -> bool:
        """Return True if item is possibly in the filter (may false-positive)."""
        for pos in self._hashes(str(item)):
            byte_idx, bit_idx = divmod(pos, 8)
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def estimated_fpr(self) -> float:
        """Current estimated false-positive rate given items inserted."""
        if self._count == 0:
            return 0.0
        return (1 - math.exp(-self.k * self._count / self.m)) ** self.k

    def to_summary(self) -> dict:
        return {
            "capacity"     : self.capacity,
            "error_rate"   : self.error_rate,
            "m_bits"       : self.m,
            "k_hashes"     : self.k,
            "items_added"  : self._count,
            "estimated_fpr": round(self.estimated_fpr(), 6),
            "memory_bytes" : len(self._bits),
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

class TrafficSketchAggregator:
    """
    Aggregated traffic sketch for DoS/DDoS log compression.
    Per the spec: packet rate summaries, flow-level statistics,
    hashed source identifiers, queue depth statistics.
    One sketch covers a fixed time window (default 10 seconds).
    """

    def __init__(self, window_seconds: int = 10):
        self.window_seconds  = window_seconds
        self._source_bloom   = BloomFilter(capacity=5000, error_rate=0.01)
        self._hashed_sources : set = set()
        self._pps_samples    : list = []
        self._bps_samples    : list = []
        self._syn_samples    : list = []
        self._entropy_samples: list = []
        self._window_count   : int  = 0
        self._attack_windows : int  = 0

    def ingest_row(self, row: pd.Series) -> None:
        """Ingest one detection window row into the sketch."""
        label = str(row.get("final_label", "NORMAL"))

        # Packet rate summaries
        pps = float(row.get("packets_per_second", 0) or 0)
        self._pps_samples.append(pps)
        self._bps_samples.append(float(row.get("bytes_per_second", 0) or 0))
        self._syn_samples.append(float(row.get("syn_packet_rate",  0) or 0))
        self._entropy_samples.append(float(row.get("src_ip_entropy", 0) or 0))

        # Hashed source identifier (privacy-preserving)
        node_id = str(row.get("node_id", ""))
        src_hash = hashlib.sha256(node_id.encode()).hexdigest()[:16]
        self._hashed_sources.add(src_hash)
        self._source_bloom.add(node_id)

        self._window_count += 1
        if label not in {"NORMAL", "SUSPICIOUS"}:
            self._attack_windows += 1

    def finalise(self) -> dict:
        """Return the compressed sketch summary for logging."""
        def safe_stats(samples):
            if not samples: return {"mean":0.0,"max":0.0,"p95":0.0}
            a = np.array(samples, dtype=float)
            return {
                "mean": round(float(a.mean()), 2),
                "max" : round(float(a.max()),  2),
                "p95" : round(float(np.percentile(a, 95)), 2),
            }

        return {
            "window_seconds"      : self.window_seconds,
            "total_windows"       : self._window_count,
            "attack_windows"      : self._attack_windows,
            "attack_fraction"     : round(
                self._attack_windows / max(self._window_count, 1), 4),
            "pps_stats"           : safe_stats(self._pps_samples),
            "bps_stats"           : safe_stats(self._bps_samples),
            "syn_rate_stats"      : safe_stats(self._syn_samples),
            "src_entropy_stats"   : safe_stats(self._entropy_samples),
            "unique_sources_hashed": len(self._hashed_sources),
            "bloom_summary"       : self._source_bloom.to_summary(),
        }


class PhysicalLayerSketch:
    """
    Physical layer metrics sketch for jamming log compression.
    Per the spec: SNR averages, noise floor deviations,
    packet error rate (PER), channel switching events.
    """

    def __init__(self):
        self._snr_samples        : list = []
        self._noise_samples      : list = []
        self._ber_samples        : list = []
        self._pdr_samples        : list = []
        self._ch_occ_samples     : list = []
        self._channel_switches   : int  = 0
        self._lte_activations    : int  = 0
        self._jam_windows        : int  = 0
        self._total_windows      : int  = 0

    def ingest_row(self, det_row: pd.Series, resp_row: pd.Series | None = None) -> None:
        self._snr_samples.append(  float(det_row.get("snr_db",                0) or 0))
        self._noise_samples.append(float(det_row.get("noise_floor_dbm",      -90) or -90))
        self._ber_samples.append(  float(det_row.get("bit_error_rate",         0) or 0))
        self._pdr_samples.append(  float(det_row.get("packet_delivery_ratio", 1) or 1))
        self._ch_occ_samples.append(float(det_row.get("channel_occupancy_pct",20) or 20))
        label = str(det_row.get("final_label","NORMAL"))
        if label in {"JAMMING","HYBRID_ATTACK"}:
            self._jam_windows += 1
        self._total_windows += 1
        if resp_row is not None:
            rf_act = str(resp_row.get("rf_action_taken",""))
            if "CHANNEL" in rf_act or "HOP" in rf_act or "SWITCH" in rf_act:
                self._channel_switches += 1
            if resp_row.get("lte_fallback_active", False):
                self._lte_activations  += 1

    def finalise(self) -> dict:
        def safe_stats(samples, default=0.0):
            if not samples: return {"mean":default,"std":0.0,"min":default,"max":default}
            a = np.array(samples, dtype=float)
            return {
                "mean": round(float(a.mean()), 4),
                "std" : round(float(a.std()),  4),
                "min" : round(float(a.min()),  4),
                "max" : round(float(a.max()),  4),
            }
        return {
            "total_windows"      : self._total_windows,
            "jamming_windows"    : self._jam_windows,
            "snr_db"             : safe_stats(self._snr_samples, 15.0),
            "noise_floor_dbm"    : safe_stats(self._noise_samples, -90.0),
            "bit_error_rate"     : safe_stats(self._ber_samples),
            "packet_delivery_ratio": safe_stats(self._pdr_samples, 1.0),
            "channel_occupancy_pct": safe_stats(self._ch_occ_samples, 20.0),
            "channel_switch_events": self._channel_switches,
            "lte_fallback_events": self._lte_activations,
        }


class LoggerAgent:
    """
    Secure, tamper-resistant event logger for the MAS-IDS pipeline.
    Mirrors LoggerAgent from the original (Cell 17) exactly.
    Extended with log_gcc_event() for the Management Agent.

    Integrity mechanism (per spec):
      - SHA-256 hash of each log entry
      - Merkle tree indexing for efficient verification
      - Distributed validation across swarm (via batch_id)
      - Tamper detection via verify_integrity()
    """

    def __init__(self, batch_size: int = 10, agent_version: str = "v1.1"):
        self.logs                : list = []
        self.pending_hashes      : list = []
        self.pending_log_indices : list = []
        self.merkle_batches      : list = []
        self.batch_size          = int(batch_size)
        self.agent_version       = agent_version

    # ── Static helpers (mirrors original exactly) ──────────────────────────────
    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_list(v):
        if v is None: return []
        return v if isinstance(v, list) else [v]

    @staticmethod
    def _safe_dict(v):
        if v is None: return {}
        return v if isinstance(v, dict) else {"value": v}

    @staticmethod
    def _safe_bool(v, d: bool = False) -> bool:
        return d if v is None else bool(v)

    @staticmethod
    def _safe_float(v, d: float = 0.0) -> float:
        try: return float(d) if v is None or v == "" else float(v)
        except: return float(d)

    @staticmethod
    def _safe_str(v, d: str = "") -> str:
        return d if v is None else str(v)

    # ── SHA-256 hash of one record (mirrors original exactly) ──────────────────
    def compute_hash(self, record: dict) -> str:
        return hashlib.sha256(
            json.dumps(record, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    # ── Merkle tree (mirrors original exactly) ─────────────────────────────────
    def build_merkle_root(self, hashes: list) -> str | None:
        if not hashes: return None
        current = hashes[:]
        while len(current) > 1:
            nxt = []
            for i in range(0, len(current), 2):
                left  = current[i]
                right = current[i + 1] if i + 1 < len(current) else current[i]
                nxt.append(hashlib.sha256((left + right).encode()).hexdigest())
            current = nxt
        return current[0]

    # ── Batch flush (mirrors original exactly) ─────────────────────────────────
    def flush_batch_if_needed(self, force: bool = False) -> None:
        if not force and len(self.pending_hashes) < self.batch_size: return
        if not self.pending_hashes: return
        batch_id  = len(self.merkle_batches) + 1
        batch_logs = [self.logs[i] for i in self.pending_log_indices]
        self.merkle_batches.append({
            "batch_id"             : batch_id,
            "batch_timestamp"      : self.utc_now_iso(),
            "batch_size"           : len(self.pending_hashes),
            "merkle_root"          : self.build_merkle_root(self.pending_hashes),
            "record_hashes"        : self.pending_hashes.copy(),
            "first_event_timestamp": batch_logs[0].get("log_timestamp") if batch_logs else None,
            "last_event_timestamp" : batch_logs[-1].get("log_timestamp") if batch_logs else None,
            "source_types"         : sorted(set(
                l.get("source_type","unknown") for l in batch_logs
            )),
        })
        for i in self.pending_log_indices:
            self.logs[i]["batch_id"] = batch_id
        self.pending_hashes      = []
        self.pending_log_indices = []

    # ── Core log_event (mirrors original exactly) ──────────────────────────────
    def log_event(self, record: dict) -> dict:
        record = deepcopy(record)
        record["event_id"]      = self._safe_str(record.get("event_id", str(uuid.uuid4())))
        record["correlation_id"]= self._safe_str(record.get("correlation_id", record["event_id"]))
        record["log_timestamp"] = self.utc_now_iso()
        record["agent_version"] = self.agent_version
        record["log_hash"]      = self.compute_hash(record)
        record["batch_id"]      = None
        self.logs.append(record)
        log_index = len(self.logs) - 1
        self.pending_hashes.append(record["log_hash"])
        self.pending_log_indices.append(log_index)
        self.flush_batch_if_needed(force=False)
        return record

    # ── log_uav_ugv_event (adapted for Jamming/DoS column names) ──────────────
    def log_uav_ugv_event(
        self,
        node_id: str,
        node_type: str,
        final_label: str,
        severity: str,
        # Jamming/DoS risk scores (replace gps/routing_anomaly_score from original)
        jam_risk_score: float          = 0.0,
        dos_risk_score: float          = 0.0,
        hybrid_risk_score: float       = 0.0,
        fusion_confidence: float       = 0.0,
        detection_reason_codes: list   = None,
        # Response actions
        rf_action_taken: str           = "",
        net_action_taken: str          = "",
        safety_override: bool          = False,
        reactive_triggered: bool       = False,
        response_status: str           = "EXECUTED",
        response_success: bool         = True,
        # Sensor / network summaries
        sensor_summary: dict           = None,
        routing_summary: dict          = None,
        status_summary: dict           = None,
        # Trust
        trust_score_before: float      = None,
        trust_score_after: float       = None,
        # Notifications
        target_nodes: list             = None,
        notified_neighbors: list       = None,
        gcc_notified: bool             = False,
        alert_scope: str               = "local",
        # Correlation
        correlation_id: str            = None,
        event_id: str                  = None,
        # Audit
        feedback_label: str            = "",
        review_notes: str              = "",
    ) -> dict:
        """
        Log one UAV/UGV detection + response event.
        Mirrors log_uav_ugv_event() from the original, adapted for
        Jamming/DoS columns (jam_risk_score, dos_risk_score, rf_action_taken
        replace gps_anomaly_score, routing_anomaly_score, local_response_action).
        """
        if event_id      is None: event_id      = str(uuid.uuid4())
        if correlation_id is None: correlation_id = event_id
        return self.log_event({
            "event_id"             : event_id,
            "correlation_id"       : correlation_id,
            "source_type"          : "uav_ugv",
            "node_id"              : node_id,
            "node_type"            : node_type,
            "final_label"          : final_label,
            "severity"             : severity,
            "jam_risk_score"       : jam_risk_score,
            "dos_risk_score"       : dos_risk_score,
            "hybrid_risk_score"    : hybrid_risk_score,
            "fusion_confidence"    : fusion_confidence,
            "detection_reason_codes": self._safe_list(detection_reason_codes),
            "rf_action_taken"      : rf_action_taken,
            "net_action_taken"     : net_action_taken,
            "safety_override"      : safety_override,
            "reactive_triggered"   : reactive_triggered,
            "response_status"      : response_status,
            "response_success"     : response_success,
            "sensor_summary"       : self._safe_dict(sensor_summary),
            "routing_summary"      : self._safe_dict(routing_summary),
            "status_summary"       : self._safe_dict(status_summary),
            "trust_score_before"   : trust_score_before,
            "trust_score_after"    : trust_score_after,
            "target_nodes"         : self._safe_list(target_nodes),
            "notified_neighbors"   : self._safe_list(notified_neighbors),
            "gcc_notified"         : gcc_notified,
            "alert_scope"          : alert_scope,
            "feedback_label"       : feedback_label,
            "review_notes"         : review_notes,
        })

    # ── log_edge_event (mirrors original exactly) ──────────────────────────────
    def log_edge_event(
        self,
        edge_id: str,
        final_label: str,
        severity: str,
        correlated_attack_confirmation: bool,
        trust_score_before: float,
        trust_score_after: float,
        coordinated_swarm_action: str,
        quarantine_decision: bool,
        rerouting_decision: bool,
        target_nodes: list,
        response_status: str,
        response_success: bool,
        correlation_id: str,
        event_id: str                  = None,
        status_summary: dict           = None,
        alert_scope: str               = "edge",
        notified_neighbors: list       = None,
        gcc_notified: bool             = False,
        feedback_label: str            = "",
        review_notes: str              = "",
    ) -> dict:
        """Log one edge/coordination event. Mirrors original exactly."""
        if event_id is None: event_id = str(uuid.uuid4())
        return self.log_event({
            "event_id"                      : event_id,
            "correlation_id"                : correlation_id,
            "source_type"                   : "edge",
            "edge_id"                       : edge_id,
            "node_type"                     : "EDGE",
            "final_label"                   : final_label,
            "severity"                      : severity,
            "correlated_attack_confirmation": correlated_attack_confirmation,
            "trust_score_before"            : trust_score_before,
            "trust_score_after"             : trust_score_after,
            "coordinated_swarm_action"      : coordinated_swarm_action,
            "quarantine_decision"           : quarantine_decision,
            "rerouting_decision"            : rerouting_decision,
            "target_nodes"                  : self._safe_list(target_nodes),
            "response_status"               : response_status,
            "response_success"              : response_success,
            "status_summary"                : self._safe_dict(status_summary),
            "alert_scope"                   : alert_scope,
            "notified_neighbors"            : self._safe_list(notified_neighbors),
            "gcc_notified"                  : gcc_notified,
            "feedback_label"                : feedback_label,
            "review_notes"                  : review_notes,
        })

    # ── log_gcc_event (NEW — Management Agent directives) ─────────────────────
    def log_gcc_event(
        self,
        gcc_id: str,
        dominant_strategy: str,
        game_value: float,
        channel_directive: str,
        network_directive: str,
        energy_directive: str,
        hop_budget_fraction: float,
        rate_limit_aggressiveness: float,
        defender_mix: dict,
        attacker_mix: dict,
        attack_distribution: dict,
        kpi_snapshot: dict,
        batch_start_row: int           = 0,
        batch_end_row: int             = 0,
        event_id: str                  = None,
        correlation_id: str            = None,
        feedback_label: str            = "",
        review_notes: str              = "",
    ) -> dict:
        """
        Log one GCC Management Agent directive.
        New method — not in original — captures game-theoretic decisions.
        """
        if event_id      is None: event_id      = str(uuid.uuid4())
        if correlation_id is None: correlation_id = event_id
        return self.log_event({
            "event_id"                 : event_id,
            "correlation_id"           : correlation_id,
            "source_type"              : "gcc",
            "gcc_id"                   : gcc_id,
            "node_type"                : "GCC",
            "dominant_strategy"        : dominant_strategy,
            "game_value"               : game_value,
            "channel_directive"        : channel_directive,
            "network_directive"        : network_directive,
            "energy_directive"         : energy_directive,
            "hop_budget_fraction"      : hop_budget_fraction,
            "rate_limit_aggressiveness": rate_limit_aggressiveness,
            "defender_mix"             : self._safe_dict(defender_mix),
            "attacker_mix"             : self._safe_dict(attacker_mix),
            "attack_distribution"      : self._safe_dict(attack_distribution),
            "kpi_snapshot"             : self._safe_dict(kpi_snapshot),
            "batch_start_row"          : batch_start_row,
            "batch_end_row"            : batch_end_row,
            "feedback_label"           : feedback_label,
            "review_notes"             : review_notes,
        })

    # ── Dataframe accessors (mirrors original exactly) ─────────────────────────
    def get_logs_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.logs)

    def get_merkle_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.merkle_batches)

    # ── Save logs (mirrors original exactly) ───────────────────────────────────
    def save_logs(self, prefix: str = "logger_agent") -> dict:
        self.flush_batch_if_needed(force=True)
        logs_df   = pd.DataFrame(self.logs)
        merkle_df = pd.DataFrame(self.merkle_batches)
        logs_df.to_csv(   f"{prefix}_logs.csv",            index=False)
        merkle_df.to_csv( f"{prefix}_merkle_batches.csv",  index=False)
        with open(f"{prefix}_logs.jsonl",   "w", encoding="utf-8") as f:
            for log in self.logs:
                f.write(json.dumps(log, default=str) + "\n")
        with open(f"{prefix}_merkle_batches.jsonl","w",encoding="utf-8") as f:
            for batch in self.merkle_batches:
                f.write(json.dumps(batch, default=str) + "\n")
        print(f"Logs saved — {len(self.logs)} events | {len(self.merkle_batches)} Merkle batches.")
        return {
            "logs_csv"      : f"{prefix}_logs.csv",
            "logs_jsonl"    : f"{prefix}_logs.jsonl",
            "merkle_csv"    : f"{prefix}_merkle_batches.csv",
            "merkle_jsonl"  : f"{prefix}_merkle_batches.jsonl",
        }


def verify_integrity(logger: LoggerAgent) -> dict:
    """
    Verify integrity of all log records and Merkle batches.
    For each record: recompute SHA-256, compare to stored log_hash.
    For each batch : reconstruct Merkle root, compare to stored root.
    Returns a detailed integrity report.
    """
    print("[IntegrityVerifier] Verifying log hashes...")
    n_ok = n_fail = 0
    failed_events = []

    for log in logger.logs:
        stored_hash = log.get("log_hash", "")
        # Recompute over exactly the fields present when the hash was taken.
        # log_hash did not exist yet, and batch_id was None at hash time (it is
        # mutated to the real batch number later during flush) — exclude both.
        record_to_hash = {k: v for k, v in log.items()
                          if k not in ("log_hash", "batch_id")}
        recomputed     = hashlib.sha256(
            json.dumps(record_to_hash, sort_keys=True, default=str).encode()
        ).hexdigest()
        if recomputed == stored_hash:
            n_ok += 1
        else:
            n_fail += 1
            failed_events.append(log.get("event_id", "?"))

    print(f"  Log records  : {n_ok} OK | {n_fail} FAILED")

    print("[IntegrityVerifier] Verifying Merkle roots...")
    m_ok = m_fail = 0
    failed_batches = []

    for batch in logger.merkle_batches:
        stored_root = batch.get("merkle_root", "")
        hashes      = batch.get("record_hashes", [])
        recomputed  = logger.build_merkle_root(hashes)
        if recomputed == stored_root:
            m_ok += 1
        else:
            m_fail += 1
            failed_batches.append(batch.get("batch_id", "?"))

    print(f"  Merkle batches: {m_ok} OK | {m_fail} FAILED")

    all_ok = (n_fail == 0) and (m_fail == 0)
    status = "INTEGRITY_VERIFIED" if all_ok else "INTEGRITY_VIOLATION_DETECTED"
    print(f"\n  Status: {status}")

    return {
        "status"              : status,
        "log_records_ok"      : n_ok,
        "log_records_failed"  : n_fail,
        "failed_event_ids"    : failed_events[:20],
        "merkle_batches_ok"   : m_ok,
        "merkle_batches_failed": m_fail,
        "failed_batch_ids"    : failed_batches,
        "verified_at"         : datetime.now(timezone.utc).isoformat(),
    }


def run_logger_agent(
    detection_df:    pd.DataFrame,
    response_df:     pd.DataFrame,
    coordination_df: pd.DataFrame,
    management_df:   pd.DataFrame,
    batch_size:      int  = 10,
    agent_version:   str  = "v1.1",
    preview:         bool = False,
) -> tuple:
    """
    Full retrospective replay of all pipeline outputs through the Logger.
    Processes in chronological order: detection -> response -> coordination
    -> management. Each row calls the appropriate log_*_event() method.
    Mirrors the logger.save_logs() call in run_full_mas_pipeline().
    Returns (logger, traffic_sketch, phy_sketch, ddos_bloom, integrity_report).
    """
    logger        = LoggerAgent(batch_size=batch_size, agent_version=agent_version)
    traffic_sketch= TrafficSketchAggregator(window_seconds=10)
    phy_sketch    = PhysicalLayerSketch()
    ddos_bloom    = BloomFilter(capacity=10_000, error_rate=0.01)

    # ── Align detection and response on (timestamp, node_id) ──────────────────
    det  = detection_df.copy()
    resp = response_df.copy()
    det["timestamp"]  = pd.to_datetime(det["timestamp"],  errors="coerce")
    resp["timestamp"] = pd.to_datetime(resp["timestamp"], errors="coerce")

    shared = ["timestamp","node_id"]
    shared = [c for c in shared if c in det.columns and c in resp.columns]
    merged = det.merge(resp, on=shared, how="left",
                       suffixes=("","_resp")).sort_values("timestamp")\
                .reset_index(drop=True)

    n = len(merged)
    print(f"[Logger] Replaying {n} detection+response rows...")

    for i, (_, row) in enumerate(merged.iterrows(), start=1):
        node_id    = str(row.get("node_id",   "unknown"))
        node_type  = str(row.get("node_type", "uav"))
        label      = str(row.get("final_label","NORMAL"))
        severity   = str(row.get("severity",   "LOW"))

        # ── Traffic sketch ingestion ───────────────────────────────────────────
        traffic_sketch.ingest_row(row)

        # ── Physical layer sketch ──────────────────────────────────────────────
        phy_sketch.ingest_row(row, resp_row=row)

        # ── Bloom filter: DDoS source tracking ────────────────────────────────
        if label in {"DOS","DDOS","HYBRID_ATTACK"}:
            ddos_bloom.add(node_id)

        # ── Build nested summaries ─────────────────────────────────────────────
        sensor_summary = {
            "signal_trust_before"    : float(row.get("signal_trust_before", 1.0) or 1.0),
            "signal_trust_after"     : float(row.get("signal_trust_after",  1.0) or 1.0),
            "channel_mode"           : str(row.get("rf_channel_mode","NORMAL_OPERATION")),
            "lte_fallback_active"    : bool(row.get("lte_fallback_active",  False)),
            "snr_db"                 : float(row.get("snr_db",              10.0) or 10.0),
            "channel_occupancy_pct"  : float(row.get("channel_occupancy_pct",20.0) or 20.0),
        }
        routing_summary = {
            "network_trust_before"   : float(row.get("network_trust_before",1.0) or 1.0),
            "network_trust_after"    : float(row.get("network_trust_after", 1.0) or 1.0),
            "selected_action"        : str(row.get("net_selected_action","NONE")),
            "alert_delivery_mode"    : str(row.get("alert_delivery_mode","PRIMARY_ROUTE")),
            "packets_per_second"     : float(row.get("packets_per_second",  50.0) or 50.0),
            "src_ip_entropy"         : float(row.get("src_ip_entropy",       0.0) or 0.0),
        }
        status_summary = {
            "safety_override"        : bool(row.get("safety_override",    False)),
            "safety_constraint"      : str(row.get("safety_constraint",   "NONE")),
            "reactive_triggered"     : bool(row.get("reactive_triggered", False)),
            "notify_neighbors"       : bool(row.get("notify_neighbors",   False)),
            "notify_edge"            : bool(row.get("notify_edge",        False)),
            "notify_gcc"             : bool(row.get("notify_gcc",         False)),
        }

        logger.log_uav_ugv_event(
            node_id                = node_id,
            node_type              = node_type,
            final_label            = label,
            severity               = severity,
            jam_risk_score         = float(row.get("jam_risk_score",   0) or 0),
            dos_risk_score         = float(row.get("dos_risk_score",   0) or 0),
            hybrid_risk_score      = float(row.get("hybrid_risk_score",0) or 0),
            fusion_confidence      = float(row.get("fusion_confidence",0) or 0),
            detection_reason_codes = row.get("detection_reason_codes", []),
            rf_action_taken        = str(row.get("rf_action_taken","NO_RF_CHANGE")),
            net_action_taken       = str(row.get("net_action_taken","NO_NETWORK_CHANGE")),
            safety_override        = bool(row.get("safety_override",   False)),
            reactive_triggered     = bool(row.get("reactive_triggered",False)),
            response_status        = str(row.get("response_status","EXECUTED")),
            response_success       = bool(row.get("response_success",  True)),
            sensor_summary         = sensor_summary,
            routing_summary        = routing_summary,
            status_summary         = status_summary,
            trust_score_before     = float(row.get("signal_trust_before",1.0) or 1.0),
            trust_score_after      = float(row.get("signal_trust_after", 1.0) or 1.0),
            notified_neighbors     = [node_id] if bool(row.get("notify_neighbors")) else [],
            gcc_notified           = bool(row.get("notify_gcc", False)),
            alert_scope            = ("gcc" if bool(row.get("notify_gcc"))
                                      else "edge" if bool(row.get("notify_edge"))
                                      else "local"),
            correlation_id         = f"{node_id}_{row.get('timestamp','')}",
        )
        if preview and (i % 500 == 0 or i == n):
            print(f"  [UAV/UGV] Processed {i}/{n} rows")

    # ── Coordination events ────────────────────────────────────────────────────
    coord = coordination_df.copy()
    coord["timestamp"] = pd.to_datetime(coord["timestamp"], errors="coerce")
    coord = coord.dropna(subset=["timestamp"]).sort_values("timestamp")
    print(f"[Logger] Replaying {len(coord)} coordination rows...")

    for _, row in coord.iterrows():
        logger.log_edge_event(
            edge_id                        = "edge_1",
            final_label                    = str(row.get("final_label","NORMAL")),
            severity                       = str(row.get("severity","LOW")),
            correlated_attack_confirmation = bool(row.get("coordinated_attack_detected",False)),
            trust_score_before             = float(row.get("trust_before",1.0) or 1.0),
            trust_score_after              = float(row.get("trust_after", 1.0) or 1.0),
            coordinated_swarm_action       = str(row.get("coordination_action","NO_GLOBAL_CHANGE")),
            quarantine_decision            = bool(row.get("quarantine_decision",False)),
            rerouting_decision             = bool(row.get("rerouting_decision",False)),
            target_nodes                   = list(row.get("coordinated_nodes",[]) or []),
            response_status                = str(row.get("coordination_status","TRUSTED")),
            response_success               = bool(row.get("coordination_status","TRUSTED")
                                                   == "TRUSTED"),
            correlation_id                 = f"coord_{row.get('node_id','')}_{row.get('timestamp','')}",
            status_summary                 = {"evidence_score": float(row.get("evidence_score",0) or 0)},
            alert_scope                    = str(row.get("alert_scope","local")),
            gcc_notified                   = bool(row.get("gcc_notified",False)),
        )

    # ── Management directives ──────────────────────────────────────────────────
    mgmt = management_df.copy()
    mgmt["timestamp"] = pd.to_datetime(mgmt["timestamp"], errors="coerce")
    mgmt = mgmt.dropna(subset=["timestamp"]).sort_values("timestamp")
    print(f"[Logger] Replaying {len(mgmt)} management directive rows...")

    for _, row in mgmt.iterrows():
        def_mix = {
            "NORMAL_OPS"    : float(row.get("def_normal_ops",    0) or 0),
            "LIGHT_DEFENSE" : float(row.get("def_light_defense", 0) or 0),
            "FULL_DEFENSE"  : float(row.get("def_full_defense",  0) or 0),
            "EMERGENCY_MODE": float(row.get("def_emergency_mode",0) or 0),
        }
        kpi = {
            "pdr"                    : float(row.get("kpi_pdr",            0) or 0),
            "swarm_connectivity_ratio": float(row.get("kpi_connectivity",   0) or 0),
            "energy_overhead"        : float(row.get("kpi_energy_overhead", 0) or 0),
            "false_positive_rate"    : float(row.get("kpi_fp_rate",         0) or 0),
            "attack_rate"            : float(row.get("kpi_attack_rate",     0) or 0),
        }
        atk_dist = {
            "JAM_CHANNEL"   : float(row.get("atk_jam_fraction",0) or 0),
            "FLOOD_NETWORK" : float(row.get("atk_dos_fraction",0) or 0),
            "HYBRID_ATTACK" : float(row.get("atk_hyb_fraction",0) or 0),
        }
        logger.log_gcc_event(
            gcc_id                    = str(row.get("gcc_id","gcc_001")),
            dominant_strategy         = str(row.get("dominant_strategy","NORMAL_OPS")),
            game_value                = float(row.get("game_value",0) or 0),
            channel_directive         = str(row.get("channel_directive","NORMAL_CHANNEL_OPS")),
            network_directive         = str(row.get("network_directive","NORMAL_NETWORK_OPS")),
            energy_directive          = str(row.get("energy_directive","FULL_POWER_MODE")),
            hop_budget_fraction       = float(row.get("hop_budget_fraction",0) or 0),
            rate_limit_aggressiveness = float(row.get("rate_limit_aggressiveness",0) or 0),
            defender_mix              = def_mix,
            attacker_mix              = {},
            attack_distribution       = atk_dist,
            kpi_snapshot              = kpi,
            batch_start_row           = int(row.get("batch_start_row",0) or 0),
            batch_end_row             = int(row.get("batch_end_row",0)   or 0),
        )

    # Force-flush any remaining pending batch
    logger.flush_batch_if_needed(force=True)

    # Finalise sketches
    traffic_sketch_summary = traffic_sketch.finalise()
    phy_sketch_summary     = phy_sketch.finalise()

    print(f"[Logger] Total events logged: {len(logger.logs)}")
    print(f"[Logger] Merkle batches     : {len(logger.merkle_batches)}")

    # Integrity check
    integrity_report = verify_integrity(logger)

    return logger, traffic_sketch_summary, phy_sketch_summary, ddos_bloom, integrity_report


# ── Run logger ────────────────────────────────────────────────────────────────

def analyse_logs(logger: LoggerAgent) -> dict:
    """Run analytics over the full log DataFrame. Mirrors original health summary."""
    logs_df = logger.get_logs_dataframe()
    if len(logs_df) == 0:
        print("No logs to analyse."); return {}

    print("=" * 65)
    print("  LOG ANALYTICS")
    print("=" * 65)

    # Source type distribution
    print("\n  Events by source type:")
    print(logs_df["source_type"].value_counts().to_string())

    # Attack label distribution (uav_ugv events only)
    uav_logs = logs_df[logs_df["source_type"] == "uav_ugv"]
    if len(uav_logs) > 0:
        print("\n  UAV/UGV events by final_label:")
        print(uav_logs["final_label"].value_counts().to_string())
        print("\n  UAV/UGV events by severity:")
        if "severity" in uav_logs.columns:
            print(uav_logs["severity"].value_counts().to_string())

    # Coordination events
    edge_logs = logs_df[logs_df["source_type"] == "edge"]
    if len(edge_logs) > 0:
        print("\n  Edge coordination events by swarm action:")
        if "coordinated_swarm_action" in edge_logs.columns:
            print(edge_logs["coordinated_swarm_action"].value_counts().to_string())
        print(f"  Coordinated attacks confirmed : "
              f"{edge_logs.get('correlated_attack_confirmation', pd.Series(False)).sum()}")

    # GCC management events
    gcc_logs = logs_df[logs_df["source_type"] == "gcc"]
    if len(gcc_logs) > 0:
        print("\n  GCC management events by dominant strategy:")
        if "dominant_strategy" in gcc_logs.columns:
            print(gcc_logs["dominant_strategy"].value_counts().to_string())

    # Alert scope coverage
    print("\n  Alert scope distribution (uav_ugv events):")
    if "alert_scope" in uav_logs.columns:
        print(uav_logs["alert_scope"].value_counts().to_string())

    # GCC notification rate
    if "gcc_notified" in uav_logs.columns:
        gcc_rate = uav_logs["gcc_notified"].apply(lambda x: bool(x)).mean()
        print(f"\n  GCC notification rate (uav_ugv): {gcc_rate:.4f} ({gcc_rate*100:.1f}%)")

    # Merkle batch coverage
    merkle_df = logger.get_merkle_dataframe()
    print(f"\n  Merkle batches: {len(merkle_df)}")
    if len(merkle_df) > 0:
        print(f"  Records covered: {merkle_df['batch_size'].sum()}")
        print(f"  Source types spanned: "
              f"{set(t for ts in merkle_df['source_types'] for t in ts)}")

    summary = {
        "total_events"     : len(logs_df),
        "uav_ugv_events"   : len(uav_logs),
        "edge_events"      : len(edge_logs),
        "gcc_events"       : len(gcc_logs),
        "merkle_batches"   : len(merkle_df),
    }
    return summary


