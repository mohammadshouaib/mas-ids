"""
mas_ids.agents.coordination_agent
=================================
Agent 5 — Coordination & Management.

Coordination: Beta Trust Model + Bayesian inference; coordinated-attack detection;
swarm-level coordination actions.

Management: game-theoretic Nash Equilibrium (LP solver) over a 3-attacker x
4-defender payoff matrix; adaptive KPI tracking and resource allocation.

Public API:
  BetaTrustModel, SimpleBayesianCoordinator, CoordinationAgent,
  run_coordination_agent, ATTACKER_STRATEGIES, DEFENDER_STRATEGIES, BASE_PAYOFF,
  solve_nash_equilibrium, ManagementAgent, run_management_agent,
  print_kpi_dashboard, coordinate_single
"""
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone
from scipy.optimize import linprog

from ..config import LABEL_MAP, KPI_WINDOW
from ..utils import NullLogger


class BetaTrustModel:
    """
    Beta distribution trust model per node.
    Mirrors BetaTrustModel from the original exactly.
    score(node_id) = alpha / (alpha + beta) in [0, 1].
    """
    def __init__(self, alpha0: float = 5.0, beta0: float = 1.0):
        self.alpha = defaultdict(lambda: float(alpha0))
        self.beta  = defaultdict(lambda: float(beta0))

    def score(self, node_id) -> float:
        a, b = self.alpha[node_id], self.beta[node_id]
        return float(a / (a + b))

    def update(self, node_id,
               positive_evidence: float = 0.0,
               negative_evidence: float = 0.0) -> float:
        self.alpha[node_id] += max(0.0, float(positive_evidence))
        self.beta[node_id]  += max(0.0, float(negative_evidence))
        return self.score(node_id)


class SimpleBayesianCoordinator:
    """
    Bayesian attack-type classifier: isolated fault vs coordinated attack.
    Mirrors SimpleBayesianCoordinator from the original, adapted for
    DoS/DDoS + Jamming attack types and column names.
    """
    def __init__(self):
        self.prior_isolated    = 0.30
        self.prior_coordinated = 0.20

    @staticmethod
    def clip_prob(x: float) -> float:
        return float(np.clip(x, 0.001, 0.999))

    def infer(self, node_row: dict, current_window: list) -> tuple:
        """
        Compute P(isolated_fault | evidence) and P(coordinated_attack | evidence).
        Returns (p_isolated, p_coordinated).
        Mirrors SimpleBayesianCoordinator.infer() from the original exactly,
        adapted for jamming/DoS/DDoS signals.
        """
        fusion_conf   = float(node_row.get("fusion_confidence", 0.0))
        jam_risk      = float(node_row.get("jam_risk_score",   0.0))
        dos_risk      = float(node_row.get("dos_risk_score",   0.0))
        cl_score      = float(node_row.get("cross_layer_anomaly_score", 0.0))
        swarm_flag    = float(node_row.get("swarm_evidence_flag",       0.0))
        notify_gcc    = bool(node_row.get("notify_gcc", False))

        # Count distinct suspicious nodes in current time window
        attack_labels = {"JAMMING","DOS","DDOS","HYBRID_ATTACK","SUSPICIOUS"}
        suspicious_nodes = [
            r for r in current_window
            if str(r.get("final_label","NORMAL")) in attack_labels
        ]
        multi_node_factor = min(
            len(set(str(r.get("node_id","")) for r in suspicious_nodes)) / 5.0,
            1.0
        )

        # Likelihood of coordinated attack (multi-source, swarm-wide)
        l_coordinated = self.clip_prob(
            0.25 * fusion_conf +
            0.20 * max(jam_risk, dos_risk) +
            0.20 * swarm_flag +
            0.20 * multi_node_factor +
            0.10 * cl_score +
            0.05 * float(notify_gcc)
        )
        # Likelihood of isolated fault (single-node)
        l_isolated = self.clip_prob(
            0.30 * fusion_conf +
            0.25 * max(jam_risk, dos_risk) +
            0.20 * (1.0 - multi_node_factor) +
            0.10 * (1.0 - swarm_flag) +
            0.10 * float(notify_gcc) +
            0.05 * cl_score
        )

        num_coord   = self.prior_coordinated * l_coordinated
        num_iso     = self.prior_isolated    * l_isolated
        denom       = num_coord + num_iso + 1e-9
        return float(num_iso / denom), float(num_coord / denom)


class CoordinationAgent:
    """
    Swarm-level coordination agent.
    Mirrors CoordinationAgent from the original (Cell 16) exactly,
    adapted for DoS/DDoS + Jamming labels and column names.
    """

    # Coordination actions available (extended for DoS/DDoS + Jamming)
    COORDINATION_ACTIONS = [
        "NO_GLOBAL_CHANGE",
        "MONITOR_AND_LIMIT_ROUTING",
        "RESTRICT_ROLE_AND_NOTIFY_NEIGHBORS",
        "RESTRICT_SWARM_AND_NOTIFY_GCC",
        "ISOLATE_NODE_AND_BROADCAST",
        "GLOBAL_CONTAINMENT_AND_REROUTE",
        "ACTIVATE_SWARM_WIDE_CHANNEL_HOP",   # Jamming-specific
        "SWARM_WIDE_RATE_LIMIT_AND_ISOLATE",  # DDoS-specific
    ]

    def __init__(self, logger=None,
                 coordinated_threshold: int   = 3,
                 confidence_threshold:  float = 0.50,
                 edge_id: str = "edge_1"):
        self.trust_model           = BetaTrustModel(alpha0=5.0, beta0=1.0)
        self.bayes_model           = SimpleBayesianCoordinator()
        self.node_history          = defaultdict(list)
        self.coordinated_threshold = int(coordinated_threshold)
        self.confidence_threshold  = float(confidence_threshold)
        self.logger                = logger or NullLogger()
        self.edge_id               = edge_id

    # ── Safe helpers (mirrors original exactly) ────────────────────────────────
    @staticmethod
    def _sf(v, d=0.0):
        try: return float(d) if (v is None or v == "") else float(v)
        except: return float(d)

    @staticmethod
    def _sl(v):
        if v is None: return []
        return v if isinstance(v, list) else [v]

    @staticmethod
    def _sb(v, d=False):
        return d if v is None else bool(v)

    @staticmethod
    def _node_type(row):
        nt = row.get("node_type")
        if nt is None or (isinstance(nt, float) and pd.isna(nt)): return "UAV"
        return str(nt)

    # ── Trust evidence builder (adapted for DoS/DDoS + Jamming) ───────────────
    def build_trust_evidence(self, merged_row: dict) -> tuple:
        """
        Compute (positive_evidence, negative_evidence, evidence_score) for one row.
        Mirrors build_trust_evidence() from the original exactly,
        using DoS/DDoS + Jamming risk scores instead of GPS/routing scores.
        """
        jam_risk   = self._sf(merged_row.get("jam_risk_score",               0.0))
        dos_risk   = self._sf(merged_row.get("dos_risk_score",               0.0))
        hyb_risk   = self._sf(merged_row.get("hybrid_risk_score",            0.0))
        fusion     = self._sf(merged_row.get("fusion_confidence",            0.0))
        cl_score   = self._sf(merged_row.get("cross_layer_anomaly_score",    0.0))
        swarm_anom = self._sf(merged_row.get("swarm_consensus_anomaly_score",0.0))
        consec     = min(self._sf(merged_row.get("consecutive_attack_count", 0.0)), 20.0) / 20.0

        evidence_score = (
            0.22 * jam_risk  +
            0.22 * dos_risk  +
            0.12 * hyb_risk  +
            0.15 * fusion    +
            0.10 * cl_score  +
            0.10 * swarm_anom +
            0.09 * consec
        )

        positive = max(0.0, 1.0 - evidence_score)
        negative = evidence_score
        label    = str(merged_row.get("final_label", "NORMAL"))

        # Label-based trust adjustment (mirrors original exactly)
        if   label in {"JAMMING", "DOS"}:    negative += 0.75
        elif label == "DDOS":                negative += 0.85
        elif label == "HYBRID_ATTACK":       negative += 1.00
        elif label == "SUSPICIOUS":          negative += 0.30; positive += 0.10
        else:                                positive  += 0.75

        # Response action adjustments
        rf_action  = str(merged_row.get("rf_action_taken",  ""))
        net_action = str(merged_row.get("net_action_taken", ""))
        if rf_action  in {"ACTIVATE_ANTI_JAMMING_HOP", "SWITCH_TO_LTE_FALLBACK"}:
            negative += 0.20
        if net_action in {"DDOS_MITIGATION_ACTIVE", "HYBRID_CONTAIN_AND_REROUTE"}:
            negative += 0.20

        # Safety override signals a hard-constraint violation — extra penalty
        if self._sb(merged_row.get("safety_override")):
            negative += 0.15

        # Clean confirmed normal — extra positive boost (mirrors Issue 3 fix)
        if rf_action == "NO_RF_CHANGE" and net_action == "NO_NETWORK_CHANGE" \
                and label == "NORMAL":
            positive += 0.20

        if self._sb(merged_row.get("notify_gcc")):
            negative += 0.10

        return positive, negative, float(np.clip(evidence_score, 0.0, 1.0))

    # ── Coordinated attack detector (mirrors original exactly) ────────────────
    def detect_coordinated_attack(self, current_window: list) -> tuple:
        """
        Returns (is_coordinated, attack_node_count, attack_node_ids).
        Mirrors detect_coordinated_attack() from the original.
        """
        attack_labels = {"JAMMING","DOS","DDOS","HYBRID_ATTACK"}
        attack_rows   = [
            r for r in current_window
            if r.get("final_label") in attack_labels
            and self._sf(r.get("fusion_confidence")) >= self.confidence_threshold
        ]
        attack_nodes = sorted(set(str(r["node_id"]) for r in attack_rows))
        return len(attack_nodes) >= self.coordinated_threshold, len(attack_nodes), attack_nodes

    # ── Coordination action selector (mirrors original, extended) ─────────────
    def recommended_action(
        self, trust_after: float, label: str,
        is_coordinated: bool, p_coordinated: float
    ) -> str:
        """
        Select swarm-level coordination action.
        Mirrors recommended_coordinated_action() from the original,
        extended with jamming/DDoS-specific actions.
        """
        attack_labels = {"JAMMING","DOS","DDOS","HYBRID_ATTACK"}

        # Highest escalation: confirmed coordinated + hybrid/DDoS
        if is_coordinated and label in {"HYBRID_ATTACK","DDOS"}:
            return "GLOBAL_CONTAINMENT_AND_REROUTE"
        # High-confidence coordinated attack
        if p_coordinated >= 0.65 and label in attack_labels:
            if label == "JAMMING":
                return "ACTIVATE_SWARM_WIDE_CHANNEL_HOP"
            if label in {"DDOS","HYBRID_ATTACK"}:
                return "SWARM_WIDE_RATE_LIMIT_AND_ISOLATE"
            return "GLOBAL_CONTAINMENT_AND_REROUTE"
        # Very low trust — isolate
        if trust_after < 0.35:
            return "ISOLATE_NODE_AND_BROADCAST"
        # Coordinated but moderate
        if is_coordinated and label in attack_labels:
            return "RESTRICT_SWARM_AND_NOTIFY_GCC"
        # Confirmed attack, single-node
        if label in attack_labels:
            return "RESTRICT_ROLE_AND_NOTIFY_NEIGHBORS"
        # Low trust but not isolated
        if trust_after < 0.60:
            return "MONITOR_AND_LIMIT_ROUTING"
        return "NO_GLOBAL_CHANGE"

    # ── Node status derivation (mirrors original exactly) ─────────────────────
    def derive_status(self, label: str, trust_after: float) -> str:
        if trust_after < 0.35: return "ISOLATED"
        if label in {"JAMMING","DOS","DDOS","HYBRID_ATTACK"}: return "RESTRICTED"
        if trust_after < 0.60: return "RESTRICTED"
        if trust_after < 0.80: return "MONITORED"
        return "TRUSTED"

    # ── Master coordinate() loop (mirrors original exactly) ───────────────────
    def coordinate(self,
                   detection_df: pd.DataFrame,
                   response_df:  pd.DataFrame) -> list:
        """
        Main coordination loop over all windows.
        Mirrors CoordinationAgent.coordinate() from the original exactly.
        Returns a list of coordination log dicts.
        """
        # NOTE: detection_df and response_df are ROW-PARALLEL — response_df is
        # produced by run_response_agent_inference(detection_df, ...), so row i
        # of one corresponds to row i of the other. The previous implementation
        # merged on ["timestamp","node_id","final_label","severity"], but those
        # columns are NOT unique (1-second windows repeat, often a single node,
        # only a few distinct labels/severities). pandas then emits the Cartesian
        # product of every colliding key group — a many-to-many blow-up that turns
        # ~200k rows into tens of millions and exhausts RAM (Kaggle OOM).
        #
        # The correct alignment is positional. We concat the response-only
        # columns onto detection by row index (cheap, exactly len(det) rows),
        # avoiding both the copies and the explosion.
        det = detection_df.reset_index(drop=True)
        det = det.assign(timestamp=pd.to_datetime(det["timestamp"], errors="coerce"))

        # Bring over only the response columns not already in detection, suffixing
        # any genuine name clashes with "_resp" to preserve the old row schema.
        resp = response_df.reset_index(drop=True)
        if len(resp) != len(det):
            # Lengths differ (shouldn't happen in the pipeline) — fall back to a
            # 1:1 merge on a stable row id rather than the non-unique columns.
            det["_row_id"]  = np.arange(len(det))
            resp = resp.copy(); resp["_row_id"] = np.arange(len(resp))
            new_cols = [c for c in resp.columns
                        if c not in det.columns or c == "_row_id"]
            merged = det.merge(resp[new_cols], on="_row_id", how="left",
                               suffixes=("", "_resp")).drop(columns="_row_id")
        else:
            new_cols = [c for c in resp.columns if c not in det.columns]
            clash    = [c for c in resp.columns
                        if c in det.columns and c not in
                        ("timestamp", "node_id", "final_label", "severity")]
            add = resp[new_cols]
            if clash:
                add = pd.concat(
                    [add, resp[clash].add_suffix("_resp")], axis=1)
            merged = pd.concat([det, add], axis=1)

        merged = merged.dropna(subset=["timestamp"])
        merged = merged.sort_values(["timestamp", "node_id"]).reset_index(drop=True)
        del det, resp

        logs   = []
        window = []   # rolling 10-row window for coordinated-attack detection

        for row in merged.itertuples(index=False):
            row = dict(row._asdict())
            node_id      = str(row.get("node_id", "unknown"))
            node_type    = self._node_type(row)
            trust_before = self.trust_model.score(node_id)

            pos_ev, neg_ev, ev_score = self.build_trust_evidence(row)
            trust_after  = self.trust_model.update(node_id, pos_ev, neg_ev)
            status       = self.derive_status(str(row.get("final_label","NORMAL")),
                                               trust_after)

            summary = {
                "timestamp"             : row["timestamp"],
                "node_id"               : node_id,
                "node_type"             : node_type,
                "final_label"           : str(row.get("final_label","NORMAL")),
                "severity"              : str(row.get("severity","LOW")),
                "fusion_confidence"     : self._sf(row.get("fusion_confidence")),
                "trust_before"          : float(trust_before),
                "trust_after"           : float(trust_after),
                "evidence_score"        : float(ev_score),
                "coordination_status"   : status,
                "reason_codes"          : self._sl(row.get("detection_reason_codes",[])),
                "rf_action_taken"       : str(row.get("rf_action_taken",  "NONE")),
                "net_action_taken"      : str(row.get("net_action_taken", "NONE")),
                "safety_override"       : self._sb(row.get("safety_override")),
                "reactive_triggered"    : self._sb(row.get("reactive_triggered")),
                "notify_neighbors"      : self._sb(row.get("notify_neighbors")),
                "notify_edge"           : self._sb(row.get("notify_edge")),
                "notify_gcc"            : self._sb(row.get("notify_gcc")),
                "jam_risk_score"        : self._sf(row.get("jam_risk_score")),
                "dos_risk_score"        : self._sf(row.get("dos_risk_score")),
                "cross_layer_anomaly_score": self._sf(row.get("cross_layer_anomaly_score")),
                "swarm_evidence_flag"   : self._sf(row.get("swarm_consensus_anomaly_score")),
            }

            self.node_history[node_id].append(summary)
            window.append(summary)
            if len(window) > 10:
                window.pop(0)

            is_coord, n_atk, atk_nodes = self.detect_coordinated_attack(window)
            p_iso, p_coord = self.bayes_model.infer(summary, window)
            coord_action   = self.recommended_action(
                trust_after, str(row.get("final_label","NORMAL")),
                is_coord, p_coord
            )

            quarantine  = bool(status == "ISOLATED")
            rerouting   = bool(coord_action in {
                "GLOBAL_CONTAINMENT_AND_REROUTE",
                "RESTRICT_SWARM_AND_NOTIFY_GCC",
                "MONITOR_AND_LIMIT_ROUTING",
                "SWARM_WIDE_RATE_LIMIT_AND_ISOLATE",
            })
            gcc_notified = bool(
                summary["notify_gcc"] or
                coord_action in {"GLOBAL_CONTAINMENT_AND_REROUTE",
                                 "RESTRICT_SWARM_AND_NOTIFY_GCC",
                                 "ACTIVATE_SWARM_WIDE_CHANNEL_HOP",
                                 "SWARM_WIDE_RATE_LIMIT_AND_ISOLATE"}
            )

            summary.update({
                "bayes_p_isolated"           : float(p_iso),
                "bayes_p_coordinated"        : float(p_coord),
                "coordinated_attack_detected": bool(is_coord),
                "coordinated_node_count"     : int(n_atk),
                "coordinated_nodes"          : atk_nodes,
                "coordination_action"        : coord_action,
                "quarantine_decision"        : quarantine,
                "rerouting_decision"         : rerouting,
                "gcc_notified"               : gcc_notified,
                "alert_scope"                : ("gcc" if gcc_notified
                                              else "edge" if is_coord
                                              else "local"),
            })
            logs.append(summary)

            # Log to Logger Agent
            self.logger.log_edge_event(
                edge_id=self.edge_id,
                final_label=row.get("final_label","NORMAL"),
                severity=row.get("severity","LOW"),
                correlated_attack_confirmation=bool(is_coord),
                trust_score_before=float(trust_before),
                trust_score_after=float(trust_after),
                coordinated_swarm_action=coord_action,
                quarantine_decision=quarantine,
                rerouting_decision=rerouting,
                target_nodes=[f"{node_type.lower()}_{x}" for x in atk_nodes],
                response_status=status,
                response_success=(status == "TRUSTED"),
                correlation_id=f"coord_{node_id}_{row['timestamp']}",
                status_summary={"evidence_score": float(ev_score)},
                alert_scope=summary["alert_scope"],
                notified_neighbors=[f"{node_type.lower()}_{x}" for x in atk_nodes],
                gcc_notified=gcc_notified,
            )
        return logs


def run_coordination_agent(
    detection_df: pd.DataFrame,
    response_df:  pd.DataFrame,
    logger                = None,
    save_outputs: bool    = False,
    preview: bool         = False,
    coordinated_threshold : int   = 3,
    confidence_threshold  : float = 0.50,
    edge_id: str          = "edge_1",
) -> tuple:
    """
    Mirrors run_coordination_agent() from the original exactly.
    Returns (coord_agent, coordination_results,
             coordination_df, trust_registry_df, logger).
    """
    coord_agent = CoordinationAgent(
        logger=logger or NullLogger(),
        coordinated_threshold=coordinated_threshold,
        confidence_threshold=confidence_threshold,
        edge_id=edge_id,
    )
    results = coord_agent.coordinate(detection_df, response_df)
    coord_df = pd.DataFrame(results)

    trust_registry = [
        {
            "node_id"      : nid,
            "final_trust"  : coord_agent.trust_model.score(nid),
            "alpha"        : coord_agent.trust_model.alpha[nid],
            "beta"         : coord_agent.trust_model.beta[nid],
            "history_count": len(coord_agent.node_history[nid]),
        }
        for nid in sorted(coord_agent.node_history)
    ]
    trust_df = pd.DataFrame(trust_registry)

    if save_outputs:
        coord_df.to_csv("coordination_output.csv", index=False)
        with open("coordination_output.jsonl","w",encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, default=str)+"\n")
        trust_df.to_csv("coordination_trust_registry.csv", index=False)

    if preview:
        print("[Coordination] Action distribution:")
        print(coord_df["coordination_action"].value_counts().to_string())
        print("\n[Coordination] Status distribution:")
        print(coord_df["coordination_status"].value_counts().to_string())
        print("\n[Coordination] Trust registry:")
        print(trust_df.round(4).to_string())

    return coord_agent, results, coord_df, trust_df, logger


# ── Run coordination ──────────────────────────────────────────────────────────

# ── Attacker and Defender strategy spaces ────────────────────────────────────
ATTACKER_STRATEGIES = [
    "JAM_CHANNEL",      # RF jamming — targets physical layer
    "FLOOD_NETWORK",    # DoS/DDoS — targets network/transport layer
    "HYBRID_ATTACK",    # Both simultaneously
]

DEFENDER_STRATEGIES = [
    "NORMAL_OPS",       # No extra defense — minimal energy cost
    "LIGHT_DEFENSE",    # Channel monitoring + rate-limiting
    "FULL_DEFENSE",     # Freq-hopping + IP blocking + rerouting
    "EMERGENCY_MODE",   # LTE fallback + full isolation + GCC alert
]

N_ATK = len(ATTACKER_STRATEGIES)
N_DEF = len(DEFENDER_STRATEGIES)

# ── Payoff matrix: defender utility U[attacker_i][defender_j] ────────────────
# Rows = attacker strategies, Cols = defender strategies
# Positive = defender wins, Negative = attacker wins
# Values derived from balancing: availability, energy, coverage
#
#               NORMAL  LIGHT  FULL  EMERGENCY
BASE_PAYOFF = np.array([
    # JAM_CHANNEL: light and full defense are effective; normal is poor
    [-0.8,  0.3,  0.7,  0.4],
    # FLOOD_NETWORK: rate-limiting (light) helps; full and emergency are costly
    [-0.6,  0.5,  0.6,  0.3],
    # HYBRID_ATTACK: only full/emergency helps; normal is catastrophic
    [-1.0, -0.1,  0.5,  0.6],
], dtype=float)


def solve_nash_equilibrium(payoff: np.ndarray) -> tuple:
    """
    Solve the two-player zero-sum game for Nash Equilibrium mixed strategies.
    Uses linear programming (minimax theorem).

    Parameters
    ----------
    payoff : (n_attacker, n_defender) array — defender utility

    Returns
    -------
    (defender_mix, attacker_mix, game_value)
      defender_mix : (n_defender,) mixed strategy for defender
      attacker_mix : (n_attacker,) mixed strategy for attacker
      game_value   : expected payoff at Nash Equilibrium
    """
    n_a, n_d = payoff.shape

    # ── Defender: maximise minimum gain (maximin) ─────────────────────────────
    # min_{q} max_{i} sum_j q_j * U[i,j] == max_{q} v
    # LP: min -v  s.t.  sum_j U[i,j]*q_j >= v  forall i,  sum q_j = 1, q >= 0
    # Variables: [q_0..q_{n_d-1}, v]
    c_def = np.zeros(n_d + 1); c_def[-1] = -1.0  # maximise v
    # Constraint: U[i,:] @ q - v >= 0  forall i
    A_ub_def = np.hstack([-payoff, np.ones((n_a, 1))])   # -U @ q + v <= 0
    b_ub_def = np.zeros(n_a)
    A_eq_def = np.ones((1, n_d + 1)); A_eq_def[0, -1] = 0.0
    b_eq_def = np.array([1.0])
    bounds_def = [(0, None)] * n_d + [(None, None)]

    res_def = linprog(
        c_def, A_ub=A_ub_def, b_ub=b_ub_def,
        A_eq=A_eq_def, b_eq=b_eq_def,
        bounds=bounds_def, method="highs"
    )
    if res_def.success:
        defender_mix = np.clip(res_def.x[:n_d], 0, None)
        defender_mix /= defender_mix.sum() + 1e-9
        game_value    = float(-res_def.fun)
    else:
        # Fallback: uniform mix
        defender_mix = np.ones(n_d) / n_d
        game_value   = float(np.mean(payoff))

    # ── Attacker: minimise maximum loss (minimax) ─────────────────────────────
    c_atk = np.zeros(n_a + 1); c_atk[-1] = 1.0   # minimise v
    A_ub_atk = np.hstack([payoff.T, -np.ones((n_d, 1))])  # U.T @ p - v <= 0
    b_ub_atk = np.zeros(n_d)
    A_eq_atk = np.ones((1, n_a + 1)); A_eq_atk[0, -1] = 0.0
    b_eq_atk = np.array([1.0])
    bounds_atk = [(0, None)] * n_a + [(None, None)]

    res_atk = linprog(
        c_atk, A_ub=A_ub_atk, b_ub=b_ub_atk,
        A_eq=A_eq_atk, b_eq=b_eq_atk,
        bounds=bounds_atk, method="highs"
    )
    if res_atk.success:
        attacker_mix = np.clip(res_atk.x[:n_a], 0, None)
        attacker_mix /= attacker_mix.sum() + 1e-9
    else:
        attacker_mix = np.ones(n_a) / n_a

    return defender_mix, attacker_mix, game_value


# ── Run base Nash equilibrium ─────────────────────────────────────────────────


class ManagementAgent:
    """
    Game-theoretic management agent.
    Balances availability, energy efficiency, and swarm coverage under attack.
    Implements the strategic coordination layer described in the requirements spec.
    """

    # KPI column mapping (from detection/response/coordination DataFrames)
    KPI_COLS = {
        "pdr"                : "packet_delivery_ratio",
        "snr"                : "snr_db",
        "channel_occ"        : "channel_occupancy_pct",
        "fusion_confidence"  : "fusion_confidence",
        "jam_risk"           : "jam_risk_score",
        "dos_risk"           : "dos_risk_score",
        "consecutive_attacks": "consecutive_attack_count",
    }

    # Defender strategy -> resource allocation weights
    STRATEGY_ALLOCATIONS = {
        "NORMAL_OPS"     : {"hop_budget":0.00,"rate_limit":0.0,"relay_factor":1.0,"energy_save":0.20},
        "LIGHT_DEFENSE"  : {"hop_budget":0.20,"rate_limit":0.3,"relay_factor":1.2,"energy_save":0.10},
        "FULL_DEFENSE"   : {"hop_budget":0.60,"rate_limit":0.7,"relay_factor":1.5,"energy_save":0.00},
        "EMERGENCY_MODE" : {"hop_budget":1.00,"rate_limit":1.0,"relay_factor":2.0,"energy_save":0.00},
    }

    def __init__(self, logger=None, gcc_id: str = "gcc_001",
                 kpi_window: int = 30):
        self.logger         = logger or NullLogger()
        self.gcc_id         = gcc_id
        self.kpi_window     = kpi_window
        self._payoff        = BASE_PAYOFF.copy()
        self._kpi_history   : list = []
        self._decision_log  : list = []
        self._attack_counts : dict = defaultdict(int)
        self._window_count  : int  = 0

    # ── KPI measurement ────────────────────────────────────────────────────────
    def measure_kpis(self,
                     detection_df: pd.DataFrame,
                     response_df:  pd.DataFrame,
                     coord_df:     pd.DataFrame) -> dict:
        """
        Compute current KPI snapshot from the three DataFrames.
        All KPIs are normalised to [0, 1] where higher = better.
        """
        def safe_mean(df, col, default=0.5):
            if col not in df.columns: return default
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(s.mean()) if len(s) > 0 else default

        # Detection quality
        attack_mask = detection_df["final_label"].isin(
            ["JAMMING","DOS","DDOS","HYBRID_ATTACK"]
        ) if "final_label" in detection_df.columns else pd.Series(False, index=detection_df.index)
        n_total   = max(len(detection_df), 1)
        attack_rt = float(attack_mask.sum() / n_total)

        # PDR: from detection features
        pdr = safe_mean(detection_df, "packet_delivery_ratio", 0.95)

        # Connectivity ratio: fraction of nodes with TRUSTED status
        if "coordination_status" in coord_df.columns:
            trusted_frac = float(
                (coord_df["coordination_status"] == "TRUSTED").mean()
            )
        else:
            trusted_frac = 1.0 - attack_rt

        # Energy overhead: fraction of nodes in EMERGENCY_MODE or LTE fallback
        energy_overhead = 0.0
        if "lte_fallback_active" in response_df.columns:
            energy_overhead = float(
                pd.to_numeric(response_df["lte_fallback_active"], errors="coerce")
                .fillna(0).mean()
            )

        # Detection latency proxy: consecutive attack count (lower = faster)
        consec_mean = safe_mean(detection_df, "consecutive_attack_count", 0)
        detect_latency_norm = float(np.clip(1.0 - consec_mean / 20.0, 0, 1))

        # False positive proxy: SUSPICIOUS or NORMAL rows that triggered GCC
        if "notify_gcc" in response_df.columns and "final_label" in response_df.columns:
            fp_mask = (
                response_df["final_label"].isin(["NORMAL","SUSPICIOUS"]) &
                pd.to_numeric(response_df["notify_gcc"], errors="coerce").fillna(0).astype(bool)
            )
            fp_rate = float(fp_mask.mean())
        else:
            fp_rate = 0.0

        # Response time proxy: fraction of HIGH attacks that got GCC notified
        if "severity" in response_df.columns and "notify_gcc" in response_df.columns:
            high_mask = response_df["severity"] == "HIGH"
            if high_mask.sum() > 0:
                response_coverage = float(
                    pd.to_numeric(
                        response_df.loc[high_mask, "notify_gcc"],
                        errors="coerce"
                    ).fillna(0).mean()
                )
            else:
                response_coverage = 1.0
        else:
            response_coverage = 1.0

        kpis = {
            "pdr"                    : round(pdr,                   4),
            "swarm_connectivity_ratio": round(trusted_frac,          4),
            "energy_overhead"        : round(energy_overhead,        4),
            "detection_latency_norm" : round(detect_latency_norm,    4),
            "false_positive_rate"    : round(fp_rate,                4),
            "response_coverage"      : round(response_coverage,      4),
            "attack_rate"            : round(attack_rt,              4),
            "timestamp"              : datetime.now(timezone.utc).isoformat(),
        }
        self._kpi_history.append(kpis)
        return kpis

    # ── Dynamic payoff update ──────────────────────────────────────────────────
    def update_payoff(self, kpis: dict, attack_distribution: dict) -> np.ndarray:
        """
        Adjust payoff matrix based on observed KPIs and current attack proportions.
        High PDR / connectivity = defender has upper hand → increase payoff.
        High energy overhead = defense is costly → reduce payoff.
        """
        payoff = BASE_PAYOFF.copy()

        pdr_bonus       = (kpis.get("pdr",             0.9) - 0.5) * 0.4
        conn_bonus      = (kpis.get("swarm_connectivity_ratio", 0.8) - 0.5) * 0.3
        energy_penalty  = kpis.get("energy_overhead", 0.0) * 0.3
        fp_penalty      = kpis.get("false_positive_rate", 0.0) * 0.2

        # Apply uniform adjustments
        payoff += pdr_bonus + conn_bonus - energy_penalty - fp_penalty

        # Scale rows by observed attack frequency
        jam_p  = attack_distribution.get("JAM_CHANNEL",   1/3)
        dos_p  = attack_distribution.get("FLOOD_NETWORK", 1/3)
        hyb_p  = attack_distribution.get("HYBRID_ATTACK", 1/3)
        payoff[0] *= (1.0 + jam_p)
        payoff[1] *= (1.0 + dos_p)
        payoff[2] *= (1.0 + hyb_p)

        self._payoff = np.clip(payoff, -2.0, 2.0)
        return self._payoff

    # ── Strategy -> resource allocation translation ────────────────────────────
    def allocate_resources(self, defender_mix: np.ndarray) -> dict:
        """
        Translate the mixed defender strategy into concrete resource allocations.
        Each allocation is the probability-weighted average of strategy-specific values.
        """
        alloc = {"hop_budget":0.0,"rate_limit":0.0,"relay_factor":0.0,"energy_save":0.0}
        for prob, strategy in zip(defender_mix, DEFENDER_STRATEGIES):
            for k, v in self.STRATEGY_ALLOCATIONS[strategy].items():
                alloc[k] += prob * v
        return {k: round(v, 4) for k, v in alloc.items()}

    # ── Swarm directive generation ─────────────────────────────────────────────
    def generate_directive(
        self, defender_mix: np.ndarray, alloc: dict,
        kpis: dict, dominant_strategy: str
    ) -> dict:
        """
        Produce a swarm-wide management directive from the Nash solution.
        This directive is logged and passed to the Logger Agent.
        """
        hop_bgt  = alloc["hop_budget"]
        rl_level = alloc["rate_limit"]

        # Determine channel hopping recommendation
        if hop_bgt >= 0.80:
            channel_directive = "ACTIVATE_FULL_SWARM_HOP"
        elif hop_bgt >= 0.40:
            channel_directive = "ACTIVATE_PARTIAL_HOP"
        elif hop_bgt >= 0.10:
            channel_directive = "MONITOR_CHANNELS_ONLY"
        else:
            channel_directive = "NORMAL_CHANNEL_OPS"

        # Determine network protection level
        if rl_level >= 0.80:
            network_directive = "AGGRESSIVE_RATE_LIMIT_AND_ISOLATE"
        elif rl_level >= 0.50:
            network_directive = "MODERATE_RATE_LIMIT"
        elif rl_level >= 0.20:
            network_directive = "LIGHT_RATE_LIMIT"
        else:
            network_directive = "NORMAL_NETWORK_OPS"

        # Energy mode
        energy_directive = (
            "ENERGY_CONSERVATION_ACTIVE"
            if alloc["energy_save"] >= 0.15
            else "FULL_POWER_MODE"
        )

        return {
            "gcc_id"                  : self.gcc_id,
            "timestamp"               : datetime.now(timezone.utc).isoformat(),
            "dominant_strategy"       : dominant_strategy,
            "defender_mix"            : {s: round(float(p),4)
                                         for s,p in zip(DEFENDER_STRATEGIES,defender_mix)},
            "channel_directive"       : channel_directive,
            "network_directive"       : network_directive,
            "energy_directive"        : energy_directive,
            "relay_redundancy_factor" : alloc["relay_factor"],
            "hop_budget_fraction"     : alloc["hop_budget"],
            "rate_limit_aggressiveness": alloc["rate_limit"],
            "kpi_snapshot"            : kpis,
            "game_value"              : None,   # filled by manage()
        }

    # ── Master manage() loop ───────────────────────────────────────────────────
    def manage(
        self,
        detection_df : pd.DataFrame,
        response_df  : pd.DataFrame,
        coord_df     : pd.DataFrame,
    ) -> list:
        """
        Main management loop. Processes in `kpi_window`-row batches.
        For each batch:
          1. Measure KPIs
          2. Estimate attack distribution from detection labels
          3. Update payoff matrix
          4. Solve Nash Equilibrium
          5. Allocate resources
          6. Generate and log swarm directive
        Returns list of directive dicts.
        """
        # Avoid full-frame .copy() of three ~200k×150 DataFrames held at once
        # (hundreds of MB each -> OOM risk). We only need a datetime timestamp
        # column and positional slicing, so reference the frames directly and
        # convert timestamps without mutating the callers.
        det  = detection_df
        resp = response_df
        cord = coord_df

        def _with_ts(df):
            if "timestamp" not in df.columns:
                return df
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            if ts.equals(df["timestamp"]):
                return df
            # assign returns a shallow copy sharing the other columns' blocks,
            # far cheaper than df.copy().
            return df.assign(timestamp=ts)

        det  = _with_ts(det)
        resp = _with_ts(resp)
        cord = _with_ts(cord)

        n = len(det)
        directives = []

        for start in range(0, n, self.kpi_window):
            end       = min(start + self.kpi_window, n)
            det_batch  = det.iloc[start:end]
            resp_batch = resp.iloc[start:end] if start < len(resp) else pd.DataFrame()
            cord_batch = cord.iloc[start:end] if start < len(cord) else pd.DataFrame()

            if resp_batch.empty:
                resp_batch = det_batch  # fallback
            if cord_batch.empty:
                cord_batch = det_batch

            # ── 1. Measure KPIs ────────────────────────────────────────────────
            kpis = self.measure_kpis(det_batch, resp_batch, cord_batch)

            # ── 2. Attack distribution ─────────────────────────────────────────
            label_counts = det_batch["final_label"].value_counts()\
                           .to_dict() if "final_label" in det_batch.columns else {}
            total = max(sum(label_counts.values()), 1)
            jam_n = sum(v for k,v in label_counts.items() if k=="JAMMING")
            dos_n = sum(v for k,v in label_counts.items() if k in {"DOS","DDOS"})
            hyb_n = sum(v for k,v in label_counts.items() if k=="HYBRID_ATTACK")
            attack_n = jam_n + dos_n + hyb_n
            if attack_n > 0:
                atk_dist = {
                    "JAM_CHANNEL"   : jam_n / attack_n,
                    "FLOOD_NETWORK" : dos_n / attack_n,
                    "HYBRID_ATTACK" : hyb_n / attack_n,
                }
            else:
                atk_dist = {k: 1/3 for k in ATTACKER_STRATEGIES}

            # ── 3. Update payoff ───────────────────────────────────────────────
            payoff = self.update_payoff(kpis, atk_dist)

            # ── 4. Solve Nash Equilibrium ──────────────────────────────────────
            def_mix, atk_mix, gv = solve_nash_equilibrium(payoff)

            # ── 5. Allocate resources ──────────────────────────────────────────
            alloc = self.allocate_resources(def_mix)
            dominant = DEFENDER_STRATEGIES[int(np.argmax(def_mix))]

            # ── 6. Generate directive ──────────────────────────────────────────
            directive = self.generate_directive(def_mix, alloc, kpis, dominant)
            directive["game_value"]          = round(float(gv), 4)
            directive["batch_start_row"]     = int(start)
            directive["batch_end_row"]       = int(end)
            directive["attack_distribution"] = atk_dist
            directive["attacker_mix"]        = {s: round(float(p),4)
                                                for s,p in zip(ATTACKER_STRATEGIES,atk_mix)}

            directives.append(directive)
            self._decision_log.append(directive)
            self._window_count += 1

            # Log to Logger Agent
            self.logger.log_gcc_event(
                gcc_id                 = self.gcc_id,
                directive              = directive,
                kpis                   = kpis,
                dominant_strategy      = dominant,
                game_value             = float(gv),
                attack_distribution    = atk_dist,
            )

        return directives


def run_management_agent(
    detection_df  : pd.DataFrame,
    response_df   : pd.DataFrame,
    coord_df      : pd.DataFrame,
    logger                = None,
    save_outputs: bool    = False,
    preview: bool         = False,
    gcc_id: str           = "gcc_001",
    kpi_window: int       = 30,
) -> tuple:
    """
    Run the Management Agent over all detection/response/coordination windows.
    Returns (mgmt_agent, directives, management_df, kpi_df).
    """
    mgmt_agent = ManagementAgent(
        logger=logger or NullLogger(),
        gcc_id=gcc_id,
        kpi_window=kpi_window,
    )
    directives = mgmt_agent.manage(detection_df, response_df, coord_df)

    # Flatten directive list to DataFrame
    mgmt_rows = []
    for d in directives:
        row = {
            "timestamp"               : d["timestamp"],
            "gcc_id"                  : d["gcc_id"],
            "batch_start_row"         : d["batch_start_row"],
            "batch_end_row"           : d["batch_end_row"],
            "dominant_strategy"       : d["dominant_strategy"],
            "game_value"              : d["game_value"],
            "channel_directive"       : d["channel_directive"],
            "network_directive"       : d["network_directive"],
            "energy_directive"        : d["energy_directive"],
            "relay_redundancy_factor" : d["relay_redundancy_factor"],
            "hop_budget_fraction"     : d["hop_budget_fraction"],
            "rate_limit_aggressiveness": d["rate_limit_aggressiveness"],
            # KPI columns
            "kpi_pdr"                 : d["kpi_snapshot"]["pdr"],
            "kpi_connectivity"        : d["kpi_snapshot"]["swarm_connectivity_ratio"],
            "kpi_energy_overhead"     : d["kpi_snapshot"]["energy_overhead"],
            "kpi_detect_latency"      : d["kpi_snapshot"]["detection_latency_norm"],
            "kpi_fp_rate"             : d["kpi_snapshot"]["false_positive_rate"],
            "kpi_response_coverage"   : d["kpi_snapshot"]["response_coverage"],
            "kpi_attack_rate"         : d["kpi_snapshot"]["attack_rate"],
            # Attack distribution
            "atk_jam_fraction"        : d["attack_distribution"].get("JAM_CHANNEL",  0),
            "atk_dos_fraction"        : d["attack_distribution"].get("FLOOD_NETWORK",0),
            "atk_hyb_fraction"        : d["attack_distribution"].get("HYBRID_ATTACK",0),
            # Defender mix
            "def_normal_ops"          : d["defender_mix"].get("NORMAL_OPS",     0),
            "def_light_defense"       : d["defender_mix"].get("LIGHT_DEFENSE",  0),
            "def_full_defense"        : d["defender_mix"].get("FULL_DEFENSE",   0),
            "def_emergency_mode"      : d["defender_mix"].get("EMERGENCY_MODE", 0),
        }
        mgmt_rows.append(row)

    mgmt_df = pd.DataFrame(mgmt_rows)
    kpi_df  = pd.DataFrame(mgmt_agent._kpi_history)

    if save_outputs:
        mgmt_df.to_csv("management_output.csv", index=False)
        kpi_df.to_csv("management_kpi_history.csv", index=False)
        with open("management_output.jsonl","w",encoding="utf-8") as f:
            for d in directives:
                f.write(json.dumps(d, default=str)+"\n")

    if preview:
        print("[Management] Directive batches:", len(directives))
        print("[Management] Channel directives:")
        print(mgmt_df["channel_directive"].value_counts().to_string())
        print("[Management] Network directives:")
        print(mgmt_df["network_directive"].value_counts().to_string())
        print("[Management] Dominant strategies:")
        print(mgmt_df["dominant_strategy"].value_counts().to_string())

    return mgmt_agent, directives, mgmt_df, kpi_df


# ── Run management agent ──────────────────────────────────────────────────────

def print_kpi_dashboard(mgmt_df: pd.DataFrame, kpi_df: pd.DataFrame,
                         coord_df: pd.DataFrame) -> dict:
    """Print the full KPI dashboard. Mirrors the health summary from the original."""
    print("=" * 70)
    print("  COORDINATION & MANAGEMENT — KPI DASHBOARD")
    print("=" * 70)

    # ── Coordination summary ───────────────────────────────────────────────────
    print("\n  COORDINATION AGENT")
    print("  " + "-" * 50)
    print("  Coordination action distribution:")
    print(coord_df["coordination_action"].value_counts().to_string())
    print("\n  Node status distribution:")
    print(coord_df["coordination_status"].value_counts().to_string())

    n_coord = int(coord_df.get("coordinated_attack_detected",
                                pd.Series(False)).sum()) \
              if "coordinated_attack_detected" in coord_df.columns else 0
    print(f"\n  Coordinated attacks detected : {n_coord}")
    print(f"  GCC notifications issued     : "
          f"{int(coord_df['gcc_notified'].sum()) if 'gcc_notified' in coord_df.columns else 0}")

    # Trust registry from coordination
    if "trust_after" in coord_df.columns:
        trust_by_node = coord_df.groupby("node_id")["trust_after"].last()
        print("\n  Final trust scores (per node):")
        print(trust_by_node.round(4).to_string())

    # ── Management summary ─────────────────────────────────────────────────────
    print("\n  MANAGEMENT AGENT")
    print("  " + "-" * 50)
    if len(mgmt_df) > 0:
        print("  Dominant strategies across batches:")
        print(mgmt_df["dominant_strategy"].value_counts().to_string())
        print("\n  Channel directives issued:")
        print(mgmt_df["channel_directive"].value_counts().to_string())
        print("\n  Network directives issued:")
        print(mgmt_df["network_directive"].value_counts().to_string())
        print(f"\n  Game value range: [{mgmt_df['game_value'].min():.4f},"
              f" {mgmt_df['game_value'].max():.4f}]"
              f"  mean={mgmt_df['game_value'].mean():.4f}")

    # ── KPI summary ────────────────────────────────────────────────────────────
    if len(kpi_df) > 0:
        print("\n  KPI SUMMARY (across all batches)")
        print("  " + "-" * 50)
        kpi_show = [
            ("Packet Delivery Ratio (PDR)",        "pdr"),
            ("Swarm Connectivity Ratio",            "swarm_connectivity_ratio"),
            ("Energy Overhead",                     "energy_overhead"),
            ("Detection Latency (norm, lower=fast)","detection_latency_norm"),
            ("False Positive Rate",                 "false_positive_rate"),
            ("Response Coverage (HIGH attacks)",    "response_coverage"),
            ("Attack Rate",                         "attack_rate"),
        ]
        for label, col in kpi_show:
            if col in kpi_df.columns:
                s = kpi_df[col]
                print(f"  {label:<45s}: "
                      f"mean={s.mean():.4f}  min={s.min():.4f}  max={s.max():.4f}")

    summary = {
        "n_coordination_logs"  : len(coord_df),
        "n_management_batches" : len(mgmt_df),
        "n_coordinated_attacks": n_coord,
        "mean_pdr"             : float(kpi_df["pdr"].mean())
                                 if len(kpi_df)>0 and "pdr" in kpi_df.columns else None,
        "mean_connectivity"    : float(kpi_df["swarm_connectivity_ratio"].mean())
                                 if len(kpi_df)>0 and "swarm_connectivity_ratio" in kpi_df.columns
                                 else None,
        "mean_game_value"      : float(mgmt_df["game_value"].mean())
                                 if len(mgmt_df)>0 else None,
    }
    return summary


def coordinate_single(detection_input: dict, preview: bool = True) -> dict:
    """
    Run Coordination + Management on a single detection window dict.
    Mirrors predict_single() from the original.
    """
    row = {
        "timestamp"                  : detection_input.get("timestamp",
                                         datetime.now(timezone.utc).isoformat()),
        "node_id"                    : detection_input.get("node_id", "uav_001"),
        "node_type"                  : detection_input.get("node_type", "uav"),
        "final_label"                : detection_input.get("final_label", "NORMAL"),
        "severity"                   : detection_input.get("severity", "LOW"),
        "fusion_confidence"          : detection_input.get("fusion_confidence", 0.0),
        "jam_risk_score"             : detection_input.get("jam_risk_score", 0.0),
        "dos_risk_score"             : detection_input.get("dos_risk_score", 0.0),
        "hybrid_risk_score"          : detection_input.get("hybrid_risk_score", 0.0),
        "cross_layer_anomaly_score"  : detection_input.get("cross_layer_anomaly_score", 0.0),
        "swarm_consensus_anomaly_score": detection_input.get("swarm_consensus_anomaly_score",0.0),
        "consecutive_attack_count"   : detection_input.get("consecutive_attack_count", 0),
        "packet_delivery_ratio"      : detection_input.get("packet_delivery_ratio", 0.95),
        "snr_db"                     : detection_input.get("snr_db", 15.0),
        "channel_occupancy_pct"      : detection_input.get("channel_occupancy_pct", 20.0),
        "notify_gcc"                 : detection_input.get("notify_gcc", False),
        "detection_reason_codes"     : detection_input.get("detection_reason_codes", []),
        "label"                      : detection_input.get("label", "unknown"),
    }
    resp_row = {
        **row,
        "rf_action_taken"    : detection_input.get("rf_action_taken", "NO_RF_CHANGE"),
        "net_action_taken"   : detection_input.get("net_action_taken","NO_NETWORK_CHANGE"),
        "safety_override"    : detection_input.get("safety_override", False),
        "reactive_triggered" : detection_input.get("reactive_triggered", False),
        "notify_neighbors"   : detection_input.get("notify_neighbors", False),
        "notify_edge"        : detection_input.get("notify_edge", False),
        "lte_fallback_active": detection_input.get("lte_fallback_active", False),
    }

    df_det  = pd.DataFrame([row])
    df_resp = pd.DataFrame([resp_row])

    # Coordination
    ca = CoordinationAgent(logger=NullLogger())
    logs = ca.coordinate(df_det, df_resp)
    cl   = logs[0] if logs else {}

    # Management (single-batch)
    ma = ManagementAgent(logger=NullLogger(), kpi_window=1)
    directives = ma.manage(df_det, df_resp,
                            pd.DataFrame([cl]) if cl else df_det)
    d = directives[0] if directives else {}

    output = {
        "final_label"              : row["final_label"],
        "severity"                 : row["severity"],
        "coordination_status"      : cl.get("coordination_status", "UNKNOWN"),
        "trust_before"             : round(cl.get("trust_before", 1.0), 4),
        "trust_after"              : round(cl.get("trust_after",  1.0), 4),
        "evidence_score"           : round(cl.get("evidence_score", 0.0), 4),
        "bayes_p_coordinated"      : round(cl.get("bayes_p_coordinated", 0.0), 4),
        "coordination_action"      : cl.get("coordination_action", "NO_GLOBAL_CHANGE"),
        "gcc_notified_coord"       : cl.get("gcc_notified", False),
        "dominant_strategy"        : d.get("dominant_strategy", "NORMAL_OPS"),
        "game_value"               : round(d.get("game_value", 0.0), 4),
        "channel_directive"        : d.get("channel_directive", "NORMAL_CHANNEL_OPS"),
        "network_directive"        : d.get("network_directive", "NORMAL_NETWORK_OPS"),
        "energy_directive"         : d.get("energy_directive", "FULL_POWER_MODE"),
        "hop_budget_fraction"      : round(d.get("hop_budget_fraction", 0.0), 4),
        "rate_limit_aggressiveness": round(d.get("rate_limit_aggressiveness", 0.0), 4),
    }

    if preview:
        print("=" * 62)
        print(f"  LABEL      : {output['final_label']}  [{output['severity']}]")
        print(f"  TRUST      : {output['trust_before']} -> {output['trust_after']}")
        print(f"  EVIDENCE   : {output['evidence_score']}  "
              f"P(coord)={output['bayes_p_coordinated']}")
        print(f"  COORD ACT  : {output['coordination_action']}")
        print(f"  STRATEGY   : {output['dominant_strategy']}  "
              f"(game_value={output['game_value']})")
        print(f"  CHANNEL    : {output['channel_directive']}")
        print(f"  NETWORK    : {output['network_directive']}")
        print(f"  ENERGY     : {output['energy_directive']}")
        print(f"  HOP BUDGET : {output['hop_budget_fraction']}  "
              f"RATE LIMIT: {output['rate_limit_aggressiveness']}")
        print("=" * 62)
    return output


