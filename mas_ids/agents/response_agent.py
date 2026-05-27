"""
mas_ids.agents.response_agent
=============================
Agent 4 — Response.

Three-tier response architecture:
  Tier 1 — Safety Layer (deterministic hard constraints)
  Tier 2 — Fast Reactive Controller (threshold-driven immediate mitigation)
  Tier 3 — DQN MARL policy (PyTorch, CPU) over a 12-action unified space

Public API:
  JAM_ACTION_SPACE, DOS_ACTION_SPACE, UNIFIED_ACTION_SPACE, STATE_DIM, ACTION_DIM,
  ReplayBuffer, ResponseQNetwork, DeepQResponsePolicy, ResponseAgent,
  prepare_response_dataframe, train_response_policy, load_response_agent,
  run_response_agent_inference, evaluate_response, respond_single
"""
import os
import json
import time
import random
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim

from ..config import (
    GLOBAL_SEED, RESPONSE_MODEL_DIR, ROUTE_POLICY_WEIGHTS, ROUTE_POLICY_CONFIG,
    TORCH_DEVICE as _CFG_TORCH_DEVICE,
)
from ..utils import NullLogger

# Torch device — always CPU for the DQN (avoids P100 CUDA kernel mismatch)
TORCH_DEVICE = _CFG_TORCH_DEVICE if _CFG_TORCH_DEVICE is not None else torch.device('cpu')


# ── Jamming-specific action space ────────────────────────────────────────────
JAM_ACTION_SPACE = [
    "CONTINUE_NORMAL_OPERATION",       # No RF anomaly — maintain current channel
    "INCREASE_MONITORING",             # Soft signal drop — heighten observation
    "SWITCH_TO_BACKUP_CHANNEL",        # Emergency channel switch
    "ACTIVATE_FREQUENCY_HOPPING",      # Spread-spectrum anti-jamming
    "REDUCE_TX_POWER_AND_REPOSITION",  # Reduce interference footprint, move node
    "SWITCH_TO_LTE_FALLBACK",          # Fall back to LTE uplink
]

# ── DoS/DDoS-specific action space ───────────────────────────────────────────
DOS_ACTION_SPACE = [
    "ALLOW_NORMAL_TRAFFIC",            # No threat — pass all traffic
    "FLAG_AND_RATE_LIMIT",             # Soft throttle suspicious sources
    "BLOCK_SUSPICIOUS_SOURCES",        # Drop packets from flagged IPs
    "ACTIVATE_SYN_COOKIE_DEFENSE",     # TCP SYN flood mitigation
    "SHED_NON_CRITICAL_LOAD",          # Drop low-priority traffic under overload
    "REROUTE_THROUGH_BACKUP_RELAY",    # Topology change — avoid saturated path
]

# ── Unified DQN action space (all 12 actions) ─────────────────────────────────
UNIFIED_ACTION_SPACE = JAM_ACTION_SPACE + DOS_ACTION_SPACE

STATE_DIM  = 15   # Extended from original 10 to cover DoS/DDoS signals
ACTION_DIM = len(UNIFIED_ACTION_SPACE)

# ── State feature mapping ─────────────────────────────────────────────────────
# Index : Feature                   : Source
# 0     : jam_risk_score            : Detection Agent Tier C
# 1     : dos_risk_score            : Detection Agent Tier C
# 2     : hybrid_risk_score         : Detection Agent Tier C
# 3     : fusion_confidence         : Detection Agent Tier C
# 4     : edge_alarm                : Detection Agent Tier A
# 5     : cross_layer_anomaly_score : Feature Engineering
# 6     : snr_db_norm               : Physical layer (norm to [0,1])
# 7     : channel_occupancy_norm    : MAC layer (pct/100)
# 8     : packet_delivery_ratio     : MAC layer
# 9     : half_open_ratio           : Transport layer (norm)
# 10    : src_ip_entropy_norm       : Network layer (norm /9)
# 11    : error_response_rate       : App layer
# 12    : consecutive_attack_norm   : Persistence (norm /20)
# 13    : signal_channel_trust      : Running trust for RF channel
# 14    : network_path_trust        : Running trust for network path


class ReplayBuffer:
    """Experience replay buffer. Mirrors original exactly."""
    def __init__(self, capacity: int = 8000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self): return len(self.buffer)


class ResponseQNetwork(nn.Module):
    """DQN Q-Network for the Response Agent. Mirrors RouteQNetwork from original.
    Three hidden layers with BatchNorm for training stability.
    """
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepQResponsePolicy:
    """
    DQN policy for the DoS/DDoS + Jamming Response Agent.
    Mirrors DeepQRoutePolicy from the original exactly.
    Trains on CPU to avoid P100 CUDA kernel mismatch.
    """

    def __init__(
        self,
        action_space: list  = None,
        gamma: float        = 0.95,
        epsilon: float      = 1.0,
        epsilon_min: float  = 0.05,
        epsilon_decay: float= 0.995,
        lr: float           = 0.001,
    ):
        self.action_space  = action_space or UNIFIED_ACTION_SPACE
        self.action_dim    = len(self.action_space)
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device        = TORCH_DEVICE

        self.online_net = ResponseQNetwork(STATE_DIM, self.action_dim).to(self.device)
        self.target_net = ResponseQNetwork(STATE_DIM, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer    = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay       = ReplayBuffer(capacity=8000)
        self.loss_history = []

    def choose_action(self, state: np.ndarray, training: bool = False):
        if training and random.random() < self.epsilon:
            idx = random.randint(0, self.action_dim - 1)
            return self.action_space[idx], idx
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.online_net(state_t)
        idx = int(q_vals.argmax().item())
        return self.action_space[idx], idx

    def remember(self, state, action_idx, reward, next_state, done):
        self.replay.add(state, action_idx, reward, next_state, done)

    def train_step(self, batch_size: int = 64) -> float | None:
        if len(self.replay) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            next_q   = self.target_net(next_states_t).max(1)[0]
            target_q = (rewards_t + (1.0 - dones_t) * self.gamma * next_q).clamp(-10.0, 10.0)

        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss      = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        val = float(loss.item())
        self.loss_history.append(val)
        return val

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_policy(self):
        os.makedirs(RESPONSE_MODEL_DIR, exist_ok=True)
        torch.save(self.online_net.state_dict(), ROUTE_POLICY_WEIGHTS)
        config = {
            "action_space"  : self.action_space,
            "state_dim"     : STATE_DIM,
            "action_dim"    : self.action_dim,
            "gamma"         : self.gamma,
            "epsilon_final" : self.epsilon,
        }
        with open(ROUTE_POLICY_CONFIG, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Policy saved: {ROUTE_POLICY_WEIGHTS}")

    @classmethod
    def load_policy(cls) -> "DeepQResponsePolicy":
        with open(ROUTE_POLICY_CONFIG) as f:
            cfg = json.load(f)
        policy = cls(
            action_space=cfg["action_space"],
            epsilon=cfg.get("epsilon_final", 0.05),
        )
        policy.online_net.load_state_dict(
            torch.load(ROUTE_POLICY_WEIGHTS, map_location=TORCH_DEVICE)
        )
        policy.online_net.eval()
        policy.target_net.load_state_dict(policy.online_net.state_dict())
        return policy


class ResponseAgent:
    """
    DoS/DDoS + Jamming Response Agent.
    Mirrors ResponseAgent from the original (Cell 14) structurally,
    adapting all methods for the jamming + DoS/DDoS attack types.
    """

    def __init__(self, logger=None, dqn_policy=None):
        self.logger        = logger or NullLogger()
        self.dqn_policy    = dqn_policy or DeepQResponsePolicy()
        # Per-node trust scores — mirrors gps_trust_scores / node_route_trust
        self.signal_channel_trust : dict = {}   # RF signal channel trust [0,1]
        self.network_path_trust   : dict = {}   # Network path trust [0,1]
        # Last known safe state (for fallback reference)
        self.last_safe_state      : dict = {}
        # Consecutive alert counter per node (for safety layer escalation)
        self._consec_alert        : dict = defaultdict(int)

    # ── Trust management (mirrors original exactly) ────────────────────────────
    def get_signal_trust(self, node_id: str) -> float:
        return self.signal_channel_trust.get(node_id, 1.0)

    def set_signal_trust(self, node_id: str, value: float):
        self.signal_channel_trust[node_id] = float(np.clip(value, 0.0, 1.0))

    def reduce_signal_trust(self, node_id: str, amount: float = 0.25):
        self.set_signal_trust(node_id, self.get_signal_trust(node_id) - amount)

    def get_network_trust(self, node_id: str) -> float:
        return self.network_path_trust.get(node_id, 1.0)

    def set_network_trust(self, node_id: str, value: float):
        self.network_path_trust[node_id] = float(np.clip(value, 0.0, 1.0))

    def reduce_network_trust(self, node_id: str, amount: float = 0.25):
        self.set_network_trust(node_id, self.get_network_trust(node_id) - amount)

    # ── State vector construction (mirrors build_state from original) ──────────
    def build_state(self, row: dict | pd.Series) -> np.ndarray:
        """
        Build normalised [0,1] 15-feature state vector for the DQN.
        Mirrors build_state() from the original exactly — all values clipped.
        """
        node_id = str(row.get("node_id", "unknown"))
        raw = np.array([
            float(row.get("jam_risk_score",                0.0)),   # 0
            float(row.get("dos_risk_score",                0.0)),   # 1
            float(row.get("hybrid_risk_score",             0.0)),   # 2
            float(row.get("fusion_confidence",             0.0)),   # 3
            float(row.get("edge_alarm",                    0.0)),   # 4
            float(row.get("cross_layer_anomaly_score",     0.0)),   # 5
            # Normalised physical layer signals
            min(max(float(row.get("snr_db", 10.0)) + 30.0, 0.0), 60.0) / 60.0,  # 6 SNR
            float(row.get("channel_occupancy_pct", 20.0)) / 100.0,              # 7 CH occ
            float(row.get("packet_delivery_ratio",  1.0)),                       # 8 PDR
            # Normalised network layer signals
            min(float(row.get("half_open_connections", 0.0)), 20000.0) / 20000.0, # 9 half-open
            float(row.get("src_ip_entropy",  0.0)) / 9.0,                        # 10 entropy
            float(row.get("error_response_rate", 0.0)),                           # 11 err rate
            # Persistence and trust
            min(float(row.get("consecutive_attack_count", 0.0)), 20.0) / 20.0,   # 12 consec
            float(self.get_signal_trust(node_id)),                               # 13 sig trust
            float(self.get_network_trust(node_id)),                              # 14 net trust
        ], dtype=np.float32)
        return np.clip(raw, 0.0, 1.0)

    # ── Reward function (mirrors original exactly, adapted for new labels) ─────
    def reward_function(self, row: dict | pd.Series, action_name: str) -> float:
        """
        Reward function for DQN training.
        Mirrors reward_function() from the original — same structure,
        adapted for JAMMING / DOS / DDOS / HYBRID_ATTACK labels.
        All reward components clipped to avoid overwhelming the signal.
        """
        label    = str(row.get("final_label", "NORMAL"))
        severity = str(row.get("severity",    "LOW"))
        fusion   = float(row.get("fusion_confidence", 0.0))

        # Desired action set per label — mirrors original desired dict
        desired = {
            "NORMAL"        : {"CONTINUE_NORMAL_OPERATION", "ALLOW_NORMAL_TRAFFIC"},
            "SUSPICIOUS"    : {"INCREASE_MONITORING", "FLAG_AND_RATE_LIMIT"},
            "JAMMING"       : {"SWITCH_TO_BACKUP_CHANNEL", "ACTIVATE_FREQUENCY_HOPPING",
                               "SWITCH_TO_LTE_FALLBACK"},
            "DOS"           : {"BLOCK_SUSPICIOUS_SOURCES", "ACTIVATE_SYN_COOKIE_DEFENSE",
                               "FLAG_AND_RATE_LIMIT"},
            "DDOS"          : {"SHED_NON_CRITICAL_LOAD", "BLOCK_SUSPICIOUS_SOURCES",
                               "REROUTE_THROUGH_BACKUP_RELAY"},
            "HYBRID_ATTACK" : {"SWITCH_TO_BACKUP_CHANNEL", "ACTIVATE_FREQUENCY_HOPPING",
                               "SHED_NON_CRITICAL_LOAD", "REROUTE_THROUGH_BACKUP_RELAY"},
        }

        reward = 1.0 if action_name in desired.get(label, set()) else -1.5

        # Penalty: aggressive action on NORMAL traffic (false positive cost)
        if label == "NORMAL" and action_name in {
            "SWITCH_TO_BACKUP_CHANNEL", "BLOCK_SUSPICIOUS_SOURCES",
            "ACTIVATE_FREQUENCY_HOPPING", "SHED_NON_CRITICAL_LOAD"
        }:
            reward -= 1.5

        # Bonus: correct HIGH-severity response
        if label in {"JAMMING","DOS","DDOS","HYBRID_ATTACK"} and severity == "HIGH":
            if action_name in desired.get(label, set()):
                reward += 2.0
            elif action_name in {"CONTINUE_NORMAL_OPERATION","ALLOW_NORMAL_TRAFFIC"}:
                reward -= 2.0

        # Bonus: correct hybrid dual-action
        if label == "HYBRID_ATTACK" and action_name in desired.get("HYBRID_ATTACK", set()):
            reward += 1.0

        # Fusion confidence bonus
        reward += fusion

        # Penalty: signal degradation and traffic congestion
        # Clipped to avoid overwhelming signal — mirrors Bug 2 fix from original
        reward -= min(float(row.get("channel_occupancy_pct", 20.0)) / 100.0, 1.0)
        reward -= min(float(row.get("error_response_rate",    0.0)),          1.0)

        return float(np.clip(reward, -5.0, 5.0))

    # ── Tier 1: Safety Layer (deterministic, always applied first) ────────────
    def safety_layer(self, row: dict | pd.Series) -> dict:
        """
        Tier 1: Hard constraints enforced unconditionally.
        Prevents resource exhaustion and swarm fragmentation.
        Returns a safety_action dict; if no constraint is triggered,
        returns an empty dict (no override needed).
        """
        node_id  = str(row.get("node_id", "unknown"))
        label    = str(row.get("final_label", "NORMAL"))
        severity = str(row.get("severity",    "LOW"))
        consec   = int(row.get("consecutive_attack_count", 0))

        # Hard constraint 1: protect mission-critical traffic
        # Always preserve one relay path regardless of attack state
        if label in {"DOS","DDOS","HYBRID_ATTACK"} and severity == "HIGH":
            return {
                "safety_override"            : True,
                "safety_constraint"          : "CRITICAL_TRAFFIC_PROTECTION",
                "forced_action"              : "SHED_NON_CRITICAL_LOAD",
                "min_bandwidth_reserved_pct" : 20.0,
                "priority_traffic_flag"      : True,
            }

        # Hard constraint 2: prevent complete swarm isolation
        # If all neighbours are lost, force LTE fallback regardless of power cost
        if label == "JAMMING" and severity == "HIGH" and consec >= 10:
            return {
                "safety_override"   : True,
                "safety_constraint" : "ANTI_FRAGMENTATION",
                "forced_action"     : "SWITCH_TO_LTE_FALLBACK",
                "lte_activated"     : True,
            }

        # Hard constraint 3: rate limiting under any confirmed DoS
        if label in {"DOS","DDOS"} and severity in {"MEDIUM","HIGH"}:
            return {
                "safety_override"    : True,
                "safety_constraint"  : "RATE_LIMIT_ENFORCEMENT",
                "forced_action"      : "FLAG_AND_RATE_LIMIT",
                "rate_limit_pps"     : 1000,
            }

        return {"safety_override": False}

    # ── Tier 2: Fast Reactive Controller ──────────────────────────────────────
    def reactive_response(self, row: dict | pd.Series) -> dict:
        """
        Tier 2: Threshold-driven immediate mitigation.
        Executes within one window (~1s) of attack detection.
        Returns a reactive_action dict with immediate mitigation steps.
        """
        node_id  = str(row.get("node_id", "unknown"))
        label    = str(row.get("final_label", "NORMAL"))
        severity = str(row.get("severity",    "LOW"))

        snr      = float(row.get("snr_db",              10.0))
        ch_occ   = float(row.get("channel_occupancy_pct",20.0))
        syn_rate = float(row.get("syn_packet_rate",      0.0))
        h_open   = float(row.get("half_open_connections",0.0))
        pps      = float(row.get("packets_per_second",   50.0))
        edge_alm = int(row.get("edge_alarm", 0))

        actions_taken = []

        # Reactive rule 1: emergency channel switch on sudden SNR drop
        if (label == "JAMMING" or edge_alm == 1) and snr < 0.0:
            self.reduce_signal_trust(node_id, amount=0.30)
            actions_taken.append("EMERGENCY_CHANNEL_SWITCH")

        # Reactive rule 2: SYN cookie on SYN flood onset
        if syn_rate > 5000 or h_open > 2000:
            actions_taken.append("ACTIVATE_SYN_COOKIE_IMMEDIATE")

        # Reactive rule 3: load shedding on extreme PPS
        if pps > 50000:
            self.reduce_network_trust(node_id, amount=0.20)
            actions_taken.append("EMERGENCY_LOAD_SHED")

        # Reactive rule 4: channel occupancy — trigger frequency hop
        if ch_occ > 80.0 and label in {"JAMMING","HYBRID_ATTACK"}:
            actions_taken.append("TRIGGER_FREQUENCY_HOP")

        return {
            "reactive_actions_taken" : actions_taken,
            "n_reactive_actions"     : len(actions_taken),
            "reactive_triggered"     : len(actions_taken) > 0,
            "signal_trust_after_reactive" : self.get_signal_trust(node_id),
            "network_trust_after_reactive": self.get_network_trust(node_id),
        }

    # ── Tier 3a: RF / Jamming response (DQN-guided) ───────────────────────────
    def rf_response(self, row: dict | pd.Series) -> dict:
        """
        Tier 3: DQN-guided RF channel management response.
        Mirrors navigation_response() from the original.
        """
        node_id             = str(row.get("node_id", "unknown"))
        label               = str(row.get("final_label", "NORMAL"))
        severity            = str(row.get("severity",    "LOW"))
        sig_trust_before    = self.get_signal_trust(node_id)

        action = {
            "response_type"         : "RF_CHANNEL",
            "node_id"               : node_id,
            "signal_trust_before"   : sig_trust_before,
            "signal_trust_after"    : sig_trust_before,
            "channel_mode"          : "NORMAL_OPERATION",
            "action_taken"          : "NO_RF_CHANGE",
            "fallback_channel"      : None,
            "lte_fallback_active"   : False,
            "notify_neighbors"      : False,
            "notify_edge"           : False,
            "notify_gcc"            : False,
        }

        if label in {"JAMMING", "HYBRID_ATTACK"}:
            reduction = 0.40 if severity == "HIGH" else 0.25
            self.reduce_signal_trust(node_id, amount=reduction)
            if severity == "HIGH":
                action.update({
                    "signal_trust_after" : self.get_signal_trust(node_id),
                    "channel_mode"       : "FREQUENCY_HOPPING",
                    "action_taken"       : "ACTIVATE_ANTI_JAMMING_HOP",
                    "fallback_channel"   : "BACKUP_5GHz",
                    "lte_fallback_active": severity == "HIGH" and
                                           int(row.get("consecutive_attack_count", 0)) >= 10,
                    "notify_neighbors"   : True,
                    "notify_edge"        : True,
                    "notify_gcc"         : True,
                })
            else:
                action.update({
                    "signal_trust_after" : self.get_signal_trust(node_id),
                    "channel_mode"       : "CHANNEL_SWITCH_MONITOR",
                    "action_taken"       : "SWITCH_TO_BACKUP_CHANNEL",
                    "fallback_channel"   : "BACKUP_2.4GHz",
                    "notify_neighbors"   : True,
                    "notify_edge"        : True,
                    "notify_gcc"         : False,
                })
        elif label == "SUSPICIOUS":
            self.reduce_signal_trust(node_id, amount=0.05)
            action.update({
                "signal_trust_after" : self.get_signal_trust(node_id),
                "channel_mode"       : "HEIGHTENED_MONITORING",
                "action_taken"       : "INCREASE_RF_MONITORING",
                "notify_edge"        : True,
            })
        else:
            # Normal — trust recovery (mirrors Issue 3 fix from original)
            current = self.get_signal_trust(node_id)
            if current < 1.0:
                self.set_signal_trust(node_id, min(1.0, current + 0.01))
                action["signal_trust_after"] = self.get_signal_trust(node_id)
            # Update last safe state
            self.last_safe_state[node_id] = {
                "snr_db"          : float(row.get("snr_db", 0)),
                "rssi_dbm"        : float(row.get("rssi_dbm", -70)),
                "channel"         : "PRIMARY",
                "timestamp"       : str(row.get("timestamp", "")),
            }
        return action

    # ── Tier 3b: Network / DoS response (DQN-guided) ──────────────────────────
    def network_response(self, row: dict | pd.Series) -> dict:
        """
        Tier 3: DQN-guided network path management response.
        Mirrors network_response() from the original exactly.
        """
        node_id              = str(row.get("node_id", "unknown"))
        label                = str(row.get("final_label", "NORMAL"))
        severity             = str(row.get("severity",    "LOW"))
        net_trust_before     = self.get_network_trust(node_id)
        state                = self.build_state(row)
        recommended, act_idx = self.dqn_policy.choose_action(state, training=False)

        action = {
            "response_type"              : "NETWORK_PATH",
            "node_id"                    : node_id,
            "network_trust_before"       : net_trust_before,
            "network_trust_after"        : net_trust_before,
            "dqn_recommendation"         : recommended,
            "selected_network_action"    : recommended,
            "action_taken"               : "NO_NETWORK_CHANGE",
            "alert_delivery_mode"        : "PRIMARY_ROUTE",
            "dqn_action_idx"             : int(act_idx),
            "dqn_state_vector"           : state.tolist(),
            "notify_neighbors"           : False,
            "notify_edge"                : False,
            "notify_gcc"                 : False,
        }

        if label == "DDOS":
            self.reduce_network_trust(node_id, 0.40 if severity=="HIGH" else 0.25)
            action.update({
                "network_trust_after"    : self.get_network_trust(node_id),
                "selected_network_action": "SHED_NON_CRITICAL_LOAD",
                "action_taken"           : "DDOS_MITIGATION_ACTIVE",
                "alert_delivery_mode"    : "BACKUP_MULTI_HOP",
                "notify_neighbors"       : True,
                "notify_edge"            : True,
                "notify_gcc"             : True,
            })
        elif label == "DOS":
            self.reduce_network_trust(node_id, 0.35 if severity=="HIGH" else 0.20)
            action.update({
                "network_trust_after"    : self.get_network_trust(node_id),
                "selected_network_action": "BLOCK_SUSPICIOUS_SOURCES",
                "action_taken"           : "DOS_BLOCK_AND_RATE_LIMIT",
                "alert_delivery_mode"    : "PRIMARY_PLUS_BROADCAST",
                "notify_neighbors"       : True,
                "notify_edge"            : True,
                "notify_gcc"             : severity == "HIGH",
            })
        elif label == "HYBRID_ATTACK":
            self.reduce_network_trust(node_id, 0.45 if severity=="HIGH" else 0.30)
            action.update({
                "network_trust_after"    : self.get_network_trust(node_id),
                "selected_network_action": "REROUTE_THROUGH_BACKUP_RELAY",
                "action_taken"           : "HYBRID_CONTAIN_AND_REROUTE",
                "alert_delivery_mode"    : "BACKUP_MULTI_HOP",
                "notify_neighbors"       : True,
                "notify_edge"            : True,
                "notify_gcc"             : True,
            })
        elif label == "SUSPICIOUS":
            self.reduce_network_trust(node_id, 0.10)
            action.update({
                "network_trust_after"    : self.get_network_trust(node_id),
                "selected_network_action": recommended
                    if recommended != "CONTINUE_NORMAL_OPERATION"
                    else "FLAG_AND_RATE_LIMIT",
                "action_taken"           : "MONITOR_NETWORK_PATH",
                "notify_edge"            : True,
            })
        else:
            # Normal — network trust recovery (mirrors Issue 3 fix)
            current = self.get_network_trust(node_id)
            if current < 1.0:
                self.set_network_trust(node_id, min(1.0, current + 0.005))
                action["network_trust_after"] = self.get_network_trust(node_id)
        return action

    # ── Master respond() — orchestrates all three tiers ───────────────────────
    def respond(self, row: dict | pd.Series) -> dict:
        """
        Orchestrate all three response tiers for a single detection window.
        Mirrors respond() from the original.
        Returns a complete response record.
        """
        node_id  = str(row.get("node_id", "unknown"))

        # Snapshot trust before any modifications
        trust_before_signal  = self.get_signal_trust(node_id)
        trust_before_network = self.get_network_trust(node_id)

        # Tier 1 — Safety Layer (always first)
        safety  = self.safety_layer(row)

        # Tier 2 — Fast Reactive Controller
        reactive = self.reactive_response(row)

        # Tier 3 — DQN-guided responses
        rf_act  = self.rf_response(row)
        net_act = self.network_response(row)

        notify_neighbors = any([
            rf_act.get("notify_neighbors"),
            net_act.get("notify_neighbors")
        ])
        notify_edge = any([
            rf_act.get("notify_edge"),
            net_act.get("notify_edge")
        ])
        notify_gcc = any([
            rf_act.get("notify_gcc"),
            net_act.get("notify_gcc")
        ])

        # Safety override can force final network action
        final_network_action = net_act["selected_network_action"]
        if safety.get("safety_override"):
            final_network_action = safety["forced_action"]
            notify_gcc = True

        return {
            "timestamp"              : row.get("timestamp", ""),
            "node_id"                : node_id,
            "node_type"              : str(row.get("node_type", "uav")),
            "final_label"            : str(row.get("final_label", "NORMAL")),
            "severity"               : str(row.get("severity", "LOW")),
            "fusion_confidence"      : float(row.get("fusion_confidence", 0.0)),
            "trust_before_signal"    : trust_before_signal,
            "trust_before_network"   : trust_before_network,
            "safety_response"        : safety,
            "reactive_response"      : reactive,
            "rf_response"            : rf_act,
            "network_response"       : net_act,
            "final_network_action"   : final_network_action,
            "notify_neighbors"       : notify_neighbors,
            "notify_edge"            : notify_edge,
            "notify_gcc"             : notify_gcc,
            "response_status"        : "EXECUTED",
            "response_success"       : True,
        }


def prepare_response_dataframe(detection_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalise detection_df for the Response Agent.
    Mirrors prepare_response_dataframe() from the original exactly.
    """
    required = ["timestamp", "node_id", "final_label", "severity"]
    missing  = [c for c in required if c not in detection_df.columns]
    if missing:
        raise ValueError(f"detection_df missing columns: {missing}")

    df = detection_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Ensure all state-relevant columns exist with safe defaults
    defaults = {
        "jam_risk_score"           : 0.0,
        "dos_risk_score"           : 0.0,
        "hybrid_risk_score"        : 0.0,
        "fusion_confidence"        : 0.0,
        "edge_alarm"               : 0,
        "cross_layer_anomaly_score": 0.0,
        "snr_db"                   : 10.0,
        "channel_occupancy_pct"    : 20.0,
        "packet_delivery_ratio"    : 1.0,
        "half_open_connections"    : 0.0,
        "src_ip_entropy"           : 0.0,
        "error_response_rate"      : 0.0,
        "consecutive_attack_count" : 0,
        "packets_per_second"       : 50.0,
        "syn_packet_rate"          : 0.0,
        "rssi_dbm"                 : -70.0,
        "detection_reason_codes"   : None,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        if col != "detection_reason_codes":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)
    return df


def train_response_policy(
    detection_df: pd.DataFrame,
    episodes: int       = 150,
    batch_size: int     = 64,
    sync_every: int     = 10,
    progress_every: int = 500,
    save_model: bool    = True,
    preview: bool       = True,
) -> tuple:
    """
    Train the DQN response policy.
    Mirrors train_response_route_policy() from the original exactly.
    Returns (trained_policy, training_summary).
    """
    # FIX: reset seeds inside training for reproducibility across notebook reruns
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    df    = prepare_response_dataframe(detection_df)
    agent = ResponseAgent(logger=NullLogger())

    if len(df) < 2:
        raise ValueError("detection_df too small for response training.")

    t0 = time.time()

    for episode in range(episodes):
        total_reward = 0.0
        steps        = 0
        last_loss    = None
        ep_start     = time.time()

        for _, group in df.groupby("node_id", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) < 2:
                continue
            for i in range(len(group) - 1):
                row      = group.iloc[i]
                nxt      = group.iloc[i + 1]
                state    = agent.build_state(row)
                action_name, action_idx = agent.dqn_policy.choose_action(
                    state, training=True
                )
                reward     = agent.reward_function(row, action_name)
                next_state = agent.build_state(nxt)
                done       = 1.0 if i == len(group) - 2 else 0.0

                agent.dqn_policy.remember(state, action_idx, reward, next_state, done)
                loss = agent.dqn_policy.train_step(batch_size=batch_size)

                total_reward += reward
                steps        += 1
                if loss is not None:
                    last_loss = loss

                if preview and steps % progress_every == 0:
                    print(f"  Ep {episode+1}/{episodes} | step {steps} | "
                          f"eps={agent.dqn_policy.epsilon:.4f} | loss={last_loss}")

        if (episode + 1) % sync_every == 0:
            agent.dqn_policy.sync_target()

        agent.dqn_policy.decay_epsilon()

        if preview:
            mean_loss = (
                float(np.mean(agent.dqn_policy.loss_history[-100:]))
                if agent.dqn_policy.loss_history else None
            )
            ml_str = f"{mean_loss:.4f}" if mean_loss is not None else "warming_up"
            print(f"Ep {episode+1}/{episodes} done | "
                  f"reward={total_reward:.3f} | "
                  f"eps={agent.dqn_policy.epsilon:.4f} | "
                  f"mean_loss={ml_str} | "
                  f"time={time.time()-ep_start:.1f}s")

    agent.dqn_policy.sync_target()

    summary = {
        "epsilon_final"       : agent.dqn_policy.epsilon,
        "training_steps"      : len(agent.dqn_policy.loss_history),
        "mean_training_loss"  : (
            float(np.mean(agent.dqn_policy.loss_history))
            if agent.dqn_policy.loss_history else None
        ),
        "available_actions"   : agent.dqn_policy.action_space,
        "training_time_seconds": round(time.time() - t0, 2),
        "device"              : str(TORCH_DEVICE),
        "episodes"            : episodes,
        "batch_size"          : batch_size,
    }

    if save_model:
        agent.dqn_policy.save_policy()
        with open(os.path.join(RESPONSE_MODEL_DIR,
                               "response_training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if preview:
        print(f"\n[DQN] Training complete")
        print(f"  epsilon_final : {summary['epsilon_final']:.4f}")
        print(f"  mean_loss     : {summary['mean_training_loss']}")
        print(f"  training_time : {summary['training_time_seconds']}s")
        print(f"  device        : {summary['device']}")

    return agent.dqn_policy, summary


# ── Balance training set: limit to 2000 normal + 2000 attack rows ─────────────


def load_response_agent(logger=None) -> ResponseAgent:
    """
    Load saved DQN policy and return a ready ResponseAgent.
    Mirrors load_response_agent() from the original exactly.
    """
    policy = DeepQResponsePolicy.load_policy()
    return ResponseAgent(
        logger     = logger or NullLogger(),
        dqn_policy = policy,
    )


# Use trained policy if available; else load from disk


def run_response_agent_inference(
    detection_df: pd.DataFrame,
    logger       = None,
    save_outputs : bool = False,
    preview      : bool = False,
) -> tuple:
    """
    Full response inference over detection_df.
    Mirrors run_response_agent_inference() from the original exactly.
    Returns (agent, response_records, response_df, inference_summary).
    """
    df    = prepare_response_dataframe(detection_df)
    agent = load_response_agent(logger=logger or NullLogger())

    response_records = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        result = agent.respond(row)
        response_records.append(result)

        # Log AFTER respond() acts — captures actual decisions made
        if logger is not None:
            rf  = result["rf_response"]
            net = result["network_response"]
            logger.log_uav_ugv_event(
                node_id                = result["node_id"],
                node_type              = result["node_type"],
                final_label            = result["final_label"],
                severity               = result["severity"],
                jam_risk_score         = float(row.get("jam_risk_score",    0)),
                dos_risk_score         = float(row.get("dos_risk_score",    0)),
                fusion_confidence      = float(row.get("fusion_confidence", 0)),
                detection_reason_codes = row.get("detection_reason_codes",  []),
                rf_action_taken        = rf["action_taken"],
                net_action_taken       = net["action_taken"],
                safety_override        = result["safety_response"].get("safety_override", False),
                reactive_triggered     = result["reactive_response"].get("reactive_triggered", False),
                response_status        = result["response_status"],
                sensor_summary = {
                    "signal_trust_before"   : rf["signal_trust_before"],
                    "signal_trust_after"    : rf["signal_trust_after"],
                    "channel_mode"          : rf["channel_mode"],
                    "lte_fallback_active"   : rf["lte_fallback_active"],
                },
                routing_summary = {
                    "network_trust_before"  : net["network_trust_before"],
                    "network_trust_after"   : net["network_trust_after"],
                    "selected_action"       : result["final_network_action"],
                    "alert_delivery_mode"   : net["alert_delivery_mode"],
                },
                status_summary = {
                    "notify_neighbors" : result["notify_neighbors"],
                    "notify_edge"      : result["notify_edge"],
                    "notify_gcc"       : result["notify_gcc"],
                },
                trust_score_before = result["trust_before_signal"],
                trust_score_after  = rf["signal_trust_after"],
                gcc_notified       = result["notify_gcc"],
                alert_scope        = ("gcc" if result["notify_gcc"]
                                     else "edge" if result["notify_edge"]
                                     else "local"),
            )

        if preview and (idx % 500 == 0 or idx == len(df)):
            print(f"  Processed {idx}/{len(df)} rows")

    # ── Build flat response_df ─────────────────────────────────────────────────
    response_df = pd.DataFrame([
        {
            "timestamp"                  : r["timestamp"],
            "node_id"                    : r["node_id"],
            "node_type"                  : r["node_type"],
            "final_label"                : r["final_label"],
            "severity"                   : r["severity"],
            "fusion_confidence"          : r["fusion_confidence"],
            "trust_before_signal"        : r["trust_before_signal"],
            "trust_before_network"       : r["trust_before_network"],
            # Safety tier
            "safety_override"            : r["safety_response"].get("safety_override", False),
            "safety_constraint"          : r["safety_response"].get("safety_constraint", "NONE"),
            "safety_forced_action"       : r["safety_response"].get("forced_action", "NONE"),
            # Reactive tier
            "reactive_triggered"         : r["reactive_response"].get("reactive_triggered", False),
            "reactive_actions"           : str(r["reactive_response"].get("reactive_actions_taken", [])),
            # RF tier
            "rf_action_taken"            : r["rf_response"]["action_taken"],
            "rf_channel_mode"            : r["rf_response"]["channel_mode"],
            "signal_trust_before"        : r["rf_response"]["signal_trust_before"],
            "signal_trust_after"         : r["rf_response"]["signal_trust_after"],
            "lte_fallback_active"        : r["rf_response"]["lte_fallback_active"],
            # Network tier
            "net_action_taken"           : r["network_response"]["action_taken"],
            "net_selected_action"        : r["final_network_action"],
            "dqn_recommendation"         : r["network_response"]["dqn_recommendation"],
            "alert_delivery_mode"        : r["network_response"]["alert_delivery_mode"],
            "network_trust_before"       : r["network_response"]["network_trust_before"],
            "network_trust_after"        : r["network_response"]["network_trust_after"],
            # Notification
            "notify_neighbors"           : r["notify_neighbors"],
            "notify_edge"                : r["notify_edge"],
            "notify_gcc"                 : r["notify_gcc"],
            "response_status"            : r["response_status"],
            "response_success"           : r["response_success"],
        }
        for r in response_records
    ])

    inference_summary = {
        "rows_processed"     : len(df),
        "epsilon_at_inference": agent.dqn_policy.epsilon,
        "available_actions"  : agent.dqn_policy.action_space,
        "device"             : str(TORCH_DEVICE),
    }

    if save_outputs:
        response_df.to_csv("response_output.csv", index=False)
        with open("response_output.jsonl", "w", encoding="utf-8") as f:
            for rec in response_records:
                f.write(json.dumps(rec, default=str) + "\n")
        with open("response_inference_summary.json", "w") as f:
            json.dump(inference_summary, f, indent=2)
        print("[Response] Saved: response_output.csv, .jsonl, summary.json")

    if preview:
        print(response_df[[
            "final_label","severity","safety_override","reactive_triggered",
            "rf_action_taken","net_selected_action",
            "signal_trust_after","network_trust_after","notify_gcc"
        ]].head(15).to_string())

    return agent, response_records, response_df, inference_summary


# ── Run full inference ────────────────────────────────────────────────────────

def evaluate_response(response_df: pd.DataFrame, detection_df: pd.DataFrame) -> dict:
    """
    Evaluate response quality against detection ground truth.
    Mirrors the health summary block from run_full_mas_pipeline() in the original.
    """
    print("=" * 65)
    print("  RESPONSE AGENT EVALUATION")
    print("=" * 65)

    # ── Action distribution by attack type ────────────────────────────────────
    print("\n  RF actions by attack label:")
    print(response_df.groupby("final_label")["rf_action_taken"]
          .value_counts().to_string())

    print("\n  Network actions by attack label:")
    print(response_df.groupby("final_label")["net_selected_action"]
          .value_counts().to_string())

    # ── Trust trajectory ───────────────────────────────────────────────────────
    print("\n  Signal trust (after response):")
    sig_stats = response_df.groupby("final_label")["signal_trust_after"].agg(["mean","min","max"])
    print(sig_stats.round(3).to_string())

    print("\n  Network trust (after response):")
    net_stats = response_df.groupby("final_label")["network_trust_after"].agg(["mean","min","max"])
    print(net_stats.round(3).to_string())

    # ── Tier activation rates ──────────────────────────────────────────────────
    safety_rate   = response_df["safety_override"].mean()
    reactive_rate = response_df["reactive_triggered"].mean()
    gcc_rate      = response_df["notify_gcc"].mean()
    lte_rate      = response_df["lte_fallback_active"].mean()

    print(f"\n  Tier activation rates:")
    print(f"    Safety override   : {safety_rate:.3f} ({safety_rate*100:.1f}%)")
    print(f"    Reactive triggered: {reactive_rate:.3f} ({reactive_rate*100:.1f}%)")
    print(f"    GCC notified      : {gcc_rate:.3f} ({gcc_rate*100:.1f}%)")
    print(f"    LTE fallback      : {lte_rate:.3f} ({lte_rate*100:.1f}%)")

    # ── GCC notification by severity ───────────────────────────────────────────
    print("\n  GCC notification rate by severity:")
    print(response_df.groupby("severity")["notify_gcc"].mean().round(3).to_string())

    # ── Correctness check: are HIGH attacks getting GCC notified? ─────────────
    high_attack = response_df[
        (response_df["severity"] == "HIGH") &
        (response_df["final_label"] != "NORMAL")
    ]
    if len(high_attack) > 0:
        gcc_coverage = high_attack["notify_gcc"].mean()
        print(f"\n  GCC coverage for HIGH-severity attacks: {gcc_coverage:.3f} ({gcc_coverage*100:.1f}%)")

    # ── Safety constraint distribution ────────────────────────────────────────
    print("\n  Safety constraints triggered:")
    print(response_df["safety_constraint"].value_counts().to_string())

    summary = {
        "safety_override_rate"   : round(float(safety_rate),   4),
        "reactive_trigger_rate"  : round(float(reactive_rate), 4),
        "gcc_notification_rate"  : round(float(gcc_rate),      4),
        "lte_fallback_rate"      : round(float(lte_rate),      4),
        "signal_trust_mean"      : round(float(response_df["signal_trust_after"].mean()),  4),
        "network_trust_mean"     : round(float(response_df["network_trust_after"].mean()), 4),
    }
    return summary


def respond_single(detection_input: dict, preview: bool = True) -> dict:
    """
    Run all three response tiers on a single detection window dict.
    Mirrors predict_single() end-to-end from the original.
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
        "edge_alarm"                 : detection_input.get("edge_alarm", 0),
        "cross_layer_anomaly_score"  : detection_input.get("cross_layer_anomaly_score", 0.0),
        "snr_db"                     : detection_input.get("snr_db", 10.0),
        "channel_occupancy_pct"      : detection_input.get("channel_occupancy_pct", 20.0),
        "packet_delivery_ratio"      : detection_input.get("packet_delivery_ratio", 1.0),
        "half_open_connections"      : detection_input.get("half_open_connections", 0),
        "src_ip_entropy"             : detection_input.get("src_ip_entropy", 2.0),
        "error_response_rate"        : detection_input.get("error_response_rate", 0.0),
        "consecutive_attack_count"   : detection_input.get("consecutive_attack_count", 0),
        "packets_per_second"         : detection_input.get("packets_per_second", 50.0),
        "syn_packet_rate"            : detection_input.get("syn_packet_rate", 0.0),
        "rssi_dbm"                   : detection_input.get("rssi_dbm", -65.0),
    }

    agent  = load_response_agent()
    result = agent.respond(row)

    output = {
        "final_label"          : result["final_label"],
        "severity"             : result["severity"],
        "safety_override"      : result["safety_response"].get("safety_override", False),
        "safety_forced_action" : result["safety_response"].get("forced_action", "NONE"),
        "reactive_triggered"   : result["reactive_response"].get("reactive_triggered", False),
        "reactive_actions"     : result["reactive_response"].get("reactive_actions_taken", []),
        "rf_action"            : result["rf_response"]["action_taken"],
        "rf_channel_mode"      : result["rf_response"]["channel_mode"],
        "signal_trust_after"   : round(result["rf_response"]["signal_trust_after"], 4),
        "lte_fallback_active"  : result["rf_response"]["lte_fallback_active"],
        "net_action"           : result["final_network_action"],
        "dqn_recommendation"   : result["network_response"]["dqn_recommendation"],
        "network_trust_after"  : round(result["network_response"]["network_trust_after"], 4),
        "notify_gcc"           : result["notify_gcc"],
        "notify_edge"          : result["notify_edge"],
    }

    if preview:
        print("=" * 58)
        print(f"  LABEL      : {output['final_label']}  [{output['severity']}]")
        print(f"  SAFETY     : override={output['safety_override']}  "
              f"forced={output['safety_forced_action']}")
        print(f"  REACTIVE   : triggered={output['reactive_triggered']}  "
              f"actions={output['reactive_actions']}")
        print(f"  RF         : {output['rf_action']}  mode={output['rf_channel_mode']}")
        print(f"  SIG TRUST  : {output['signal_trust_after']}  "
              f"LTE={output['lte_fallback_active']}")
        print(f"  NETWORK    : {output['net_action']}  (DQN: {output['dqn_recommendation']})")
        print(f"  NET TRUST  : {output['network_trust_after']}")
        print(f"  NOTIFY     : GCC={output['notify_gcc']}  Edge={output['notify_edge']}")
        print("=" * 58)
    return output


# ── Five canonical tests ──────────────────────────────────────────────────────


