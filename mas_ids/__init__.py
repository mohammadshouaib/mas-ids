"""
MAS-IDS: Multi-Agent Intrusion Detection System for DoS/DDoS and Jamming
=========================================================================
A six-agent pipeline for UAV/UGV network intrusion detection.

Quick start
-----------
    from mas_ids.config import setup_environment
    from mas_ids.pipeline import run_full_mas_pipeline

    setup_environment()                 # seeds, GPU, output dirs
    results = run_full_mas_pipeline()   # runs all 6 agents end-to-end

Agents
------
    1. data_agent          — synthetic generation, collection, cleaning
    2. feature_agent       — temporal / cross-layer / swarm features
    3. detection_agent     — CNN+LSTM, LSTM-AE, GRU, DBSCAN, fusion
    4. response_agent      — safety / reactive / DQN MARL
    5. coordination_agent  — Beta trust, Bayesian, Nash equilibrium
    6. logger_agent        — SHA-256, Merkle tree, Bloom filter, sketches
"""
__version__ = "1.1.0"

from . import config
from . import utils
from . import pipeline

__all__ = ["config", "utils", "pipeline"]
