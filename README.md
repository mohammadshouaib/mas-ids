# MAS-IDS — Multi-Agent Intrusion Detection System

A research-grade, six-agent intrusion detection system for **DoS / DDoS and
Jamming** attacks in UAV/UGV (drone / ground-robot) networks. The codebase is a
clean, modular Python package that you can host on GitHub and import directly
into a Kaggle notebook for GPU training.

---

## Architecture

| # | Agent | Module | Role |
|---|-------|--------|------|
| 1 | Data Collection & Cleaning | `mas_ids/agents/data_agent.py` | Feature schema, synthetic generator, two-tier collection (Edge + GCC), quality checks, normalisation |
| 2 | Feature Engineering | `mas_ids/agents/feature_agent.py` | Temporal (EWMA, CUSUM, rolling), cross-layer fusion, swarm consensus, MI feature selection, sequence windows, balancing |
| 3 | Detection | `mas_ids/agents/detection_agent.py` | CNN+LSTM classifiers, LSTM-AE anomaly scoring, GRU edge predictors, DBSCAN, three-tier fusion |
| 4 | Response | `mas_ids/agents/response_agent.py` | Safety layer + fast reactive controller + DQN MARL (PyTorch) |
| 5 | Coordination & Management | `mas_ids/agents/coordination_agent.py` | Beta trust model, Bayesian coordinator, game-theoretic Nash equilibrium, KPI dashboard |
| 6 | Logger | `mas_ids/agents/logger_agent.py` | SHA-256 per-event hashing, Merkle-tree batches, Bloom filter for DDoS sources, traffic/PHY sketches, tamper detection |

Shared infrastructure:

- `mas_ids/config.py` — all tunable constants in one place + `setup_environment()`
- `mas_ids/utils.py` — `NullLogger` and shared helpers (scaling, sequences, etc.)
- `mas_ids/pipeline.py` — `run_full_mas_pipeline()` runs all six agents end-to-end

**Data flow:**

```
raw data -> cleaned_df -> feat_df -> detection_df -> response_df
         -> coord_df + mgmt_df -> logger (integrity-verified event log)
```

---

## Repository layout

```
mas-ids/
├── mas_ids/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── pipeline.py
│   ├── agents/                 # the six detection/response agents
│   │   ├── __init__.py
│   │   ├── data_agent.py
│   │   ├── feature_agent.py
│   │   ├── detection_agent.py
│   │   ├── response_agent.py
│   │   ├── coordination_agent.py
│   │   └── logger_agent.py
│   └── data/                   # dataset ingestion & adapters
│       ├── __init__.py
│       └── uav_nidd_loader.py
├── notebooks/
│   └── kaggle_runner.ipynb     # the driver you run on Kaggle
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Quick start (local)

```bash
pip install -r requirements.txt
python -c "from mas_ids.config import setup_environment; \
           from mas_ids.pipeline import run_full_mas_pipeline; \
           setup_environment(); run_full_mas_pipeline()"
```

Or in Python:

```python
from mas_ids.config import setup_environment
from mas_ids.pipeline import run_full_mas_pipeline

setup_environment()                     # seeds, GPU growth, mixed precision, dirs
results = run_full_mas_pipeline()       # runs all 6 agents

detection_df = results["detection_df"]
print("F1:", results["eval_results"]["f1"])
print("Integrity:", results["integrity"]["status"])
```

---

## Running on Kaggle (import from GitHub)

1. **Create a GitHub repo** and push this project:

   ```bash
   cd mas-ids
   git init
   git add .
   git commit -m "MAS-IDS modular package"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/mas-ids.git
   git push -u origin main
   ```

2. **On Kaggle**, create a new notebook and turn on:
   - Settings → **Accelerator: GPU**
   - Settings → **Internet: On** (needed to clone the repo)

3. **Upload `notebooks/kaggle_runner.ipynb`** (or copy its cells), then edit the
   first code cell:

   ```python
   GITHUB_USER = "YOUR_GITHUB_USERNAME"
   REPO        = "mas-ids"
   BRANCH      = "main"
   ```

4. **Run all cells.** The notebook clones your repo, puts it on `sys.path`, and
   runs the full pipeline. Whenever you push new code to GitHub, just re-run the
   clone cell to pull the latest version.

### Offline alternative (no Internet)

If you cannot enable Internet on Kaggle, upload the `mas_ids/` folder as a
Kaggle **Dataset** and set:

```python
import sys
sys.path.insert(0, "/kaggle/input/<your-dataset-slug>")
```

Everything else stays the same.

---

## Dataset

The system is **fully self-contained**: `TrafficDataGenerator` synthesises
labelled UAV/UGV traffic windows for all five classes
(`normal, jamming, dos, ddos, hybrid`). **No external dataset upload is
required.**

To use a **real dataset** instead, replace the generator call inside
`run_full_mas_pipeline()` (or your own driver) with:

```python
raw_df = pd.read_csv("/kaggle/input/<your-dataset>/traffic.csv")
```

Your CSV must contain the columns defined in
`mas_ids/agents/data_agent.py` (`JAMMING_FEATURES`, `DOS_FEATURES`,
metadata columns, and a `label` column).

---

## Configuration

All knobs live in `mas_ids/config.py`. Override them before calling the
pipeline:

```python
import mas_ids.config as cfg
cfg.TRAIN_EPOCHS = 30
cfg.DQN_EPISODES = 200
cfg.N_NORMAL     = 5000
```

| Constant | Meaning | Default |
|----------|---------|---------|
| `SEQ_LEN` | sliding-window length | 10 |
| `TRAIN_EPOCHS` | CNN+LSTM / LSTM-AE epochs | 50 |
| `DQN_EPISODES` | Response DQN episodes | 150 |
| `KPI_WINDOW` | rows per Nash solve | 30 |
| `LOG_BATCH_SIZE` | Merkle batch size | 10 |
| `N_NORMAL / N_JAMMING / N_DOS / N_DDOS / N_HYBRID` | synthetic sample counts | 2000 / 800 / 600 / 600 / 400 |

---

## Debugging individual agents

Because each agent is its own module, you can import and run one in isolation:

```python
from mas_ids.agents.data_agent import TrafficDataGenerator
from mas_ids.agents.feature_agent import prepare_feature_dataframe

raw = TrafficDataGenerator(seed=42).generate(n_normal=200, n_jamming=80,
                                              n_dos=60, n_ddos=60, n_hybrid=40)
feats = prepare_feature_dataframe(raw)
```

Single-window inference helpers are available for deployment-style tests:
`predict_single` (detection), `respond_single` (response),
`coordinate_single` (coordination).

---

## Outputs

Written to the working directory (`/kaggle/working` on Kaggle):

- `saved_detection_models/` — trained `.keras` models + `detection_config.json`
- `saved_response_models/` — DQN policy weights (`.pt`) + config
- `mas_ids_unified_logs.csv` / `.jsonl` — tamper-evident event log
- `mas_ids_unified_merkle_batches.csv` — Merkle integrity batches

---

## Requirements

Python ≥ 3.9 and:

```
numpy, pandas, scipy, scikit-learn, tensorflow >= 2.15, torch >= 2.0
```

The DQN runs on CPU by default (`TORCH_DEVICE = cpu`) to avoid CUDA/runtime
mismatches; the TensorFlow models use the GPU when available (with mixed
precision). This matches Kaggle's standard GPU image.

---

## License

MIT — see `LICENSE`.

---

## UAV-NIDD dataset integration — feature coverage

When using the real UAV-NIDD dataset (via `mas_ids/data/uav_nidd_loader.py`), the loader derives
features from raw WiFi/radiotap packet captures and maps them onto the threat-model
schema. Coverage:

**Populated with real values (25 of 28 schema features):**

- *Jamming (PHY/MAC):* `rssi_dbm`, `snr_db`, `sinr_db` (SNR proxy), `noise_floor_dbm`,
  `bit_error_rate`, `bad_packet_ratio`, `packet_delivery_ratio`, `mac_retry_count`,
  `retransmission_count`, `channel_occupancy_pct`, `cca_failure_count`, `backoff_count`
- *DoS/DDoS (NET/TRANSPORT):* `packets_per_second`, `bytes_per_second`,
  `unique_source_count`, `src_ip_entropy`, `dst_ip_entropy`, `icmp_packet_rate`,
  `connection_attempts_per_sec`, `half_open_connections`, `tcp_retransmission_rate`,
  `udp_packet_rate`, `syn_packet_rate`, `port_scan_score`

**Intentionally absent (3 features — not present in WiFi-frame captures):**

- Application-layer L7 metrics: `api_request_rate`, `command_frequency`,
  `error_response_rate`. The threat model marks L7 as *optional*; these require
  application-level instrumentation that packet captures do not contain. They are
  omitted rather than injected as misleading zeros.

**Proxy-derived features (documented approximations):**

- `sinr_db` ← SNR (no separate interference measurement in capture)
- `bit_error_rate`, `cca_failure_count` ← derived from bad-FCS rate
- `connection_attempts_per_sec`, `syn_packet_rate` ← TCP-ACK activity rate
  (raw TCP flag bits are not in the capture)
- `backoff_count` ← WLAN sequence-number variance (contention proxy)

### Window labeling

The loader aggregates packet-level rows into 1-second windows. A window is labeled
with its dominant **attack** class when attack packets exceed `attack_frac` (default
0.20) of the window; otherwise `normal`. This attack-priority rule prevents simple
majority voting from erasing minority attack classes — important because attacks are
often interleaved with normal traffic. Tune via:

```python
load_uav_nidd_split(csv_dir=..., attack_frac=0.20)
```
