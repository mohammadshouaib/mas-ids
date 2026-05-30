"""
mas_ids.pipeline
================
End-to-end orchestration: run_full_mas_pipeline() runs all six agents in order
and returns a dict with every intermediate DataFrame plus summary stats.

Usage:
    from mas_ids.config import setup_environment
    from mas_ids.pipeline import run_full_mas_pipeline
    setup_environment()
    results = run_full_mas_pipeline()        # all defaults from config
    detection_df = results["detection_df"]
"""
import numpy as np
import pandas as pd

from .config import (
    N_NORMAL, N_JAMMING, N_DOS, N_DDOS, N_HYBRID,
    TRAIN_EPOCHS, TRAIN_BATCH, DQN_EPISODES, DQN_BATCH, DQN_SYNC_EVERY,
    KPI_WINDOW, LOG_BATCH_SIZE, LOG_PREFIX, SEQ_LEN, GLOBAL_SEED, LABEL_MAP,
)
from .utils import (
    NullLogger, safe_numeric_fill, fit_scaler_on_normal, apply_scaler,
    create_sequences, split_train_test, balance_sequences,
)
from .agents.data_agent import (
    TrafficDataGenerator, EdgeCollector, GCCCollector,
    DataQualityChecker, DataCleaner,
    JAMMING_FEATURES, DOS_FEATURES,
)
from .agents.feature_agent import (
    TemporalJammingFeatureBuilder, TemporalDoSFeatureBuilder,
    CrossLayerFeatureBuilder, SwarmConsensusFeatureBuilder,
    select_features_by_mutual_info,
)
from .agents.detection_agent import (
    train_detection_models, load_detection_models,
    run_detection_agent_inference, evaluate_detection,
)
from .agents.response_agent import (
    train_response_policy, run_response_agent_inference,
)
from .agents.coordination_agent import (
    run_coordination_agent, run_management_agent,
)
from .agents.logger_agent import LoggerAgent, run_logger_agent


def run_full_mas_pipeline(
    cleaned_df = None,           # pass a pre-loaded DataFrame to skip generation
    n_normal   : int   = N_NORMAL,
    n_jamming  : int   = N_JAMMING,
    n_dos      : int   = N_DOS,
    n_ddos     : int   = N_DDOS,
    n_hybrid   : int   = N_HYBRID,
    train_epochs: int  = TRAIN_EPOCHS,
    dqn_episodes: int  = DQN_EPISODES,
    preview    : bool  = True,
) -> dict:
    """
    Run the complete 6-agent MAS-IDS pipeline end-to-end.
    Returns a dict with all intermediate DataFrames and summary stats.
    """
    import warnings; warnings.filterwarnings('ignore')

    # ── 1. Data Collection & Cleaning ─────────────────────────────────────────
    if preview: print('\n=== Agent 1: Data Collection & Cleaning ===')
    if cleaned_df is not None:
        cdf = cleaned_df.copy()
        if preview: print(f'  Using provided cleaned_df: {cdf.shape}')
    else:
        gen  = TrafficDataGenerator(seed=GLOBAL_SEED)
        rdf  = gen.generate(n_normal=n_normal, n_jamming=n_jamming,
                            n_dos=n_dos, n_ddos=n_ddos, n_hybrid=n_hybrid)
        emask = rdf['collection_point'].isin(['edge','edge_gcc'])
        gmask = rdf['collection_point'].isin(['gcc', 'edge_gcc'])
        ec = EdgeCollector('uav_001','uav'); gc = GCCCollector('gcc_001')
        chk = DataQualityChecker()
        clr = DataCleaner(drop_thresh=0.5, scale=True)
        combined = chk.check(pd.concat(
            [ec.collect_from_dataframe(rdf[emask].copy()),
             gc.collect_from_dataframe(rdf[gmask].copy())], ignore_index=True))
        cdf   = clr.fit_transform(combined.copy())
    # Edge/GCC split — runs for BOTH the synthetic and provided-cleaned_df paths.
    # If a provided cleaned_df has no collection_point column, treat every row as
    # available to both tiers (edge_gcc) so neither split is empty.
    if 'collection_point' not in cdf.columns:
        cdf['collection_point'] = 'edge_gcc'
    e_df  = cdf[cdf['collection_point'].isin(['edge','edge_gcc'])].copy()
    g_df  = cdf[cdf['collection_point'].isin(['gcc', 'edge_gcc'])].copy()
    # Guard against an empty tier (e.g. a dataset with only network-layer rows)
    if len(e_df) == 0: e_df = cdf.copy()
    if len(g_df) == 0: g_df = cdf.copy()
    if preview: print(f'  cleaned_df: {cdf.shape}  Labels: {dict(cdf["label"].value_counts())}')

    # ── ID-column inspection (diagnostic only) ────────────────────────────────
    # Surfaces any column that could be used to identify a node/source/host.
    # Helps diagnose the "DBSCAN: 1 node(s)" collapse — if a real identifier is
    # in the data but isn't being routed to swarm/coordination, this print will
    # make it obvious. Purely informational; does not change pipeline behaviour.
    if preview:
        print('  [diag] columns:', cdf.columns.tolist())
        id_keys = ('node', 'id', 'src', 'dst', 'mac', 'ip',
                   'device', 'host', 'addr')
        id_cols = [c for c in cdf.columns
                   if any(k in c.lower() for k in id_keys)]
        if id_cols:
            print('  [diag] id-like columns (name, n_unique, sample):')
            for c in id_cols:
                try:
                    n = cdf[c].nunique(dropna=True)
                    samp = cdf[c].dropna().iloc[0] if cdf[c].notna().any() else 'NA'
                    print(f'    {c:30s} n_unique={n:<8} sample={samp!r}')
                except Exception as ex:
                    print(f'    {c:30s} <unreadable: {ex}>')
        else:
            print('  [diag] no id-like columns found')

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    if preview: print('\n=== Agent 2: Feature Engineering ===')
    _jb = TemporalJammingFeatureBuilder(); _db = TemporalDoSFeatureBuilder()
    ef   = _jb.build(e_df.copy()); gf = _db.build(g_df.copy())
    adf  = cdf.copy()
    for c in [col for col in ef.columns if col not in cdf.columns]:
        adf[c] = np.nan; idx = ef.index.intersection(adf.index)
        adf.loc[idx, c] = ef.loc[idx, c]
    for c in [col for col in gf.columns
              if col not in cdf.columns and col not in ef.columns]:
        if c not in adf.columns: adf[c] = np.nan
        idx = gf.index.intersection(adf.index); adf.loc[idx, c] = gf.loc[idx, c]
    adf = adf.fillna(0.0)
    adf = CrossLayerFeatureBuilder().build(adf)
    adf = SwarmConsensusFeatureBuilder().build(adf)
    jf   = select_features_by_mutual_info(adf, list(set(
               [c for c in adf.columns
                if any(c.startswith(f) or f in c for f in JAMMING_FEATURES)
                and c not in ['label','timestamp','node_id','window_id','source_row_id']]
           )), min_mi=0.005)
    df   = select_features_by_mutual_info(adf, list(set(
               [c for c in adf.columns
                if any(c.startswith(f) or f in c for f in DOS_FEATURES)
                and c not in ['label','timestamp','node_id','window_id','source_row_id']]
           )), min_mi=0.005)
    xf   = [c for c in [
        'jam_congestion_coupling','cross_layer_loss_index','noise_traffic_coupling',
        'cross_layer_anomaly_score','cross_layer_attack_flag','swarm_consensus_anomaly_score',
        'swarm_anomaly_flag','mac_loss_rate','phy_app_loss_divergence','entropy_asymmetry',
    ] if c in adf.columns]
    hf   = list(set(jf+df+xf))
    proc = safe_numeric_fill(adf, hf)
    sj   = fit_scaler_on_normal(proc, jf); sd = fit_scaler_on_normal(proc, df)
    # Cross-layer-only features (in the hybrid set but not in jam or dos) get their
    # own scaler so every column is scaled exactly ONCE.
    xf_only = [c for c in xf if c not in jf and c not in df]
    sx   = fit_scaler_on_normal(proc, xf_only) if xf_only else None
    # BUG FIX (double-scaling): the previous code did
    #     fdf = apply_scaler(apply_scaler(apply_scaler(proc, jf, sj), df, sd), hf, sh)
    # Since hf == set(jf+df+xf), the third pass re-scaled the ALREADY-scaled
    # jam/DoS columns with a scaler fit on RAW values. Double-scaling collapsed the
    # dynamic range (a DoS flood and a normal window ended up ~0.13 apart instead of
    # cleanly separated) — the reason the DoS/hybrid CNN+LSTM flatlined at 0.8169
    # and the DoS AE/GRU losses exploded to ~1e8.
    #
    # Now each column is scaled once with its own branch scaler. Because the hybrid
    # feature set is the union of these same columns, the hybrid sequences are built
    # from the SAME consistently-scaled frame — so training and inference (which
    # both read fdf) match exactly.
    fdf  = apply_scaler(apply_scaler(proc, jf, sj), df, sd)
    if sx is not None:
        fdf = apply_scaler(fdf, xf_only, sx)
    sh   = None  # retained key below for backward-compat of fe_state
    Xj,yj,_ = create_sequences(fdf, jf, SEQ_LEN)
    Xd,yd,_ = create_sequences(fdf, df, SEQ_LEN)
    Xh,yh,_ = create_sequences(fdf, hf, SEQ_LEN)
    def _sp(X,y): return split_train_test(X,y,[])[:4]
    Xjtr,Xjte,yjtr,yjte = _sp(Xj,yj)
    Xdtr,Xdte,ydtr,ydte = _sp(Xd,yd)
    Xhtr,Xhte,yhtr,yhte = _sp(Xh,yh)
    Xjtr,yjtr = balance_sequences(Xjtr,yjtr,LABEL_MAP,0.40)
    Xdtr,ydtr = balance_sequences(Xdtr,ydtr,LABEL_MAP,0.40)
    Xhtr,yhtr = balance_sequences(Xhtr,yhtr,LABEL_MAP,0.40)
    _fe = {'JAM_FEAT_FINAL':jf,'DOS_FEAT_FINAL':df,'CROSS_FEATS':xf,
           'HYBRID_FEAT_FINAL':hf,'scaler_jam':sj,'scaler_dos':sd,
           'scaler_cross':sx,'seq_len':SEQ_LEN,'label_map':LABEL_MAP}
    if preview: print(f'  JAM:{len(jf)} DoS:{len(df)} Hybrid:{len(hf)} features')

    # ── 3. Detection ──────────────────────────────────────────────────────────
    if preview: print('\n=== Agent 3: Detection ===')
    _arr = {'X_jam_tr':Xjtr,'y_jam_tr':yjtr,'X_dos_tr':Xdtr,'y_dos_tr':ydtr,
            'X_hyb_tr':Xhtr,'y_hyb_tr':yhtr,'X_jam_te':Xjte,'y_jam_te':yjte,
            'X_dos_te':Xdte,'y_dos_te':ydte,'X_hyb_te':Xhte,'y_hyb_te':yhte}
    _tr  = train_detection_models(_arr, _fe, SEQ_LEN, train_epochs, TRAIN_BATCH, True, preview)
    _tr.update(load_detection_models())
    det_df = run_detection_agent_inference(fdf, _tr, _fe, False, preview)
    ev     = evaluate_detection(det_df)
    if preview: print(f'  F1={ev.get("f1",0):.4f}  |  {det_df["final_label"].value_counts().to_dict()}')

    # ── 4. Response ───────────────────────────────────────────────────────────
    if preview: print('\n=== Agent 4: Response ===')
    nn = len(det_df[det_df['final_label']=='NORMAL']); na = len(det_df[det_df['final_label']!='NORMAL'])
    dqn_df = pd.concat([
        det_df[det_df['final_label']=='NORMAL'].sample(min(2000,nn),random_state=GLOBAL_SEED,replace=min(2000,nn)>nn),
        det_df[det_df['final_label']!='NORMAL'].sample(min(2000,max(na,1)),random_state=GLOBAL_SEED,replace=min(2000,max(na,1))>na)
        if na>0 else pd.DataFrame()
    ], ignore_index=True).sample(frac=1,random_state=GLOBAL_SEED).reset_index(drop=True)
    _tp, _ts = train_response_policy(
        dqn_df, dqn_episodes, DQN_BATCH, DQN_SYNC_EVERY, 999999, True, preview)
    _lg = NullLogger()
    _, _rr, rsp_df, _ = run_response_agent_inference(det_df, _lg, False, False)
    if preview: print(f'  Rows: {len(rsp_df)}  |  Safety overrides: {int(rsp_df["safety_override"].sum())}')

    # ── 5. Coordination & Management ──────────────────────────────────────────
    if preview: print('\n=== Agent 5: Coordination & Management ===')
    _ca, _cr, crd_df, _tdf = run_coordination_agent(det_df, rsp_df, _lg, False, False, 3, 0.5, 'e1')[:4]
    _ma, _dir, mgt_df, _kpi = run_management_agent(det_df, rsp_df, crd_df, _lg, False, False, 'gcc_001', KPI_WINDOW)
    if preview:
        print(f'  Coord logs: {len(crd_df)}  |  Mgmt batches: {len(_dir)}')
        print(f'  Mean game_value: {mgt_df["game_value"].mean():.4f}')

    # ── 6. Logger ─────────────────────────────────────────────────────────────
    if preview: print('\n=== Agent 6: Logger ===')
    _full_lg = LoggerAgent(LOG_BATCH_SIZE, 'v1.1')
    _lg_out  = run_logger_agent(det_df, rsp_df, crd_df, mgt_df,
                                LOG_BATCH_SIZE, 'v1.1', False)
    _full_lg, _ts2, _ps, _bf, _ir = _lg_out
    _lf = _full_lg.save_logs(prefix=LOG_PREFIX)
    if preview:
        print(f'  Events: {len(_full_lg.logs)}  |  Integrity: {_ir["status"]}')

    if preview:
        print('\n' + '='*65)
        print('  MAS-IDS UNIFIED — PIPELINE COMPLETE')
        print('='*65)
        print(f'  Detection F1    : {ev.get("f1",0):.4f}')
        print(f'  Precision       : {ev.get("precision",0):.4f}')
        print(f'  Recall          : {ev.get("recall",0):.4f}')
        print(f'  Coord attacks   : {int(crd_df.get("coordinated_attack_detected",pd.Series(False)).sum())}')
        print(f'  Logger events   : {len(_full_lg.logs)}')
        print(f'  Integrity       : {_ir["status"]}')
        print('='*65)

    return {
        'cleaned_df'   : cdf,
        'feat_df'      : fdf,
        'fe_state'     : _fe,
        'detection_df' : det_df,
        'eval_results' : ev,
        'response_df'  : rsp_df,
        'coord_df'     : crd_df,
        'trust_df'     : _tdf,
        'mgmt_df'      : mgt_df,
        'logger'       : _full_lg,
        'integrity'    : _ir,
        'det_models'   : _tr,
        'response_policy': _tp,
    }