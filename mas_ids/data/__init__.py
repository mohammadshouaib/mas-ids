"""
mas_ids.data — dataset ingestion and adapters.

This subpackage holds everything that brings external or generated data INTO
the pipeline in the cleaned-window schema the agents expect. It is deliberately
separate from `mas_ids.agents`, which contains the detection/response logic.

Loaders:
    uav_nidd_loader — adapts the real UAV-NIDD dataset (802.11 radiotap CSVs)
                      onto the mas_ids feature schema.
"""
from .uav_nidd_loader import load_uav_nidd, load_uav_nidd_split, load_uav_nidd_perpacket

__all__ = ["load_uav_nidd", "load_uav_nidd_split", "load_uav_nidd_perpacket"]
