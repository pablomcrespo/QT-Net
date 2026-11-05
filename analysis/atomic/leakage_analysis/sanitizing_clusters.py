"""
Shared constants and helpers of the holdout leakage analysis: the SOAP settings,
the streamed covariance used for PCA, and the holdout loader that
``leakage_vs_radius`` reuses.

Memory: raw SOAP (3696-dim) for all aimel atoms is ~8 GB in float32, so it is
streamed to per-element memmaps and PCA is computed from a streamed covariance
matrix, transforming atoms chunk-wise.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# atomic_env_split and result_analysis sit in different sub-trees of the repo.
for _rel in ("../../../data_curation/atomic/cluster_analysis", ".."):
    _dir = os.path.abspath(os.path.join(_HERE, _rel))
    if _dir not in sys.path:
        sys.path.append(_dir)

ELEMENTS = ["H", "C", "N", "O"]
RANDOM_STATE = 42

# The cutoff the 8-Bohr models use; Bohr.
SOAP8_PARAMS: Dict = {
    "r_cut": 8.0,
    "n_max": 8,
    "l_max": 6,
    "sigma": 0.7,
    "species": ELEMENTS,
    "periodic": False,
    "sparse": False,
}

# The paper's held-out environment labels (4.5-Bohr labels).
HOLDOUT_TARGET_LABELS = {"H_10", "C_11", "N_13", "O_10"}

PROPERTIES = ["N", "LI", "|Mu|", "|Q|"]


def _streaming_mean_cov(mm: np.memmap, chunk: int = 20_000) -> Tuple[np.ndarray, np.ndarray]:
    """Exact mean and covariance of a (n, d) memmap, one chunked pass."""
    n, d = mm.shape
    s = np.zeros(d, dtype=np.float64)
    ss = np.zeros((d, d), dtype=np.float64)
    for i in range(0, n, chunk):
        X = np.asarray(mm[i:i + chunk], dtype=np.float64)
        s += X.sum(axis=0)
        ss += X.T @ X
    mean = s / n
    cov = (ss - n * np.outer(mean, mean)) / (n - 1)
    return mean, cov


def load_holdout_positions(aimel_df: pd.DataFrame, test_pkl_path: str) -> np.ndarray:
    """Positional rows (reset-index) of aimel molecules that are in the original
    held-out set, matched by the 'file' column."""
    test_df = pd.read_pickle(test_pkl_path)
    holdout_files = set(test_df["file"])
    work = aimel_df.reset_index(drop=True)
    pos = work.index[work["file"].isin(holdout_files)].to_numpy()
    assert len(pos) == len(test_df), (len(pos), len(test_df))
    return pos
