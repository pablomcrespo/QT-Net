#!/usr/bin/env python3

import argparse
import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from flax import nnx

# ---------------------------------------------------------------------------
# Repo / project root (mirrors train_QTNet_ensemble.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, 'data_curation')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(os.path.join(start_dir, '..', '..'))
        cur = parent


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

for _root, _dirs, _ in os.walk(REPO_ROOT):
    if 'qtnet' in _dirs:
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break

_proj_candidate = os.path.dirname(REPO_ROOT)
if (os.path.isdir(os.path.join(_proj_candidate, 'experiments')) or
        os.path.exists(os.path.join(_proj_candidate, 'qm9-aim-jax.sif')) or
        os.path.exists(os.path.join(_proj_candidate, 'pyproject.toml'))):
    PROJECT_ROOT = _proj_candidate
else:
    PROJECT_ROOT = REPO_ROOT

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from qtnet.jax_models.models_inference import SGNN_v2, EquivariantGNN_Flex
from qtnet.jax_models.representations import (
    precompute_complexes,
    prepare_padded_batches,
)
from qtnet.jax_models.train_utils import load_checkpoint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_ELEMENTS = ['H', 'C', 'N', 'O']
NUM_SPECIES = len(ALL_ELEMENTS)

TARGET_COLUMNS = [
    'N', 'LI', 'Mu_X', 'Mu_Y', 'Mu_Z',
    'Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ',
]


def main():

    parser = argparse.ArgumentParser(
        description="Precompute QM9 complexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example
        """,
    )
    parser.add_argument('--qm9-pkl', type=str,
                        default=os.path.join(
                            REPO_ROOT, 'data_curation', 'molecular',
                            'qm9_filtered.pkl'))
    parser.add_argument('--qm9-complexes-pkl', type=str, default=None,
                        help='Cache path for QM9 cutoff-graph complexes; '
                             'auto-named from connectivity settings if omitted')
    args = parser.parse_args()

    print(f"\nLoading QM9: {args.qm9_pkl}")
    qm9 = pd.read_pickle(args.qm9_pkl)
    print(f"  {len(qm9)} molecules")

    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}

    # ------------------------------------------------------------------
    # 3. Precompute or load cutoff-graph complexes for QM9
    # ------------------------------------------------------------------
    max_neighbors = 12
    max_neighbors_arg = max_neighbors if max_neighbors > 0 else None
    #fully_connected = ref_cfg.get('fully_connected', False)
    cutoff = 8.0
    max_dim = 1

    if args.qm9_complexes_pkl is None:
        suffix = f'cutoff{cutoff}_max{max_neighbors}'
        args.qm9_complexes_pkl = os.path.join(
            REPO_ROOT, 'data_curation', 'molecular',
            f'precomputed_complexes_qm9_{suffix}.pkl',
        )

        print(f"\nPrecomputing QM9 cutoff-graph complexes...")
        print(f"  cutoff={cutoff}  max_neighbors={max_neighbors}  "
              f"fully_connected=False  max_dim={max_dim}")
        qm9_complexes = precompute_complexes(
            qm9,
            element_to_idx=element_to_idx,
            cutoff=cutoff,
            max_neighbors=max_neighbors_arg,
            fully_connected=False,
            max_dim=max_dim,
            output_path=args.qm9_complexes_pkl,
            verbose=True,
        )

        print('Done')


if __name__ == '__main__':
    main()