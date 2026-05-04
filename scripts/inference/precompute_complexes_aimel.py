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
        description="Precompute AIMEl complexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example
        """,
    )
    parser.add_argument('--data-pkl', type=str,
                        default=os.path.join(
                            REPO_ROOT, 'data_curation', 'molecular',
                            'aimel_clustered_molecular.pkl'))
    parser.add_argument('--complexes-pkl', type=str, default=None,
                        help='Pre-built complexes cache; auto-named if omitted')
    
    # Graph connectivity
    parser.add_argument('--cutoff', type=float, default=8.0,
                        help='Distance cutoff in Bohr for edge creation')
    parser.add_argument('--max-neighbors', type=int, default=12,
                        help='Max neighbours per atom (0 = no limit)')
    parser.add_argument('--fully-connected', action='store_true',
                        help='Use fully connected graph (no cutoff)')
    parser.add_argument('--max-dim', type=int, default=1,
                        help='Max cochain dimension (1 skips ring bags)')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Auto-name complexes cache
    # ------------------------------------------------------------------
    def _connectivity_suffix() -> str:
        if args.fully_connected:
            return 'fc'
        s = f'cutoff{args.cutoff}'
        if args.max_neighbors > 0:
            s += f'_max{args.max_neighbors}'
        return s

    _mol_data_dir = os.path.join(REPO_ROOT, 'data_curation', 'molecular')
    if args.complexes_pkl is None:
        args.complexes_pkl = os.path.join(
            _mol_data_dir,
            f'precomputed_complexes_aimel_{_connectivity_suffix()}.pkl',
        )

    print(f"\nLoading dataset: {args.data_pkl}")
    df = pd.read_pickle(args.data_pkl)
    print(f"  {len(df)} molecules")

    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}

    max_neighbors = args.max_neighbors if args.max_neighbors > 0 else None
    if os.path.exists(args.complexes_pkl):
        print(f"Loading precomputed complexes from {args.complexes_pkl}")
        with open(args.complexes_pkl, 'rb') as f:
            complexes = pickle.load(f)
        print(f"  Loaded {len(complexes)} complexes")
    else:
        print("Precomputing complexes (this is shared; run only once)...")
        complexes = precompute_complexes(
            df,
            element_to_idx=element_to_idx,
            cutoff=args.cutoff,
            max_neighbors=max_neighbors,
            fully_connected=args.fully_connected,
            max_dim=args.max_dim,
            output_path=args.complexes_pkl,
            verbose=True,
        )

        print('Done')


if __name__ == '__main__':
    main()