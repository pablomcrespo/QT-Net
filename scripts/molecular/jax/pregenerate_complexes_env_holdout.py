#!/usr/bin/env python3
"""
Precompute the molecular complexes of the environment-holdout study.

One distance-cutoff complex per molecule (``row_to_complex``: edges within
``--cutoff`` Bohr, union of each atom's ``--max-neighbors`` nearest), nodes and
edges only (``max_dim=1``: GeoScalarGNNMolecular reads no bags), with node
positions, edge gyration tensors and the 10 per-atom QT-Net features in
``node_static``.  The blind variant simply ignores the features, so both
variants share one file per population.

Two populations, the SAME graph rule:
  aimel      data_curation/molecular/aimel_clustered_molecular.pkl
             (30,812 molecules, ground-truth QTAIM features)
             → precomputed_geo_aimel.pkl
  remainder  data_curation/molecular/qm9_inferred.pkl
             (98,096 QM9 molecules outside AIMEl, features inferred by the
             QT-Net ensemble — infer_QTAIM_QM9.py)
             → precomputed_geo_qm9_inferred.pkl

Complexes are keyed by the QM9 ``index`` (never by row position), so every
downstream join is on the molecule id.

Why 3.5 Bohr / 4 neighbours (not the 6-Bohr radius of the environment
classification): a short, sparse graph limits the capacity of the molecular
model, while its perception radius after message passing still covers the
environment — with 4 layers, 98.3 % of the atoms inside each atom's 6-Bohr
sphere are reached (88.1 % of atoms have their whole 6-Bohr environment
reached; 3,000 random AIMEl molecules).

Usage:
    python pregenerate_complexes_env_holdout.py                      # both
    python pregenerate_complexes_env_holdout.py --populations aimel
    python pregenerate_complexes_env_holdout.py --limit 200          # smoke test
"""

import argparse
import os
import pickle
import sys
import time
import warnings

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, 'data_curation')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(os.path.join(start_dir, '..', '..', '..'))
        cur = parent


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
for _root, _dirs, _ in os.walk(REPO_ROOT):
    if 'qtnet' in _dirs:
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break

from qtnet.jax_models.models_molecular_geo import ATOM_FEATURE_KEYS
from qtnet.jax_models.representations import row_to_complex

ALL_ELEMENTS = ['H', 'C', 'N', 'O']
MOLECULAR_DIR = os.path.join(REPO_ROOT, 'data_curation', 'molecular')

POPULATIONS = {
    'aimel': ('aimel_clustered_molecular.pkl', 'precomputed_geo_aimel.pkl'),
    'remainder': ('qm9_inferred.pkl', 'precomputed_geo_qm9_inferred.pkl'),
}


def build_complexes(df: pd.DataFrame, element_to_idx: dict, cutoff: float,
                    max_neighbors: int) -> dict:
    """{qm9_index: Complex}; raises if any molecule fails (a silent gap would
    shrink a test set without anyone noticing)."""
    missing = [k for k in ATOM_FEATURE_KEYS if k not in df.columns]
    if missing:
        raise KeyError(f"frame lacks atomic feature columns {missing}")
    complexes = {}
    t0 = time.perf_counter()
    for n, (_, row) in enumerate(df.iterrows()):
        qid = int(row['index'])
        complexes[qid] = row_to_complex(
            row, element_to_idx, cutoff=cutoff, max_neighbors=max_neighbors,
            include_atom_features=True, max_dim=1,
        )
        if (n + 1) % 5000 == 0:
            print(f"  {n + 1:,}/{len(df):,} ({time.perf_counter() - t0:.0f}s)", flush=True)
    if len(complexes) != len(df):
        raise ValueError(f"{len(df) - len(complexes)} duplicate QM9 ids in the frame")
    print(f"  built {len(complexes):,} complexes in {time.perf_counter() - t0:.0f}s")
    return complexes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--populations', nargs='+', default=list(POPULATIONS),
                        choices=list(POPULATIONS))
    parser.add_argument('--input-dir', default=MOLECULAR_DIR)
    parser.add_argument('--output-dir', default=MOLECULAR_DIR)
    parser.add_argument('--cutoff', type=float, default=3.5, help='edge cutoff (Bohr)')
    parser.add_argument('--max-neighbors', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None, help='first N molecules (smoke test)')
    args = parser.parse_args()

    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}
    os.makedirs(args.output_dir, exist_ok=True)
    for pop in args.populations:
        src, dst = POPULATIONS[pop]
        if args.limit:
            dst = dst.replace('.pkl', f'_limit{args.limit}.pkl')
        print(f"[{pop}] {src} → {dst}  (cutoff {args.cutoff} Bohr, "
              f"max_neighbors {args.max_neighbors}, max_dim 1)", flush=True)
        df = pd.read_pickle(os.path.join(args.input_dir, src))
        if args.limit:
            df = df.iloc[:args.limit]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            complexes = build_complexes(df, element_to_idx, args.cutoff, args.max_neighbors)
        out = os.path.join(args.output_dir, dst)
        with open(out, 'wb') as f:
            pickle.dump({
                'complexes': complexes,
                'element_to_idx': element_to_idx,
                'use_atom_features': True,
                'atom_feature_keys': list(ATOM_FEATURE_KEYS),
                'cutoff': args.cutoff,
                'max_neighbors': args.max_neighbors,
                'max_dim': 1,
                'keyed_by': 'qm9_index',
                'source': src,
            }, f, protocol=4)
        print(f"  wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB)", flush=True)


if __name__ == '__main__':
    main()
