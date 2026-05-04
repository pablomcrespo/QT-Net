#!/usr/bin/env python3
"""
Precompute molecular complexes from the AIMEl DataFrame.

Builds Complex objects (BCP/RCP topology, no 3D geometry) for every row of the
molecular .pkl and saves them as pickled dicts — no CV splits, no padding,
no z-score normalisation.  Downstream training scripts load these files and
apply splits / batching on the fly.

Output files (under --output-dir, default ``data_curation/molecular/``):
    precomputed_blind.pkl  — variant without per-atom N, LI, Mu, Q features
    precomputed_gta.pkl    — variant with    per-atom N, LI, Mu, Q features
                             ("ground-truth atomic" / informed variant)

Each file contains::
    {'complexes': {df_index: Complex},
     'element_to_idx': {'H': 0, 'C': 1, 'N': 2, 'O': 3},
     'use_atom_features': bool}

Usage:
    python pregenerate_batches_molecular.py
    python pregenerate_batches_molecular.py --variants blind
    python pregenerate_batches_molecular.py --pkl-file path/to/custom.pkl
"""

import argparse
import os
import pickle
import sys
import time
import warnings

import pandas as pd

# ---------------------------------------------------------------------------
# Locate repository root and ensure qtnet is importable
# ---------------------------------------------------------------------------
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
for root, dirs, _ in os.walk(REPO_ROOT):
    if 'qtnet' in dirs:
        if root not in sys.path:
            sys.path.insert(0, root)
        break

from qtnet.jax_models.representations import row_to_complex

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_ELEMENTS = ['H', 'C', 'N', 'O']   # QM9 / AIMEl vocabulary

VARIANT_CONFIG = {
    'blind': {'filename': 'precomputed_blind.pkl', 'use_atom_features': False},
    'gta':   {'filename': 'precomputed_gta.pkl',   'use_atom_features': True},
}


def build_all_complexes(df: pd.DataFrame, element_to_idx: dict,
                        use_atom_features: bool) -> dict:
    """Build {df_index: Complex} for every row; skip and warn on errors."""
    complexes, errors = {}, []
    t0 = time.perf_counter()
    for idx, row in df.iterrows():
        try:
            complexes[idx] = row_to_complex(
                row, element_to_idx, cutoff = 3.5, max_neighbors = 4,
                include_atom_features=use_atom_features,
            )
        except Exception as exc:
            warnings.warn(f"Row {idx}: skipping ({exc})")
            errors.append(idx)
    elapsed = time.perf_counter() - t0
    print(f"  Built {len(complexes)} complexes in {elapsed:.1f}s "
          f"({len(errors)} errors skipped)")
    return complexes


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pkl-file", type=str,
        default=os.path.join(REPO_ROOT, 'data_curation', 'molecular',
                             'aimel_clustered_molecular.pkl'),
        help="Path to the molecular DataFrame pickle",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(REPO_ROOT, 'data_curation', 'molecular'),
        help="Directory for the output pickle files",
    )
    parser.add_argument(
        "--variants", nargs='+', default=['blind', 'gta'],
        choices=list(VARIANT_CONFIG.keys()),
        help="Which variants to build (default: both)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Truncate to first N rows (smoke tests)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Molecular complex precomputation")
    print(f"  Input:      {args.pkl_file}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Variants:   {args.variants}")
    print("=" * 72)

    print(f"\nLoading {args.pkl_file} ...")
    t0 = time.perf_counter()
    df = pd.read_pickle(args.pkl_file)
    if args.limit is not None:
        df = df.iloc[:args.limit].copy()
        print(f"  Limited to {args.limit} molecules")
    print(f"  {len(df)} molecules ({time.perf_counter() - t0:.1f}s)")

    #required_cols = ['a_name', 'BCP_connectivity', 'RCP_connectivity']
    #missing = [c for c in required_cols if c not in df.columns]
    #if missing:
    #    raise ValueError(f"DataFrame missing required columns: {missing}")

    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}
    os.makedirs(args.output_dir, exist_ok=True)

    for variant in args.variants:
        cfg = VARIANT_CONFIG[variant]
        use_af = cfg['use_atom_features']

        print(f"\n[{variant}] building complexes (use_atom_features={use_af}) ...")
        complexes = build_all_complexes(df, element_to_idx, use_af)

        out_path = os.path.join(args.output_dir, cfg['filename'])
        print(f"  Saving → {out_path}")
        with open(out_path, 'wb') as f:
            pickle.dump({
                'complexes':         complexes,
                'element_to_idx':    element_to_idx,
                'use_atom_features': use_af,
            }, f, protocol=4)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Wrote {len(complexes)} complexes ({size_mb:.1f} MB)")

    print("\n" + "=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
