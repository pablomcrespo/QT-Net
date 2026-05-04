#!/usr/bin/env python3
"""
Precompute cutoff-based molecular complexes for QM9.

Builds Complex objects (distance cutoff topology) for every row of the QM9
DataFrame and saves them as pickled dicts — no CV splits, no padding, no
z-score normalisation.

Mirrors pregenerate_batches_molecular.py (which does the same for
aimel_clustered_molecular.pkl) but for qm9_filtered.pkl / qm9_inferred.pkl.

Output files (under --output-dir, default data_curation/molecular/):
    precomputed_blind_qm9.pkl  — no per-atom N, LI, Mu, Q features
    precomputed_gta_qm9.pkl    — with inferred per-atom N, LI, Mu, Q features
                                 (requires qm9_inferred.pkl from infer_QTAIM_QM9.py)

Each file contains:
    {'complexes': {df_index: Complex},
     'element_to_idx': {'H': 0, 'C': 1, 'N': 2, 'O': 3},
     'use_atom_features': bool}

Usage:
    python precompute_molecular_QM9.py
    python precompute_molecular_QM9.py --variants blind
    python precompute_molecular_QM9.py --limit 200   # smoke test
"""

import argparse
import os
import pickle
import sys
import time
import warnings

import pandas as pd

# ---------------------------------------------------------------------------
# Repo root discovery (mirrors pregenerate_batches_molecular.py)
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

from qtnet.jax_models.representations import row_to_complex

# ---------------------------------------------------------------------------
# Constants — must match aimel training convention
# ---------------------------------------------------------------------------
ALL_ELEMENTS = ['H', 'C', 'N', 'O']

# Connectivity settings matching pregenerate_batches_molecular.py
DEFAULT_CUTOFF = 3.5
DEFAULT_MAX_NEIGHBORS = 4

VARIANT_CONFIG = {
    'blind': {
        'filename':           'precomputed_blind_qm9.pkl',
        'use_atom_features':  False,
        'pkl_key':            'qm9_filtered_pkl',
    },
    'gta': {
        'filename':           'precomputed_gta_qm9.pkl',
        'use_atom_features':  True,
        'pkl_key':            'qm9_inferred_pkl',
    },
}


def build_all_complexes(
    df: pd.DataFrame,
    element_to_idx: dict,
    use_atom_features: bool,
    cutoff: float,
    max_neighbors: int,
) -> dict:
    """Build {df_index: Complex} for every row; skip and warn on errors."""
    complexes, errors = {}, []
    t0 = time.perf_counter()
    for idx, row in df.iterrows():
        try:
            complexes[idx] = row_to_complex(
                row,
                element_to_idx,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
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
        '--qm9-filtered-pkl', type=str,
        default=os.path.join(
            REPO_ROOT, 'data_curation', 'molecular', 'qm9_filtered.pkl'),
        help='Path to qm9_filtered.pkl (used for blind variant)',
    )
    parser.add_argument(
        '--qm9-inferred-pkl', type=str,
        default=os.path.join(
            REPO_ROOT, 'data_curation', 'molecular', 'qm9_inferred.pkl'),
        help='Path to qm9_inferred.pkl (used for gta variant; '
             'produced by infer_QTAIM_QM9.py)',
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(REPO_ROOT, 'data_curation', 'molecular'),
        help='Directory for the output pickle files',
    )
    parser.add_argument(
        '--variants', nargs='+', default=['blind', 'gta'],
        choices=list(VARIANT_CONFIG.keys()),
        help='Which variants to build (default: both)',
    )
    parser.add_argument(
        '--cutoff', type=float, default=DEFAULT_CUTOFF,
        help=f'Distance cutoff in Bohr (default {DEFAULT_CUTOFF}; '
             'must match aimel training setting)',
    )
    parser.add_argument(
        '--max-neighbors', type=int, default=DEFAULT_MAX_NEIGHBORS,
        help=f'Max neighbours per atom (default {DEFAULT_MAX_NEIGHBORS}; '
             'must match aimel training setting)',
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Truncate to first N rows (smoke tests)',
    )
    args = parser.parse_args()

    pkl_paths = {
        'qm9_filtered_pkl': args.qm9_filtered_pkl,
        'qm9_inferred_pkl': args.qm9_inferred_pkl,
    }

    print("=" * 72)
    print("QM9 molecular complex precomputation")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Variants:      {args.variants}")
    print(f"  cutoff:        {args.cutoff} Bohr")
    print(f"  max_neighbors: {args.max_neighbors}")
    print("=" * 72)

    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}
    os.makedirs(args.output_dir, exist_ok=True)

    for variant in args.variants:
        cfg = VARIANT_CONFIG[variant]
        use_af = cfg['use_atom_features']
        src_pkl = pkl_paths[cfg['pkl_key']]

        print(f"\n[{variant}] loading {src_pkl} ...")
        if not os.path.exists(src_pkl):
            raise FileNotFoundError(
                f"Input file not found for variant '{variant}': {src_pkl}\n"
                + ("Run infer_QTAIM_QM9.py first to produce qm9_inferred.pkl."
                   if variant == 'gta' else "")
            )

        t0 = time.perf_counter()
        df = pd.read_pickle(src_pkl)
        if args.limit is not None:
            df = df.iloc[:args.limit].copy()
            print(f"  Limited to {args.limit} molecules")
        print(f"  {len(df)} molecules ({time.perf_counter() - t0:.1f}s)")

        print(f"[{variant}] building complexes (use_atom_features={use_af}) ...")
        complexes = build_all_complexes(
            df, element_to_idx, use_af,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
        )

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


if __name__ == '__main__':
    main()
