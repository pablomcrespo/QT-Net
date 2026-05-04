#!/usr/bin/env python3
"""Precompute Complex representations and save to .pkl files.

Usage:
    python precompute_complexes.py --cutoff 8.0 --max-neighbors 12
    python precompute_complexes.py --fully-connected
"""
import os, sys, argparse, pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_repo_root(start_dir):
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
for root, dirs, _ in os.walk(REPO_ROOT):
    if 'qtnet' in dirs:
        if root not in sys.path:
            sys.path.insert(0, root)
        break

import pandas as pd
from qtnet.jax_models.representations import precompute_complexes

ALL_ELEMENTS = ['H', 'C', 'N', 'O']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=float, default=5.25)
    parser.add_argument("--max-neighbors", type=int, default=5)
    parser.add_argument("--fully-connected", action="store_true")
    parser.add_argument("--max-dim", type=int, default=2,
                        help="Maximum cochain dimension (1=skip bags-of-bonds)")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(REPO_ROOT, 'data_curation', 'atomic'))
    args = parser.parse_args()

    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}

    if args.fully_connected:
        suffix = "fc"
    else:
        suffix = f"cutoff{args.cutoff}_max{args.max_neighbors}"

    for name, pkl_name in [("train_and_val", f"precomputed_complexes_{suffix}.pkl"),
                           ("test",          f"precomputed_test_{suffix}.pkl")]:
        src = os.path.join(args.data_dir, f"{name}.pkl")
        dst = os.path.join(args.data_dir, pkl_name)
        if os.path.exists(dst):
            print(f"Already exists: {dst} — skipping")
            continue
        print(f"\nLoading {src} ...")
        df = pd.read_pickle(src)
        print(f"Precomputing {len(df)} complexes → {dst}")
        precompute_complexes(
            df, element_to_idx=element_to_idx,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors if not args.fully_connected else None,
            fully_connected=args.fully_connected,
            max_dim=args.max_dim,
            output_path=dst, verbose=True,
        )

    print("\nDone.")

if __name__ == "__main__":
    main()
