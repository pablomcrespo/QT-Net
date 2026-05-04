#!/usr/bin/env python3
"""
Run molecular property prediction on QM9 using trained ScalarGNNMolecular ensembles.

For each (variant, fraction) combination, picks the 5 best folds (one per repeat,
chosen by lowest best_val_so_far in loss_history.json) from experiments/molecular/,
runs inference on qm9_inferred.pkl, and writes ensemble-averaged predictions.

Prerequisites:
  1. scripts/inference/infer_QTAIM_QM9.py        → qm9_inferred.pkl
  2. scripts/inference/precompute_molecular_QM9.py → precomputed_blind_qm9.pkl
                                                   → precomputed_gta_qm9.pkl

Output (qm9_molecular_preds.pkl) has one row per QM9 molecule with columns:
  {variant}_{fraction}_pred_{prop}   — ensemble-mean prediction (physical units)
  {variant}_{fraction}_std_{prop}    — ensemble std across 5 folds
  target_{prop}                      — ground-truth from qm9_inferred.pkl

Usage:
  python predict_from_inferred.py --variants informed blind --fractions 1.0
  python predict_from_inferred.py --variants blind --fractions 0.1 0.5 1.0
"""

import argparse
import functools
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from flax import nnx

# ---------------------------------------------------------------------------
# Repo / project root
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
from qtnet.jax_models.models_molecular import ScalarGNNMolecular
from qtnet.jax_models.representations import prepare_padded_batches
from qtnet.jax_models.train_utils import load_checkpoint
from qtnet.data_utils import MOLECULAR_PROPERTIES_PRED

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOLECULAR_PROPERTIES = MOLECULAR_PROPERTIES_PRED   # ['alpha', 'gap', 'U0', 'Cv']
NUM_SPECIES = 4   # H, C, N, O


# ---------------------------------------------------------------------------
# Fold selection
# ---------------------------------------------------------------------------

def select_best_folds(
    experiments_root: str,
    variant: str,
    fraction: float,
    n_repeats: int = 5,
    n_splits: int = 5,
) -> list:
    """Return 5 fold indices with lowest best_val_so_far, one per repeat.

    Groups folds by repeat = fold // n_splits, picks argmin within each repeat.
    """
    n_folds = n_repeats * n_splits
    frac_tag = f"frac_{fraction}"

    best_val = {}
    for fold_idx in range(n_folds):
        hist_path = os.path.join(
            experiments_root, variant,
            f"fold_{fold_idx}", frac_tag, "loss", "loss_history.json",
        )
        if not os.path.exists(hist_path):
            continue
        try:
            with open(hist_path) as f:
                hist = json.load(f)
            val = hist.get('best_val_so_far')
            if val is not None:
                best_val[fold_idx] = float(val)
        except Exception:
            pass

    if not best_val:
        raise FileNotFoundError(
            f"No loss_history.json found for "
            f"{variant}/{frac_tag} under {experiments_root}"
        )

    selected = []
    for repeat in range(n_repeats):
        repeat_folds = {
            fi: v for fi, v in best_val.items()
            if fi // n_splits == repeat
        }
        if not repeat_folds:
            raise RuntimeError(
                f"No valid folds for repeat {repeat} ({variant}/{frac_tag})"
            )
        selected.append(min(repeat_folds, key=repeat_folds.get))

    return selected


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _run_fold_inference(
    model: nnx.Module,
    batches: list,
    norm_stats: dict,
    n_molecules: int,
) -> np.ndarray:
    """Run one fold over all batches, denormalise, return (n_mols, 4)."""

    @functools.partial(nnx.jit, static_argnames=('num_graphs',))
    def _predict(model, batch, num_graphs):
        graph_idx = batch.cochain_batches[0].owner_cochains
        return model(batch, graph_idx=graph_idx, num_graphs=num_graphs)

    raw_chunks = []
    for batch in batches:
        n_mols = int(batch.num_complexes[0])
        out = _predict(model, batch, num_graphs=n_mols)
        raw_chunks.append(np.asarray(out['predictions'])[:n_mols])

    preds_norm = (
        np.concatenate(raw_chunks, axis=0) if raw_chunks
        else np.zeros((0, len(MOLECULAR_PROPERTIES)), dtype=np.float32)
    )

    preds_unreg = np.zeros_like(preds_norm)
    for i, prop in enumerate(MOLECULAR_PROPERTIES):
        preds_unreg[:, i] = (
            preds_norm[:, i] * norm_stats[prop]['std']
            + norm_stats[prop]['mean']
        )

    return preds_unreg[:n_molecules]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run ScalarGNNMolecular ensemble inference on QM9.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_from_inferred.py --variants informed blind --fractions 1.0
  python predict_from_inferred.py --variants blind --fractions 0.1 0.5 1.0
        """,
    )
    parser.add_argument(
        '--inferred-pkl', type=str,
        default=os.path.join(
            REPO_ROOT, 'data_curation', 'molecular', 'qm9_inferred.pkl'),
        help='Output of infer_QTAIM_QM9.py',
    )
    parser.add_argument(
        '--blind-complexes', type=str,
        default=os.path.join(
            REPO_ROOT, 'data_curation', 'molecular',
            'precomputed_blind_qm9.pkl'),
        help='Blind complexes from precompute_molecular_QM9.py',
    )
    parser.add_argument(
        '--gta-complexes', type=str,
        default=os.path.join(
            REPO_ROOT, 'data_curation', 'molecular',
            'precomputed_gta_qm9.pkl'),
        help='GTA (informed) complexes from precompute_molecular_QM9.py',
    )
    parser.add_argument(
        '--variants', nargs='+', default=['informed', 'blind'],
        choices=['informed', 'blind'],
    )
    parser.add_argument(
        '--fractions', nargs='+', type=float, default=[1.0],
        help='Training fractions (e.g. 0.1 0.5 1.0)',
    )
    parser.add_argument(
        '--molecular-experiments-root', type=str,
        default=os.path.join(PROJECT_ROOT, 'experiments', 'molecular'),
    )
    parser.add_argument(
        '--output-pkl', type=str,
        default=os.path.join(
            REPO_ROOT, 'data_curation', 'molecular',
            'qm9_molecular_preds.pkl'),
    )
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--no-std', action='store_true',
                        help='Skip writing ensemble std columns')

    args = parser.parse_args()

    print("=" * 72)
    print("QM9 molecular property prediction from inferred AIM features")
    print(f"  Variants:  {args.variants}")
    print(f"  Fractions: {args.fractions}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Load QM9 inferred frame
    # ------------------------------------------------------------------
    print(f"\nLoading {args.inferred_pkl} ...")
    qm9 = pd.read_pickle(args.inferred_pkl)
    print(f"  {len(qm9)} molecules")

    # ------------------------------------------------------------------
    # Load complex caches (only what's needed)
    # ------------------------------------------------------------------
    cpx_cache = {}

    def _load_complexes(path: str, label: str) -> dict:
        print(f"\nLoading {label} complexes: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Complex file not found: {path}\n"
                "Run precompute_molecular_QM9.py first."
            )
        with open(path, 'rb') as f:
            data = pickle.load(f)
        cpx = data['complexes']
        print(f"  {len(cpx)} complexes")
        return cpx

    if 'blind' in args.variants:
        cpx_cache['blind'] = _load_complexes(args.blind_complexes, 'blind')
    if 'informed' in args.variants:
        cpx_cache['gta'] = _load_complexes(args.gta_complexes, 'gta')

    # ------------------------------------------------------------------
    # Start output DataFrame
    # ------------------------------------------------------------------
    out = qm9.copy()

    # ------------------------------------------------------------------
    # Run each (variant, fraction) combination
    # ------------------------------------------------------------------
    for variant in args.variants:
        cpx_key = 'gta' if variant == 'informed' else 'blind'
        complexes = cpx_cache[cpx_key]

        # Restrict QM9 to rows that have a valid complex
        valid_idx = [i for i in qm9.index if i in complexes]
        if len(valid_idx) < len(qm9):
            print(f"\n  [{variant}] Restricting to {len(valid_idx)} / {len(qm9)} "
                  "molecules with valid complexes")
        qm9_sub = qm9.loc[valid_idx]
        n_molecules = len(qm9_sub)

        # Batches are built once per variant, shared across fractions
        print(f"\n[{variant}] Building padded batches ({n_molecules} molecules)...")
        batches = prepare_padded_batches(
            complexes, qm9_sub,
            target_columns=[],
            batch_size=args.batch_size,
            verbose=True,
            as_numpy=True,
        )
        print(f"  {len(batches)} batches")

        for fraction in args.fractions:
            tag = f"{variant}_{fraction}"
            print(f"\n[{tag}] Selecting best folds ...")

            try:
                fold_indices = select_best_folds(
                    args.molecular_experiments_root, variant, fraction,
                )
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"  WARNING: skipping {tag} — {exc}")
                continue

            print(f"  Best folds: {fold_indices}")

            fold_preds = []
            for fold_idx in fold_indices:
                frac_tag = f"frac_{fraction}"
                run_dir = os.path.join(
                    args.molecular_experiments_root,
                    variant, f"fold_{fold_idx}", frac_tag,
                )
                cfg_path  = os.path.join(run_dir, 'config.json')
                best_ckpt = os.path.join(run_dir, 'checkpoints', 'model_best_so_far')

                if not os.path.exists(cfg_path):
                    print(f"  WARNING: config.json missing for fold {fold_idx}, skipping")
                    continue
                if not os.path.exists(best_ckpt):
                    print(f"  WARNING: checkpoint missing for fold {fold_idx}, skipping")
                    continue

                with open(cfg_path) as f:
                    cfg = json.load(f)

                print(f"  fold {fold_idx}: loading {cfg['model_class']} checkpoint ...")
                model = ScalarGNNMolecular(
                    num_species=NUM_SPECIES,
                    num_outputs=len(MOLECULAR_PROPERTIES),
                    use_atom_features=cfg['use_atom_features'],
                    rngs=nnx.Rngs(cfg['seed']),
                    **cfg['model_kwargs'],
                )
                load_checkpoint(model, best_ckpt)

                preds = _run_fold_inference(
                    model, batches, cfg['norm_stats'], n_molecules,
                )
                fold_preds.append(preds)
                print(f"    shape {preds.shape}, "
                      f"alpha mean={preds[:,0].mean():.3f}")

            if not fold_preds:
                print(f"  WARNING: no valid fold predictions for {tag}, skipping")
                continue

            # Average across folds
            stack = np.stack(fold_preds, axis=0)   # (n_folds, n_mols, 4)
            mean_preds = stack.mean(axis=0)
            std_preds  = stack.std(axis=0)

            for i, prop in enumerate(MOLECULAR_PROPERTIES):
                out.loc[valid_idx, f"{tag}_pred_{prop}"] = mean_preds[:, i]
                if not args.no_std:
                    out.loc[valid_idx, f"{tag}_std_{prop}"] = std_preds[:, i]

            print(f"  [{tag}] wrote {len(fold_preds)} folds × {n_molecules} molecules")

    # ------------------------------------------------------------------
    # Ground-truth target columns
    # ------------------------------------------------------------------
    for prop in MOLECULAR_PROPERTIES:
        if prop in qm9.columns:
            out[f'target_{prop}'] = qm9[prop]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output_pkl)), exist_ok=True)
    out.to_pickle(args.output_pkl)

    pred_cols = [c for c in out.columns if '_pred_' in c]
    print(f"\nSaved {len(out)} molecules → {args.output_pkl}")
    print(f"  {len(pred_cols)} prediction columns: {pred_cols}")
    print("=" * 72)
    print("Done.")


if __name__ == '__main__':
    main()
