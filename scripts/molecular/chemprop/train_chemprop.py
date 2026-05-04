#!/usr/bin/env python3
"""
ChemProp molecular property prediction — fold × fraction training script.

Trains one ChemPropPredictor at a given CV fold × training fraction.
Loads hyperparameters from an Optuna JSON produced by run_hpo_chemprop.py,
applies create_cv_splits to obtain fold-specific train / val / test indices,
and writes un-normalised predictions on val and test.

Target properties: alpha, gap, U0, Cv

Usage (sbatch array):
    python train_chemprop.py --fold 0  --fraction 1.0
    python train_chemprop.py --fold 12 --fraction 0.5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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
for _root, _dirs, _ in os.walk(REPO_ROOT):
    if 'qtnet' in _dirs:
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break

_proj_candidate = os.path.dirname(REPO_ROOT)
if (os.path.isdir(os.path.join(_proj_candidate, 'experiments'))
        or os.path.exists(os.path.join(_proj_candidate, 'pyproject.toml'))):
    PROJECT_ROOT = _proj_candidate
else:
    PROJECT_ROOT = REPO_ROOT

from qtnet.data_utils import create_cv_splits, MOLECULAR_PROPERTIES_PRED
from qtnet.chemprop_models.train_utils import fit_chemprop
from qtnet.chemprop_models.models import ChemPropPredictor

MOLECULAR_PROPERTIES = MOLECULAR_PROPERTIES_PRED   # ['alpha', 'gap', 'U0', 'Cv']


# ---------------------------------------------------------------------------
# Hyperparameter loading
# ---------------------------------------------------------------------------

def load_optuna_config(optuna_file: str) -> Dict:
    """Return best-trial params from the HPO JSON produced by run_hpo_chemprop.py.

    The file is a list sorted by value (ascending, i.e. most-negative R²
    first), where each record has a 'params' dict containing all architecture
    and optimiser hyperparameters.
    """
    with open(optuna_file, 'r') as f:
        trials = json.load(f)
    return dict(trials[0]['params'])   # best trial is first


# ---------------------------------------------------------------------------
# Split helper (mirrors train_molecular_jax.py)
# ---------------------------------------------------------------------------

def get_fold_fraction_split(
    df: pd.DataFrame,
    fold: int,
    fraction: float,
    n_splits: int,
    n_repeats: int,
    group_col: str,
    base_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) for the given fold × fraction."""
    gcol = None if (group_col is None or group_col.lower() == 'none') \
           else group_col
    for split_info in create_cv_splits(
        df,
        n_splits=n_splits,
        n_repeats=n_repeats,
        group_col=gcol,
        base_seed=base_seed,
        training_fractions=[fraction],
        val_fraction=0.1,
    ):
        if split_info['fold'] == fold:
            return (
                split_info['train_idx'],
                split_info['val_idx'],
                split_info['test_idx'],
            )
    raise ValueError(
        f"Fold {fold} not found "
        f"(n_splits × n_repeats = {n_splits * n_repeats})"
    )


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

def collect_predictions(
    predictor: ChemPropPredictor,
    smiles_list: List[str],
    df_orig: pd.DataFrame,
) -> pd.DataFrame:
    """Run predictor on smiles_list and return a DataFrame of predictions.

    Predictions are already in the original (un-normalised) target scale
    because ChemPropPredictor.predict() applies the stored UnscaleTransform.

    Returns a DataFrame indexed by df_orig.index with columns
    pred_{prop} and target_{prop} for each target property.
    """
    preds = predictor.predict(smiles_list)   # shape: (n_samples, n_targets)

    records = []
    for i, idx in enumerate(df_orig.index):
        if i >= len(preds):
            break
        rec = {'df_index': idx}
        for ci, prop in enumerate(MOLECULAR_PROPERTIES):
            rec[f'pred_{prop}']   = float(preds[i, ci])
            rec[f'target_{prop}'] = float(df_orig.loc[idx, prop])
        records.append(rec)

    result = pd.DataFrame(records).set_index('df_index')
    result.index.name = None
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ChemProp at fold × fraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (sbatch array):
  python train_chemprop.py --fold 0  --fraction 1.0
  python train_chemprop.py --fold 12 --fraction 0.2
        """,
    )
    # Required job-array args
    parser.add_argument('--fold',     type=int,   required=True,
                        help='CV fold index (0 to n_splits * n_repeats - 1)')
    parser.add_argument('--fraction', type=float, required=True,
                        help='Training fraction in (0, 1]')
    # Data path
    parser.add_argument('--pkl-file', type=str,
        default=os.path.join(REPO_ROOT, 'data_curation', 'molecular',
                             'aimel_clustered_molecular.pkl'))
    # Optuna config
    parser.add_argument('--optuna-dir', type=str,
        default=os.path.join(REPO_ROOT, 'optimal_hyperparams'))
    parser.add_argument('--optuna-file', type=str, default=None,
        help='Override full path to optuna JSON '
             '(default: <optuna-dir>/ChemProp_blind_optuna.json)')
    # Output
    parser.add_argument('--output-dir', type=str,
        default=os.path.join(PROJECT_ROOT, 'experiments', 'molecular', 'chemprop'))
    # CV configuration
    parser.add_argument('--n-splits',  type=int, default=5)
    parser.add_argument('--n-repeats', type=int, default=5)
    parser.add_argument('--group-col', type=str, default='Murcko_Scaffold',
                        help="Column for grouped KFold; 'none' for plain KFold")
    parser.add_argument('--base-seed', type=int, default=42)
    # Training
    parser.add_argument('--batch-size',    type=int, default=64)
    parser.add_argument('--max-epochs',    type=int, default=300)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--accelerator',   type=str, default='auto')

    args = parser.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise ValueError(f"--fraction must be in (0, 1]; got {args.fraction}")

    # ------------------------------------------------------------------
    # Optuna hyperparameters
    # ------------------------------------------------------------------
    optuna_file = args.optuna_file or os.path.join(
        args.optuna_dir, 'ChemProp_blind_optuna.json',
    )
    best_params = load_optuna_config(optuna_file)

    # ------------------------------------------------------------------
    # Load DataFrame
    # ------------------------------------------------------------------
    print('=' * 80)
    print(f"ChemProp (blind)  fold={args.fold}  fraction={args.fraction}")
    print('=' * 80)

    print(f"\nLoading DataFrame: {args.pkl_file} ...")
    df = pd.read_pickle(args.pkl_file)
    print(f"  {len(df)} molecules")

    missing = [c for c in ['smiles'] + MOLECULAR_PROPERTIES if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # ------------------------------------------------------------------
    # CV split at (fold, fraction)
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = get_fold_fraction_split(
        df, args.fold, args.fraction,
        n_splits=args.n_splits, n_repeats=args.n_repeats,
        group_col=args.group_col, base_seed=args.base_seed,
    )

    df_train = df.iloc[train_idx].copy()
    df_val   = df.iloc[val_idx].copy()
    df_test  = df.iloc[test_idx].copy()
    print(f"\nSplit: {len(df_train)} train | {len(df_val)} val | "
          f"{len(df_test)} test")

    # ------------------------------------------------------------------
    # Output directory: <output>/<variant>/fold_N/frac_M/
    # ------------------------------------------------------------------
    frac_tag = f"frac_{args.fraction}"
    run_dir = os.path.join(
        args.output_dir, 'blind', f"fold_{args.fold}", frac_tag,
    )
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    fixed_kwargs = {
        'max_epochs':    args.max_epochs,
        'warmup_epochs': args.warmup_epochs,
        'batch_size':    args.batch_size,
        'accelerator':   args.accelerator,
    }
    all_kwargs = {**best_params, **fixed_kwargs}

    print(f"\nBest params: {best_params}")
    print(f"Fixed kwargs: {fixed_kwargs}")
    print(f"\nTraining for {args.max_epochs} epochs ...")

    t0 = time.perf_counter()
    predictor = fit_chemprop(
        train_df=df_train,
        val_df=df_val,
        smiles_col='smiles',
        target_cols=MOLECULAR_PROPERTIES,
        **all_kwargs,
    )
    print(f"  Training complete ({time.perf_counter() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model_path = os.path.join(checkpoint_dir, 'model')
    predictor.save(model_path)
    print(f"  Model saved to {model_path}_*.pt")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config = {
        'model_class':    'ChemProp',
        'variant':        'blind',
        'fold':           args.fold,
        'fraction':       args.fraction,
        'n_splits':       args.n_splits,
        'n_repeats':      args.n_repeats,
        'group_col':      args.group_col,
        'base_seed':      args.base_seed,
        'max_epochs':     args.max_epochs,
        'batch_size':     args.batch_size,
        'warmup_epochs':  args.warmup_epochs,
        'best_params':    best_params,
        'n_train':        len(df_train),
        'n_val':          len(df_val),
        'n_test':         len(df_test),
        'target_columns': list(MOLECULAR_PROPERTIES),
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # Predictions on val + test
    # ------------------------------------------------------------------
    print("Generating val predictions ...")
    val_pred_df = collect_predictions(
        predictor, df_val['smiles'].tolist(), df_val,
    )
    val_pred_df['fold']     = args.fold
    val_pred_df['fraction'] = args.fraction
    val_pred_df['variant']  = 'blind'
    val_pred_df.to_pickle(os.path.join(run_dir, 'val_preds.pkl'))
    print(f"  Saved {len(val_pred_df)} val predictions")

    print("Generating test predictions ...")
    test_pred_df = collect_predictions(
        predictor, df_test['smiles'].tolist(), df_test,
    )
    test_pred_df['fold']     = args.fold
    test_pred_df['fraction'] = args.fraction
    test_pred_df['variant']  = 'blind'
    test_pred_df.to_pickle(os.path.join(run_dir, 'test_preds.pkl'))
    print(f"  Saved {len(test_pred_df)} test predictions")

    print('\n' + '=' * 80)
    print(f"DONE: ChemProp (blind)  fold={args.fold}  fraction={args.fraction}")
    print(f"Output: {run_dir}/")
    print('=' * 80)


if __name__ == '__main__':
    main()
