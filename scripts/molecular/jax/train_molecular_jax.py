#!/usr/bin/env python3
"""
Molecular Property Prediction Training Script  (ScalarTPaiNNMolecular)

Trains one model at a given CV fold × training fraction.  Loads precomputed
complexes (built by pregenerate_batches_molecular.py), applies
create_cv_splits to obtain fold-specific train / val / test indices, computes
z-score normalisation from the training set, builds padded batches, trains
with ReduceOnPlateau, and writes unregularized predictions on val and test.

Target properties: alpha, gap, U0, Cv

Hyperparameters are read from:
    <optuna-dir>/ScalarTPaiNNMolecular_blind_optuna.json

for both blind and informed variants (override with --optuna-file).

Usage (sbatch array):
    python train_molecular_jax.py --fold 0  --fraction 1.0
    python train_molecular_jax.py --fold 12 --fraction 0.5 --blind
"""

import argparse
import functools
import json
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import flax.nnx as nnx

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
from qtnet.jax_models.representations import prepare_padded_batches
from qtnet.jax_models.models_molecular import ScalarGNNMolecular
from qtnet.jax_models.train_utils import (
    train_molecular_multitask,
    count_parameters,
    load_checkpoint,
    make_optimizer,
)

MOLECULAR_PROPERTIES = MOLECULAR_PROPERTIES_PRED   # ['alpha', 'gap', 'U0', 'Cv']
ALL_ELEMENTS = ['H', 'C', 'N', 'O']


# ---------------------------------------------------------------------------
# Hyperparameter loading (mirrors atomic load_optuna_config)
# ---------------------------------------------------------------------------

def load_optuna_config(optuna_file: str) -> Tuple[Dict, float, float]:
    """Return (model_kwargs, lr, weight_decay) from the best trial in the JSON.

    The file is the output of run_hpo_molecular.py: a list sorted by value
    (ascending), where each record has a 'params' dict containing both
    architecture kwargs and 'lr' / 'weight_decay'.
    """
    with open(optuna_file, 'r') as f:
        trials = json.load(f)
    params = dict(trials[0]['params'])   # best trial is first
    lr           = params.pop('lr')
    weight_decay = params.pop('weight_decay', 0.0)
    return params, lr, weight_decay


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def create_model(model_kwargs: Dict, seed: int,
                 use_atom_features: bool, num_species: int) -> nnx.Module:
    return ScalarGNNMolecular(
        num_species=num_species,
        num_outputs=len(MOLECULAR_PROPERTIES),
        use_atom_features=use_atom_features,
        rngs=nnx.Rngs(seed),
        **model_kwargs,
    )


# ---------------------------------------------------------------------------
# Split + normalisation helpers
# ---------------------------------------------------------------------------

def get_fold_fraction_split(df, fold, fraction, n_splits, n_repeats,
                            group_col, base_seed):
    """Return (train_idx, val_idx, test_idx) for the given fold × fraction."""
    group_col = None if (group_col is None or group_col.lower() == 'none') \
                else group_col
    for split_info in create_cv_splits(
        df,
        n_splits=n_splits,
        n_repeats=n_repeats,
        group_col=group_col,
        base_seed=base_seed,
        training_fractions=[fraction],
        val_fraction=0.1,
    ):
        if split_info['fold'] == fold:
            return (split_info['train_idx'],
                    split_info['val_idx'],
                    split_info['test_idx'])
    raise ValueError(
        f"Fold {fold} not found "
        f"(n_splits × n_repeats = {n_splits * n_repeats})"
    )


def norm_stats_from_train(df_train: pd.DataFrame) -> Dict:
    stats = {}
    for prop in MOLECULAR_PROPERTIES:
        vals = df_train[prop].values.astype(np.float64)
        std = float(vals.std())
        if std < 1e-8:
            std = 1.0
        stats[prop] = {'mean': float(vals.mean()), 'std': std}
    return stats


def normalize_targets(df: pd.DataFrame, stats: Dict) -> np.ndarray:
    out = np.zeros((len(df), len(MOLECULAR_PROPERTIES)), dtype=np.float32)
    for i, prop in enumerate(MOLECULAR_PROPERTIES):
        vals = df[prop].values.astype(np.float32)
        out[:, i] = (vals - stats[prop]['mean']) / stats[prop]['std']
    return out


# ---------------------------------------------------------------------------
# Unregularized prediction collection
# ---------------------------------------------------------------------------

def collect_molecular_predictions(model, batches, df_orig,
                                   norm_stats) -> pd.DataFrame:
    """Run model, un-normalise predictions, return DataFrame."""

    @functools.partial(nnx.jit, static_argnames=('num_graphs',))
    def predict(model, batch, num_graphs):
        graph_idx = batch.cochain_batches[0].owner_cochains
        return model(batch, graph_idx=graph_idx, num_graphs=num_graphs)

    all_preds: List[np.ndarray] = []
    for batch in batches:
        n_mols = int(batch.num_complexes[0])
        out    = predict(model, batch, num_graphs=n_mols)
        all_preds.append(np.asarray(out['predictions'])[:n_mols])

    preds_norm = (np.concatenate(all_preds, axis=0) if all_preds
                  else np.zeros((0, len(MOLECULAR_PROPERTIES)), dtype=np.float32))

    preds_unreg = np.zeros_like(preds_norm)
    for i, prop in enumerate(MOLECULAR_PROPERTIES):
        preds_unreg[:, i] = (
            preds_norm[:, i] * norm_stats[prop]['std'] + norm_stats[prop]['mean']
        )

    records = []
    for i, idx in enumerate(df_orig.index):
        if i >= len(preds_unreg):
            break
        rec = {'df_index': idx}
        for ci, prop in enumerate(MOLECULAR_PROPERTIES):
            rec[f'pred_{prop}']   = float(preds_unreg[i, ci])
            rec[f'target_{prop}'] = float(df_orig.loc[idx, prop])
        records.append(rec)

    result = pd.DataFrame(records).set_index('df_index')
    result.index.name = None
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train ScalarGNMolecular at fold × fraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (sbatch array):
  python train_molecular_jax.py --fold 0  --fraction 1.0
  python train_molecular_jax.py --fold 12 --fraction 0.2 --blind
        """,
    )
    # Required job-array args
    parser.add_argument("--fold",     type=int,   required=True,
                        help="CV fold index (0 to n_splits * n_repeats - 1)")
    parser.add_argument("--fraction", type=float, required=True,
                        help="Training fraction in (0, 1]")
    parser.add_argument("--blind", action="store_true",
                        help="Use blind complexes (no atom features). "
                             "Default is informed (gta).")
    # Data / cache paths
    parser.add_argument("--pkl-file", type=str,
        default=os.path.join(REPO_ROOT, 'data_curation', 'molecular',
                             'aimel_clustered_molecular.pkl'))
    parser.add_argument("--complexes-blind", type=str,
        default=os.path.join(REPO_ROOT, 'data_curation', 'molecular',
                             'precomputed_blind.pkl'))
    parser.add_argument("--complexes-gta", type=str,
        default=os.path.join(REPO_ROOT, 'data_curation', 'molecular',
                             'precomputed_gta.pkl'))
    # Optuna config
    parser.add_argument("--optuna-dir", type=str,
        default=os.path.join(REPO_ROOT, 'optimal_hyperparams'))
    parser.add_argument("--optuna-file", type=str, default=None,
        help="Override full path to optuna JSON "
             "(default: <optuna-dir>/ScalarGNNMolecular_blind_optuna.json)")
    # Output
    parser.add_argument("--output-dir", type=str,
        default=os.path.join(PROJECT_ROOT, 'experiments', 'molecular'))
    # CV configuration
    parser.add_argument("--n-splits",  type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--group-col", type=str, default='Murcko_Scaffold',
                        help="Column for grouped KFold; 'none' for plain KFold")
    parser.add_argument("--base-seed", type=int, default=42)
    # Training
    parser.add_argument("--batch-size",  type=int, default=8192)
    parser.add_argument("--epochs",      type=int, default=2000)
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for model initialisation")
    parser.add_argument("--save-every",  type=int, default=250)
    parser.add_argument("--print-every", type=int, default=10)

    args = parser.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise ValueError(f"--fraction must be in (0, 1]; got {args.fraction}")

    use_atom_features = not args.blind
    variant = 'blind' if args.blind else 'informed'
    complexes_path = args.complexes_blind if args.blind else args.complexes_gta

    # ------------------------------------------------------------------
    # Optuna hyperparameters
    # ------------------------------------------------------------------
    optuna_file = args.optuna_file or os.path.join(
        args.optuna_dir, 'ScalarGNNMolecular_blind_optuna.json',
    )
    model_kwargs, lr, wd = load_optuna_config(optuna_file)

    # ------------------------------------------------------------------
    # Load DataFrame + precomputed complexes
    # ------------------------------------------------------------------
    print("=" * 80)
    print(f"ScalarGNNMolecular ({variant})  "
          f"fold={args.fold}  fraction={args.fraction}")
    print("=" * 80)

    print(f"\nLoading DataFrame: {args.pkl_file} ...")
    df = pd.read_pickle(args.pkl_file)
    print(f"  {len(df)} molecules")

    print(f"Loading complexes: {complexes_path} ...")
    with open(complexes_path, 'rb') as f:
        cpx_data = pickle.load(f)
    complexes      = cpx_data['complexes']
    element_to_idx = cpx_data['element_to_idx']
    num_species    = len(element_to_idx)
    print(f"  {len(complexes)} complexes (num_species={num_species})")

    valid = set(complexes.keys())
    df = df.loc[df.index.isin(valid)]
    print(f"  DataFrame restricted to {len(df)} rows with valid complexes")

    # ------------------------------------------------------------------
    # CV split at (fold, fraction)  — val_idx carved by create_cv_splits
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
    # z-score normalisation (fit on training rows only)
    # ------------------------------------------------------------------
    norm_stats    = norm_stats_from_train(df_train)
    train_targets = normalize_targets(df_train, norm_stats)
    val_targets   = normalize_targets(df_val,   norm_stats)

    # ------------------------------------------------------------------
    # Padded batches
    # ------------------------------------------------------------------
    print("\nBuilding padded batches ...")
    train_batches = prepare_padded_batches(
        complexes, df_train, target_columns=[],
        batch_size=args.batch_size, verbose=False, as_numpy=True,
    )
    val_batches = prepare_padded_batches(
        complexes, df_val, target_columns=[],
        batch_size=args.batch_size, verbose=False, as_numpy=True,
    )
    test_batches = prepare_padded_batches(
        complexes, df_test, target_columns=[],
        batch_size=args.batch_size, verbose=False, as_numpy=True,
    )
    print(f"  train={len(train_batches)}  val={len(val_batches)}  "
          f"test={len(test_batches)} batches")

    # ------------------------------------------------------------------
    # Model + optimizer (AdamW + ReduceOnPlateau)
    # ------------------------------------------------------------------
    model = create_model(
        model_kwargs,
        seed=args.seed + args.fold,
        use_atom_features=use_atom_features,
        num_species=num_species,
    )
    n_params = count_parameters(model)
    print(f"\nModel: ScalarTPaiNNMolecular ({variant}) — {n_params:,} params")
    print(f"  lr={lr:.2e}  weight_decay={wd:.2e}  "
          f"model_kwargs={model_kwargs}")

    tx = make_optimizer(
        lr=lr,
        weight_decay=wd,
        use_reduce_on_plateau=True,
        plateau_accumulation_size=len(train_batches),
    )
    optimizer = nnx.Optimizer(model, tx)

    # ------------------------------------------------------------------
    # Output directory: <output>/<variant>/fold_N/frac_M/
    # ------------------------------------------------------------------
    frac_tag = f"frac_{args.fraction}"
    run_dir = os.path.join(
        args.output_dir, variant, f"fold_{args.fold}", frac_tag,
    )
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    loss_dir       = os.path.join(run_dir, 'loss')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    config = {
        'model_class':       'ScalarGNNMolecular',
        'model_kwargs':      model_kwargs,
        'variant':           variant,
        'use_atom_features': use_atom_features,
        'fold':              args.fold,
        'fraction':          args.fraction,
        'n_splits':          args.n_splits,
        'n_repeats':         args.n_repeats,
        'group_col':         args.group_col,
        'base_seed':         args.base_seed,
        'seed':              args.seed + args.fold,
        'epochs':            args.epochs,
        'batch_size':        args.batch_size,
        'learning_rate':     lr,
        'weight_decay':      wd,
        'n_params':          n_params,
        'n_train':           len(df_train),
        'n_val':             len(df_val),
        'n_test':            len(df_test),
        'target_columns':    list(MOLECULAR_PROPERTIES),
        'norm_stats':        norm_stats,
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs ...")
    train_molecular_multitask(
        model=model,
        optimizer=optimizer,
        train_batches=train_batches,
        train_targets=train_targets,
        val_batches=val_batches,
        val_targets=val_targets,
        target_names=list(MOLECULAR_PROPERTIES),
        epochs=args.epochs,
        save_every=args.save_every,
        checkpoint_dir=checkpoint_dir,
        loss_dir=loss_dir,
        verbose=True,
        print_every=args.print_every,
    )

    # ------------------------------------------------------------------
    # Load best model for predictions
    # ------------------------------------------------------------------
    best_path = os.path.join(checkpoint_dir, 'model_best_so_far')
    if os.path.exists(best_path):
        print("\nLoading best model for predictions ...")
        model = load_checkpoint(model, best_path)
    else:
        print("\nNo best-model checkpoint; using final model for predictions.")

    # ------------------------------------------------------------------
    # Unregularized predictions on val + test (fold holdout)
    # ------------------------------------------------------------------
    print("Generating val predictions ...")
    val_pred_df = collect_molecular_predictions(
        model, val_batches, df_val, norm_stats,
    )
    val_pred_df['fold']     = args.fold
    val_pred_df['fraction'] = args.fraction
    val_pred_df['variant']  = variant
    val_pred_df.to_pickle(os.path.join(run_dir, 'val_preds.pkl'))
    print(f"  Saved {len(val_pred_df)} val predictions")

    print("Generating test predictions ...")
    test_pred_df = collect_molecular_predictions(
        model, test_batches, df_test, norm_stats,
    )
    test_pred_df['fold']     = args.fold
    test_pred_df['fraction'] = args.fraction
    test_pred_df['variant']  = variant
    test_pred_df.to_pickle(os.path.join(run_dir, 'test_preds.pkl'))
    print(f"  Saved {len(test_pred_df)} test predictions")

    print("\n" + "=" * 80)
    print(f"DONE: ScalarGNNMolecular ({variant})  "
          f"fold={args.fold}  fraction={args.fraction}")
    print(f"Output: {run_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
