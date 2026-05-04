#!/usr/bin/env python3
"""
Simple command-line front end for the `atomic_optuna.ipynb` workflow.

Usage examples:

    python run_hpo.py --model-name TPaiNN
    python run_hpo.py --model-name ScalarGNN --batch-size 2048
    python run_hpo.py --model-name DeepSets   # baseline search
    python run_hpo.py --model-name ScalarBaseline

The script loads the pre‑curated dataset, optionally precomputes the
`precomputed_complexes.pkl` geometry cache, forms the first cross‑validation
fold, and then fires off an Optuna search for the requested model.  The
behaviour is intentionally kept very close to the notebook so that the same
code path is exercised inside batch jobs.
"""

import os
import argparse
import pickle
import pandas as pd

# ensure the repository root is on sys.path so that the script can be run
# from any working directory (including from inside a container).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from data_utils import *  # noqa: F401,F403
from qtnet.jax_models.representations import precompute_complexes, prepare_padded_batches
from qtnet.jax_models.optuna_hpo import OptunaHPO, make_factories


# ---- configuration constants ------------------------------------------------
ALL_ELEMENTS = ['H', 'C', 'N', 'O']
TARGET_COLUMNS = [
    'N', 'LI', 'Mu_X', 'Mu_Y', 'Mu_Z',
    'Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ',
]


# ----------------------------------------------------------------------------
# command‑line interface
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimisation for an atomic model",
    )
    p.add_argument(
        '--model-name',
        required=True,
        choices=[
            'TPaiNN', 'EquivariantGNN',
            'ScalarGNN', 'ScalarTPaiNN',
            'ScalarBaseline', 'ScalarBaselineEdges',
            'EquivariantGNN_FC', 'ScalarGNN_FC',
        ],
        help='name of the model to search (must match a key of make_factories)',
    )
    p.add_argument(
        '--data-pkl',
        default='data_curation/atomic/train_and_val.pkl',
        help='pickle file containing pre‑split dataframe',
    )
    p.add_argument(
        '--complexes-pkl',
        default=None,
        help='geometry cache used by `prepare_padded_batches` (auto-named from connectivity settings if omitted)',
    )
    p.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='batch size forwarded to prepare_padded_batches '
             '(default: 64 for fully-connected, 512 for sparse graphs)',
    )
    # global training configuration; defaults may be overridden below per-model
    p.add_argument(
        '--n-trials',
        type=int,
        default=None,
        help='number of Optuna trials (overrides model default)',
    )
    p.add_argument(
        '--epochs-per-trial',
        type=int,
        default=None,
        help='number of epochs to train for each trial',
    )
    p.add_argument(
        '--save-top-n',
        type=int,
        default=5,
        help='number of best trials to dump into JSON via OptunaHPO',
    )
    p.add_argument(
        '--cutoff',
        type=float,
        default=5.25,
        help='distance cutoff in Bohr for edge creation (ignored when --fully-connected)',
    )
    p.add_argument(
        '--max-neighbors',
        type=int,
        default=5,
        help='max neighbours per atom; omit for no limit (ignored when --fully-connected)',
    )
    p.add_argument(
        '--fully-connected',
        action='store_true',
        help='use fully connected graphs (all atom pairs as edges, no dim-2 cochain)',
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-derive complexes filename from connectivity settings if not given explicitly
    def _connectivity_suffix():
        if args.fully_connected:
            return "fc"
        s = f"cutoff{args.cutoff}"
        if args.max_neighbors is not None:
            s += f"_max{args.max_neighbors}"
        return s

    if args.complexes_pkl is None:
        args.complexes_pkl = os.path.join(
            'data_curation', 'atomic',
            f"precomputed_complexes_{_connectivity_suffix()}.pkl"
        )

    # Fully-connected graphs have O(N²) edges per molecule; use a much smaller
    # batch size to avoid OOM during JIT compilation and forward passes.
    if args.batch_size is None:
        args.batch_size = 64 if args.fully_connected else 512

    # --- load dataset --------------------------------------------------------
    df = pd.read_pickle(args.data_pkl)

    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}

    # --- precompute/load complexes (geometry only, fold independent) ---------
    if os.path.exists(args.complexes_pkl):
        print(f"Loading precomputed complexes from {args.complexes_pkl}")
        with open(args.complexes_pkl, 'rb') as f:
            complexes = pickle.load(f)
        print(f"  Loaded {len(complexes)} complexes")
    else:
        complexes = precompute_complexes(
            df,
            element_to_idx=element_to_idx,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
            fully_connected=args.fully_connected,
            output_path=args.complexes_pkl,
            verbose=False,
        )

    # --- single CV fold ------------------------------------------------------
    for fold in create_cv_splits(df, group_col='Murcko_Scaffold'):
        train_idx, val_idx = fold['train_idx'], fold['test_idx']
        train_df = df.iloc[train_idx, :].copy()
        val_df = df.iloc[val_idx, :].copy()

        molecular_stats = compute_molecular_stats(train_df)
        atomic_stats = compute_per_atom_stats(train_df)
        regularized_train = apply_z_regularization(
            train_df, molecular_stats, atomic_stats
        )
        regularized_val = apply_z_regularization(
            val_df, molecular_stats, atomic_stats
        )

        train_batches_padded = prepare_padded_batches(
            complexes,
            regularized_train,
            TARGET_COLUMNS,
            batch_size=args.batch_size,
            verbose=False,
            as_numpy=True,
        )
        val_batches_padded = prepare_padded_batches(
            complexes,
            regularized_val,
            TARGET_COLUMNS,
            batch_size=args.batch_size,
            verbose=False,
            as_numpy=True,
        )

        print(f"\nFold {fold['fold']}: "
              f"{len(train_batches_padded)} train batches, "
              f"{len(val_batches_padded)} val batches")
        break

    # --- run the Optuna search ----------------------------------------------
    factories = make_factories(num_species=len(ALL_ELEMENTS))

    # default training settings keyed by model name; these can be
    # overridden via CLI arguments.  Geometry/optimizer hyperparameters
    # such as weight_decay or patience are now part of the search space and
    # therefore are *not* specified here.
    model_defaults = {
        # Equivariant models: large space + slow forward pass → fewer trials,
        # more epochs so the median pruner (warmup=30) has a meaningful window.
        'TPaiNN':             {'n_trials': 40,  'epochs': 100},
        'EquivariantGNN':     {'n_trials': 40,  'epochs': 100},
        'EquivariantGNN_FC':  {'n_trials': 40,  'epochs': 100},
        # Scalar message-passing: faster, moderate space → more trials, shorter budget.
        'ScalarGNN':          {'n_trials': 100, 'epochs': 100},
        'ScalarTPaiNN':       {'n_trials': 100, 'epochs': 100},
        'ScalarGNN_FC':       {'n_trials': 100, 'epochs': 100},
        # Baselines: fast forward, but deeper layers need more epochs to converge.
        'ScalarBaseline':     {'n_trials': 80,  'epochs': 150},
        'ScalarBaselineEdges':{'n_trials': 80,  'epochs': 150},
    }

    defaults = model_defaults.get(args.model_name, {})
    n_trials = args.n_trials if args.n_trials is not None else defaults.get('n_trials')
    epochs = args.epochs_per_trial if args.epochs_per_trial is not None else defaults.get('epochs')

    builder = OptunaHPO(args.model_name, factories[args.model_name],
                        train_batches_padded, val_batches_padded)
    # apply training configuration
    builder = builder.with_training(n_trials=n_trials, epochs_per_trial=epochs)

    study = builder.run(save_top_n=args.save_top_n)

    print(f"Completed HPO for {args.model_name}; study name is '{study.study_name}'")


if __name__ == '__main__':
    main()
