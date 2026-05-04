#!/usr/bin/env python3
"""
Train one member of a QTNet ensemble for per-atom AIM property prediction.

Each invocation trains a single ensemble member on a scaffold-grouped 80/20
train/val split of aimel_clustered_molecular.pkl.  Run once per --member-idx
(0-4) to build a full ensemble; suited for SLURM job arrays.

The data split is deterministic: seeded from --ensemble-label + --member-idx.
Run with the same arguments to reproduce identically.

Model types:
  SGN2 - SGNN_v2             (scalar, needs SO(3) augmentation, --rotate-every)
  EGNX - EquivariantGNN_Flex (equivariant, augmentation optional)

Output layout:
  experiments/inference/{model_type}/model_{member_idx}/
    config.json      - all hyperparameters and connectivity settings
    stats.json       - per-atom stats for this member (used to denormalise)
    checkpoints/     - model_best_so_far, periodic + final snapshots
    loss/            - loss_history.json

Usage:
  python train_QTNet_ensemble.py --model-type SGN2 --ensemble-label qtnet_v1 --member-idx 0
  python train_QTNet_ensemble.py --model-type EGNX --ensemble-label qtnet_v1 --member-idx 2 --rotate-every 0
"""

import json
import os
import pickle
import sys
import argparse

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# Repo root discovery (mirrors train_multitask.py)
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
from qtnet.jax_models.augmentation import augment_batches_fast
from qtnet.data_utils import (
    create_cv_splits,
    compute_per_atom_stats,
    apply_z_regularization,
)
from qtnet.jax_models.representations import (
    precompute_complexes,
    prepare_padded_batches,
)
from qtnet.jax_models.train_utils import (
    make_optimizer,
    train_multitask,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    _state_to_numpy,
    _CHECKPOINTER,
    _resolve_checkpoint_path,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_ELEMENTS = ['H', 'C', 'N', 'O']
NUM_SPECIES = len(ALL_ELEMENTS)

TARGET_COLUMNS = [
    'N', 'LI', 'Mu_X', 'Mu_Y', 'Mu_Z',
    'Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ',
]

# Model registry — subset of train_multitask.py's MODEL_TYPE_MAP
MODEL_TYPE_MAP = {
    'SGN2': ('SGNN_v2',            'SGNN_v2'),
    'EGNX': ('EquivariantGNN_Flex', None),   # hardcoded hyperparams
}

MODEL_CLASSES = {
    'SGNN_v2':            SGNN_v2,
    'EquivariantGNN_Flex': EquivariantGNN_Flex,
}

# Default hyperparameters for EGNX (no optuna file; mirrors train_multitask.py)
_EGNX_MODEL_KWARGS = {
    'num_node_scalars': 16,
    'num_node_vectors': 8,
    'num_node_tensors': 8,
    'num_edge_scalars': 16,
    'num_edge_vectors': 8,
    'num_edge_tensors': 8,
    'embedding_dim': 32,
    'hidden_dim': 64,
    'hidden_l1_channels': 12,
    'hidden_l2_channels': 12,
    'num_layers': 4,
    'geometric_filter_dim': 32,
    'geo_basis_dim': 16,
    'hidden_tp_dim': 12,
    'use_node_geo_twice': False,
    'use_edge_geo_reminder': True,
    'use_tensor_products': True,
}
_EGNX_LR = 0.0038322168504927897
_EGNX_WD = 0.00011423254155608379

# Empty molecular stats — no-op placeholder so apply_z_regularization skips
# the molecular-level z-scoring step (we only normalise per-atom properties here)
_EMPTY_MOL_STATS = pd.DataFrame(index=['mean', 'std'])


# ---------------------------------------------------------------------------
# Helpers (mirrors train_multitask.py)
# ---------------------------------------------------------------------------

def load_optuna_config(model_class_name: str, optuna_dir: str):
    """Return (model_kwargs, lr, weight_decay) from the best trial JSON."""
    path = os.path.join(optuna_dir, f"{model_class_name}_optuna.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Optuna config not found: {path}")
    with open(path) as f:
        trials = json.load(f)
    best = trials[0]
    params = dict(best['params'])
    lr = params.pop('lr')
    weight_decay = params.pop('weight_decay', 0.0)
    params.pop('patience', None)
    return params, lr, weight_decay


def create_model(model_class_name: str, model_kwargs: dict, seed: int = 0):
    """Instantiate model from class name and kwargs."""
    cls = MODEL_CLASSES[model_class_name]
    return cls(num_species=NUM_SPECIES, rngs=nnx.Rngs(seed), **model_kwargs)


def get_member_split(df, ensemble_label: str, member_idx: int, n_splits: int = 5):
    """Return (train_idx, val_idx) for one ensemble member.

    Uses create_cv_splits with n_repeats=1 so that the five folds partition
    the dataset into disjoint 80/20 val sets.  The seed is derived from
    ensemble_label so that runs with the same label always produce the same
    splits regardless of machine or invocation order.
    """
    base_seed = hash(ensemble_label) % (2 ** 31)
    for fold_info in create_cv_splits(
        df,
        n_splits=n_splits,
        n_repeats=1,
        group_col='Murcko_Scaffold',
        base_seed=base_seed,
    ):
        if fold_info['cv_fold'] == member_idx:
            return fold_info['train_idx'], fold_info['test_idx']
    raise ValueError(
        f"member_idx={member_idx} not found for n_splits={n_splits}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train one QTNet ensemble member for per-atom AIM properties.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model types:
  SGN2  - SGNN_v2             (scalar; needs --rotate-every for SO(3) symmetry)
  EGNX  - EquivariantGNN_Flex (equivariant; augmentation optional)

Examples:
  python train_QTNet_ensemble.py --model-type SGN2 --ensemble-label qtnet_v1 --member-idx 0
  python train_QTNet_ensemble.py --model-type EGNX --ensemble-label qtnet_v1 --member-idx 4 --epochs 3000 --rotate-every 0
        """,
    )

    # Required
    parser.add_argument('--model-type', required=True,
                        choices=list(MODEL_TYPE_MAP.keys()))
    parser.add_argument('--ensemble-label', required=True,
                        help="Stable string tag for this ensemble run, e.g. 'qtnet_v1'")
    parser.add_argument('--member-idx', type=int, required=True,
                        help="Ensemble member index (0 to ensemble-size-1)")

    # Ensemble config
    parser.add_argument('--ensemble-size', type=int, default=5,
                        help="Total ensemble members (default 5; must be ≥ member-idx+1)")

    # Data
    parser.add_argument('--data-pkl', type=str,
                        default=os.path.join(
                            REPO_ROOT, 'data_curation', 'molecular',
                            'aimel_clustered_molecular.pkl'))
    parser.add_argument('--complexes-pkl', type=str, default=None,
                        help='Pre-built complexes cache; auto-named if omitted')
    parser.add_argument('--optuna-dir', type=str,
                        default=os.path.join(REPO_ROOT, 'optimal_hyperparams'))

    # Graph connectivity
    parser.add_argument('--cutoff', type=float, default=8.0,
                        help='Distance cutoff in Bohr for edge creation')
    parser.add_argument('--max-neighbors', type=int, default=12,
                        help='Max neighbours per atom (0 = no limit)')
    parser.add_argument('--fully-connected', action='store_true',
                        help='Use fully connected graph (no cutoff)')
    parser.add_argument('--max-dim', type=int, default=1,
                        help='Max cochain dimension (1 skips ring bags)')

    # Training
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rotate-every', type=int, default=1,
                        help='SO(3) augmentation every N epochs (0 = off). '
                             'Required for SGN2; optional for EGNX.')
    parser.add_argument('--no-element-weights', action='store_true',
                        help='Disable per-element loss weighting')
    parser.add_argument('--mse-loss', action='store_true',
                        help='Use per-component MSE loss instead of norm-based loss')
    parser.add_argument('--resume', type=int, default=None,
                        help='Epoch from which to resume (loads model + optimizer)')

    # Hyperparameter overrides (applied on top of optuna/defaults)
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Override weight decay')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Override num_layers in model kwargs')

    # Output
    parser.add_argument('--output-root', type=str,
                        default=os.path.join(PROJECT_ROOT, 'experiments', 'inference'))

    args = parser.parse_args()

    if args.member_idx >= args.ensemble_size:
        parser.error(
            f"--member-idx {args.member_idx} >= --ensemble-size {args.ensemble_size}"
        )

    model_class_name, optuna_key = MODEL_TYPE_MAP[args.model_type]

    # ------------------------------------------------------------------
    # 1. Hyperparameters
    # ------------------------------------------------------------------
    if optuna_key is None:
        model_kwargs = dict(_EGNX_MODEL_KWARGS)
        lr = _EGNX_LR
        weight_decay = _EGNX_WD
    else:
        model_kwargs, lr, weight_decay = load_optuna_config(optuna_key, args.optuna_dir)

    if args.lr is not None:
        lr = args.lr
    if args.weight_decay is not None:
        weight_decay = args.weight_decay
    if args.num_layers is not None:
        model_kwargs['num_layers'] = args.num_layers

    print(f"Model:  {args.model_type} ({model_class_name})")
    print(f"Member: {args.member_idx}/{args.ensemble_size-1}  "
          f"label={args.ensemble_label}")
    print(f"Hyperparameters: {json.dumps(model_kwargs, indent=2)}")
    print(f"lr={lr:.2e}  weight_decay={weight_decay:.2e}")

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

    # ------------------------------------------------------------------
    # 3. Load dataset
    # ------------------------------------------------------------------
    print(f"\nLoading dataset: {args.data_pkl}")
    df = pd.read_pickle(args.data_pkl)
    print(f"  {len(df)} molecules")

    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}

    # ------------------------------------------------------------------
    # 4. Precompute or load complexes (shared across all ensemble members)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Member-specific train/val split
    # ------------------------------------------------------------------
    print(f"\nBuilding split for member {args.member_idx} "
          f"(ensemble_label='{args.ensemble_label}')...")
    train_idx, val_idx = get_member_split(
        df, args.ensemble_label, args.member_idx, n_splits=args.ensemble_size,
    )
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    print(f"  Train: {len(train_df)} molecules  Val: {len(val_df)} molecules")

    # Keep originals for later use if needed
    train_df_orig = train_df.copy()
    val_df_orig   = val_df.copy()

    # ------------------------------------------------------------------
    # 6. Per-atom stats (train split only) + z-regularization
    #    No molecular-level stats needed — this task predicts per-atom
    #    properties only.  Pass an empty mol_stats and mol_cols=[].
    # ------------------------------------------------------------------
    atomic_stats = compute_per_atom_stats(train_df)

    regularized_train = apply_z_regularization(
        train_df, _EMPTY_MOL_STATS, atomic_stats, mol_cols=[],
    )
    regularized_val = apply_z_regularization(
        val_df, _EMPTY_MOL_STATS, atomic_stats, mol_cols=[],
    )

    element_weight_map = {
        el: float(atomic_stats.at[el, 'weight'])
        for el in ALL_ELEMENTS if el in atomic_stats.index
    }
    print(f"  Element weights: {element_weight_map}")

    # ------------------------------------------------------------------
    # 7. Padded batches
    # ------------------------------------------------------------------
    print("\nPreparing padded batches...")
    train_batches = prepare_padded_batches(
        complexes, regularized_train, TARGET_COLUMNS,
        batch_size=args.batch_size, verbose=True, as_numpy=True,
    )
    val_batches = prepare_padded_batches(
        complexes, regularized_val, TARGET_COLUMNS,
        batch_size=args.batch_size, verbose=True, as_numpy=True,
    )
    print(f"  {len(train_batches)} train batches, {len(val_batches)} val batches")

    # ------------------------------------------------------------------
    # 8. Model + optimizer
    # ------------------------------------------------------------------
    model_seed = args.seed + args.member_idx
    model = create_model(model_class_name, model_kwargs, seed=model_seed)
    n_params = count_parameters(model)
    print(f"\nModel parameters: {n_params:,}")

    tx = make_optimizer(
        lr=lr,
        weight_decay=weight_decay,
        use_reduce_on_plateau=True,
        plateau_accumulation_size=len(train_batches),
    )
    optimizer = nnx.Optimizer(model, tx)

    # ------------------------------------------------------------------
    # 9. Output directories
    # ------------------------------------------------------------------
    member_dir = os.path.join(
        args.output_root, args.model_type, f'model_{args.member_idx}',
    )
    checkpoint_dir = os.path.join(member_dir, 'checkpoints')
    loss_dir = os.path.join(member_dir, 'loss')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 10. Optionally resume
    # ------------------------------------------------------------------
    start_epoch = 0
    aug_key = jax.random.PRNGKey(model_seed + 7777)

    if args.resume is not None:
        ckpt_path = os.path.join(checkpoint_dir, f'model_epoch{args.resume}')
        model = load_checkpoint(model, ckpt_path)
        opt_path = os.path.join(checkpoint_dir, f'optimizer_epoch{args.resume}')
        if os.path.exists(opt_path):
            ref_opt = _state_to_numpy(nnx.state(optimizer))
            restored_opt = _CHECKPOINTER.restore(
                _resolve_checkpoint_path(opt_path), item=ref_opt,
            )
            nnx.update(optimizer, restored_opt)
            print(f"Loaded optimizer state from {opt_path}")
        start_epoch = args.resume

    # ------------------------------------------------------------------
    # 11. Save config.json (everything needed to reconstruct the model
    #     and rebuild matching batches at inference time)
    # ------------------------------------------------------------------
    config = {
        'model_type':       args.model_type,
        'model_class':      model_class_name,
        'model_kwargs':     model_kwargs,
        'lr':               lr,
        'weight_decay':     weight_decay,
        'ensemble_label':   args.ensemble_label,
        'member_idx':       args.member_idx,
        'ensemble_size':    args.ensemble_size,
        'seed':             model_seed,
        'epochs':           args.epochs,
        'batch_size':       args.batch_size,
        'n_params':         n_params,
        'cutoff':           args.cutoff,
        'max_neighbors':    args.max_neighbors,
        'fully_connected':  args.fully_connected,
        'max_dim':          args.max_dim,
        'rotate_every':     args.rotate_every,
        'augmentation_key': jax.random.key_data(aug_key).tolist(),
        'element_weights':  element_weight_map if not args.no_element_weights else None,
        'n_train':          len(train_df),
        'n_val':            len(val_df),
    }
    with open(os.path.join(member_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Save per-member atomic stats for denormalisation at inference time
    stats_payload = {
        'atomic_stats': atomic_stats.to_dict(),
    }
    with open(os.path.join(member_dir, 'stats.json'), 'w') as f:
        json.dump(stats_payload, f, indent=2)

    print(f"\nConfig + stats written to {member_dir}/")

    # ------------------------------------------------------------------
    # 12. Build element-weight array for the loss
    # ------------------------------------------------------------------
    if args.no_element_weights:
        element_weight_array = None
    else:
        element_weight_array = jnp.array(
            [element_weight_map.get(el, 1.0) for el in ALL_ELEMENTS],
            dtype=jnp.float32,
        )

    # ------------------------------------------------------------------
    # 13. Train
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs - start_epoch} epochs "
          f"(total={args.epochs})...")
    train_multitask(
        model=model,
        optimizer=optimizer,
        train_batches=train_batches,
        val_batches=val_batches,
        epochs=args.epochs,
        start_epoch=start_epoch,
        save_every=args.save_every,
        checkpoint_dir=checkpoint_dir,
        loss_dir=loss_dir,
        verbose=True,
        print_every=args.print_every,
        use_mse_loss=args.mse_loss,
        element_weight_array=element_weight_array,
        augment_fn=augment_batches_fast if args.rotate_every > 0 else None,
        augment_key=aug_key if args.rotate_every > 0 else None,
        rotate_every=args.rotate_every,
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DONE")
    print(f"  Model:   {args.model_type} ({model_class_name})")
    print(f"  Member:  {args.member_idx}  label={args.ensemble_label}")
    print(f"  Output:  {member_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
