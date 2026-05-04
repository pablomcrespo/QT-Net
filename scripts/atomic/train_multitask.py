#!/usr/bin/env python3
"""
Multitask training script for atomic models on AIMEl dataset.

Each model type loads its best hyperparameters (architecture + optimizer)
from the corresponding ``*_optuna.json`` file in ``optimal_hyperparams/``.
A single fold index (0-24, from 5x5-fold grouped CV) selects the train/val
split; data preparation mirrors the workflow in ``run_hpo.py``.

Model types:
  ETNN - TPaiNN                 (equivariant, latent edges, fresh bags)
  EGNN - EquivariantGNN         (equivariant, latent edges)
  SGNN - ScalarGNN              (scalar, latent edges)
  STNN - ScalarTPaiNN           (scalar, latent edges, fresh bags)
  SBAS - ScalarBaseline         (scalar, no edges)
  SBAE - ScalarBaselineEdges    (scalar, fresh edges)

Output layout (relative to --output-dir, default experiments/atomic):

    <model_type>/fold_<n>/checkpoints/   - periodic + best + final model
    <model_type>/fold_<n>/loss/          - loss_history.json
    <model_type>/fold_<n>/val_preds.pkl  - unregularized val predictions
    <model_type>/fold_<n>/test_preds.pkl - unregularized test predictions

Usage:
    python train_multitask.py --model-type ETNN --fold 0
    python train_multitask.py --model-type SGNN --fold 12 --epochs 500
"""

import os
import sys
import argparse
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# Locate repository root (project root) and ensure package imports work
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_repo_root(start_dir: str):
    """Ascend from start_dir until a folder that looks like the project root.

    Heuristic: directory contains `data_curation`. Falls back two levels up.
    """
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, 'data_curation')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            # reached filesystem root; fallback to two levels up from start_dir
            return os.path.abspath(os.path.join(start_dir, '..', '..'))
        cur = parent

REPO_ROOT = _find_repo_root(SCRIPT_DIR)

# Ensure REPO_ROOT is on sys.path for file-based operations
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Find the directory that contains the `qtnet` package and add its parent
# to sys.path so `import qtnet...` works irrespective of where the script lives.
qtnet_parent = None
for root, dirs, files in os.walk(REPO_ROOT):
    if 'qtnet' in dirs:
        qtnet_parent = root
        break

if qtnet_parent and qtnet_parent not in sys.path:
    sys.path.insert(0, qtnet_parent)

# Prefer placing experiment outputs in the project-level directory (parent of JAX_TMP)
# if that parent contains an `experiments` directory or project artifacts (SIF, pyproject).
proj_candidate = os.path.dirname(REPO_ROOT)
if (os.path.isdir(os.path.join(proj_candidate, 'experiments')) or
        os.path.exists(os.path.join(proj_candidate, 'qm9-aim-jax.sif')) or
        os.path.exists(os.path.join(proj_candidate, 'pyproject.toml'))):
    PROJECT_ROOT = proj_candidate
else:
    PROJECT_ROOT = REPO_ROOT

# Package-style imports (make module layout explicit)
from qtnet.jax_models.models_equivariant import TPaiNN, EquivariantGNN
from qtnet.jax_models.models_scalar import (
    ScalarGNN,
    ScalarTPaiNN,
    ScalarBaseline,
    ScalarBaselineEdges,
)
from qtnet.jax_models.models_inference import SGNN_v2, EGNN_v2, EquivariantGNN_Flex
from qtnet.jax_models.augmentation import augment_batches_fast
from qtnet.data_utils import (
    create_cv_splits,
    compute_molecular_stats,
    compute_per_atom_stats,
    apply_z_regularization,
)
from qtnet.jax_models.representations import (
    precompute_complexes,
    prepare_padded_batches,
)
from qtnet.jax_models.train_utils import (
    make_optimizer,
    compute_multitask_loss,
    train_multitask,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    get_lr_scale,
    ATOMIC_PROPERTY_NAMES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_ELEMENTS = ['H', 'C', 'N', 'O']
TARGET_COLUMNS = [
    'N', 'LI', 'Mu_X', 'Mu_Y', 'Mu_Z',
    'Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ',
]

MODEL_TYPE_MAP = {
    'ETNN': ('TPaiNN',           'TPaiNN'),
    'EGNN': ('EquivariantGNN',   'EquivariantGNN'),
    'SGNN': ('ScalarGNN',        'ScalarGNN'),
    'STNN': ('ScalarTPaiNN',     'ScalarTPaiNN'),
    'SBAS': ('ScalarBaseline',   'ScalarBaseline'),
    'SBAE': ('ScalarBaselineEdges', 'ScalarBaselineEdges'),
    'EGNF': ('EquivariantGNN',   'EquivariantGNN_FC'),
    'SGNF': ('ScalarGNN',        'ScalarGNN_FC'),
    'SGN2': ('SGNN_v2',          'SGNN_v2'),
    'SNF2': ('SGNN_v2',          'SGNN_v2_FC'),
    'EGN2': ('EGNN_v2',          'EGNN_v2'),
    'EGNX': ('EquivariantGNN_Flex', None),   # hardcoded hyperparams, no optuna file
    'SGNN_final': ('SGNN_v2',    'SGNN_v2'),
    'EGX2': ('EquivariantGNN_Flex', None)
}

# Hardcoded hyperparams for EGNX (no optuna file)
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

MODEL_CLASSES = {
    'TPaiNN': TPaiNN,
    'EquivariantGNN': EquivariantGNN,
    'ScalarGNN': ScalarGNN,
    'ScalarTPaiNN': ScalarTPaiNN,
    'ScalarBaseline': ScalarBaseline,
    'ScalarBaselineEdges': ScalarBaselineEdges,
    'SGNN_v2': SGNN_v2,
    'EGNN_v2': EGNN_v2,
    'EquivariantGNN_Flex': EquivariantGNN_Flex,
}

NUM_SPECIES = len(ALL_ELEMENTS)


# augment_batches_fast imported from augmentation module — operates on
# numpy-backed padded batches in-place (no JAX→numpy round-trip needed).


# ============================================================================
# Model / config helpers
# ============================================================================

def load_optuna_config(model_class_name: str, optuna_dir: str):
    """Return the best trial's params dict from the optuna JSON.

    The file is expected at ``<optuna_dir>/<model_class_name>_optuna.json``.
    Returns (model_kwargs, lr, weight_decay).
    """
    path = os.path.join(optuna_dir, f"{model_class_name}_optuna.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Optuna config not found: {path}")
    with open(path, 'r') as f:
        trials = json.load(f)
    best = trials[0]  # sorted by value (ascending = best first)
    params = dict(best['params'])

    lr = params.pop('lr')
    weight_decay = params.pop('weight_decay', 0.0)
    params.pop('patience', None)  # not used in constructor

    return params, lr, weight_decay


def create_model(model_class_name: str, model_kwargs: dict, seed: int = 0):
    """Instantiate a model from its class name and kwargs."""
    cls = MODEL_CLASSES[model_class_name]
    rngs = nnx.Rngs(seed)
    return cls(num_species=NUM_SPECIES, rngs=rngs, **model_kwargs)


# ============================================================================
# Prediction + un-regularization helpers
# ============================================================================

def _unregularize_predictions(
    pred_array: np.ndarray,
    atoms: list,
    per_atom_stats: pd.DataFrame,
):
    """Invert the z-regularization applied to atomic targets.

    pred_array: (n_atoms, 10) in order TARGET_COLUMNS =
        [N, LI, Mu_X, Mu_Y, Mu_Z, Q_XY, Q_XZ, Q_YZ, Q_aniso, Q_ZZ]

    Returns an array of the same shape with the original scale restored.
    """
    out = np.copy(pred_array)
    for i, at in enumerate(atoms):
        if at not in per_atom_stats.index:
            warnings.warn(f"Atom type '{at}' not in per_atom_stats; cannot unregularize")
            continue
        row_stats = per_atom_stats.loc[at]
        # N  (col 0)
        n_std = row_stats['N_std']
        n_mean = row_stats['N_mean']
        if not (pd.isna(n_std) or n_std == 0):
            out[i, 0] = out[i, 0] * n_std + n_mean
        # LI (col 1)
        li_std = row_stats['LI_std']
        li_mean = row_stats['LI_mean']
        if not (pd.isna(li_std) or li_std == 0):
            out[i, 1] = out[i, 1] * li_std + li_mean
        # Mu_X, Mu_Y, Mu_Z (cols 2-4)  -- divided by Mu_rms
        mu_rms = row_stats['Mu_rms']
        if not (pd.isna(mu_rms) or mu_rms == 0):
            out[i, 2:5] = out[i, 2:5] * mu_rms
        # Q_XY..Q_ZZ (cols 5-9) -- divided by Q_rms
        q_rms = row_stats['Q_rms']
        if not (pd.isna(q_rms) or q_rms == 0):
            out[i, 5:10] = out[i, 5:10] * q_rms
    return out


def _collect_predictions(model, batches, df_orig, complexes, per_atom_stats):
    """Run the model on *batches* and build an unregularized predictions DataFrame.

    Returns a DataFrame with one row per molecule (same index as
    ``df_orig``).  The ``pred_*`` and ``target_*`` columns contain
    per-atom arrays (lists of length ``n_atoms``) on the original
    (unregularized) scale.  The ``atom`` and ``atom_cluster_labels``
    columns are copied from the original DataFrame for convenience.
    """
    # prepare_padded_batches iterates df.iterrows() in order, so the
    # i-th molecule across all batches corresponds to the i-th row.

    @nnx.jit
    def predict(model, batch):
        return model(batch)

    df_indices = list(df_orig.index)
    atom_col = 'atoms' if 'atoms' in df_orig.columns else 'atom'
    has_cluster = 'atom_cluster_labels' in df_orig.columns

    # Collect per-molecule predictions from batches
    all_preds = []  # list of (n_atoms, 10) np arrays per molecule
    for batch in batches:
        preds = predict(model, batch)
        pred_concat = np.asarray(jnp.concatenate(
            [preds['scalars'], preds['vectors'], preds['tensors']], axis=-1
        ))
        # Split batch back into individual molecules
        num_cells = np.asarray(batch.cochain_batches[0].num_cells)
        offset = 0
        for nc in num_cells:
            nc = int(nc)
            all_preds.append(pred_concat[offset:offset + nc])
            offset += nc

    # Build DataFrame: one row per molecule, indexed by df_orig.index
    records = []
    for i, idx in enumerate(df_indices):
        if i >= len(all_preds):
            break
        row = df_orig.loc[idx]
        atoms = list(row[atom_col])
        n_atoms = len(atoms)
        pred = all_preds[i][:n_atoms]
        unreg_pred = _unregularize_predictions(pred, atoms, per_atom_stats)

        # Original (unregularized) targets from the original df
        orig_targets = np.zeros((n_atoms, len(TARGET_COLUMNS)), dtype=np.float32)
        for ci, col in enumerate(TARGET_COLUMNS):
            vals = np.asarray(row[col], dtype=np.float32)
            orig_targets[:, ci] = vals[:n_atoms]

        rec = {'df_index': idx, atom_col: atoms}
        if has_cluster:
            rec['atom_cluster_labels'] = list(row['atom_cluster_labels'])
        for ci, prop in enumerate(TARGET_COLUMNS):
            rec[f'pred_{prop}'] = unreg_pred[:, ci].tolist()
            rec[f'target_{prop}'] = orig_targets[:, ci].tolist()
        records.append(rec)

    result = pd.DataFrame(records)
    if 'df_index' in result.columns:
        result = result.set_index('df_index')
        result.index.name = None
    return result


# ============================================================================
# Fold selection helper
# ============================================================================

def get_fold(df, fold_index):
    """Return (train_idx, val_idx) for the given fold number (0-24)."""
    for fold in create_cv_splits(df, group_col='Murcko_Scaffold'):
        if fold['fold'] == fold_index:
            return fold['train_idx'], fold['test_idx']
    raise ValueError(f"Fold {fold_index} not found (expected 0-24)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train atomic models on AIMEl dataset (single fold)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model types:
  ETNN - TPaiNN          (equivariant, with rings)
  EGNN - EquivariantGNN  (equivariant, no rings)
  SGNN - ScalarGNN       (scalar, no rings)
  STNN - ScalarTPaiNN    (scalar, with rings)
  SBAS - ScalarBaseline  (scalar, nodes only, with geometry)
  EGNF - EquivariantGNN FC (equivariant, fully-connected graph)
  SGNF - ScalarGNN FC      (scalar, fully-connected graph)
  SGN2 - SGNN_v2  (scalar, Bessel+cutoff normalised gates)
  SNF2 - SGNN_v2 FC  (scalar FC, Gaussian RBF normalised gates)
  EGN2 - EGNN_v2  (equivariant, GeoReminder at NodeCob + EdgeGeoReminder)
  EGNX - EquivariantGNN_Flex  (equivariant, configurable TP + geo reminders)
Examples:
  python train_multitask.py --model-type ETNN --fold 0
  python train_multitask.py --model-type SGNN --fold 12 --epochs 500
  python train_multitask.py --model-type SGN2 --fold 0 --rotate-every 5
        """
    )
    parser.add_argument(
        "--model-type", type=str, required=True,
        choices=list(MODEL_TYPE_MAP.keys()),
        help="Model type to train",
    )
    parser.add_argument(
        "--fold", type=int, required=True,
        help="Fold index (0-24) for cross-validation",
    )
    parser.add_argument(
        "--data-pkl", type=str,
        default=os.path.join(REPO_ROOT, 'data_curation', 'atomic', 'train_and_val.pkl'),
        help="Path to train_and_val.pkl",
    )
    parser.add_argument(
        "--test-pkl", type=str,
        default=os.path.join(REPO_ROOT, 'data_curation', 'atomic', 'test.pkl'),
        help="Path to test.pkl",
    )
    parser.add_argument(
        "--complexes-pkl", type=str,
        default=None,
        help="Precomputed complexes for train_and_val data (auto-named from connectivity settings if omitted)",
    )
    parser.add_argument(
        "--test-complexes-pkl", type=str,
        default=None,
        help="Precomputed complexes for test data (auto-named from connectivity settings if omitted)",
    )
    parser.add_argument(
        "--optuna-dir", type=str,
        default=os.path.join(REPO_ROOT, 'optimal_hyperparams'),
        help="Directory containing *_optuna.json files",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(PROJECT_ROOT, 'experiments', 'atomic'),
        help="Root output directory",
    )
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=250,
                        help="Checkpoint every N epochs")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for model initialisation")
    parser.add_argument(
        "--cutoff", type=float, default=5.25,
        help="Distance cutoff in Bohr for edge creation (ignored when --fully-connected)",
    )
    parser.add_argument(
        "--max-neighbors", type=int, default=5,
        help="Max neighbours per atom; omit for no limit (ignored when --fully-connected)",
    )
    parser.add_argument(
        "--fully-connected", action="store_true",
        help="Use fully connected graphs (all atom pairs as edges, no dim-2 cochain)",
    )
    parser.add_argument(
        "--resume", type = int, default = None,
        help="Epoch from which to resume training",
    )
    parser.add_argument(
        "--rotate-every", type=int, default=0,
        help="Apply a random SO(3) rotation to training batches every N epochs (0 = disabled)",
    )
    parser.add_argument(
        "--max-dim", type=int, default=2,
        help="Maximum cochain dimension (1 = skip bags-of-bonds, faster precompute)",
    )
    parser.add_argument(
        "--no-element-weights", action="store_true",
        help="Disable per-element loss weighting (use uniform atom weights)",
    )
    parser.add_argument(
        "--mse-loss", action="store_true",
        help="Use per-component MSE loss instead of the default norm-based loss "
             "(no Frobenius weighting on tensor components)",
    )

    args = parser.parse_args()

    # Auto-derive complexes filenames from connectivity settings if not given explicitly
    def _connectivity_suffix():
        if args.fully_connected:
            return "fc"
        s = f"cutoff{args.cutoff}"
        if args.max_neighbors is not None:
            s += f"_max{args.max_neighbors}"
        return s

    _atomic_data_dir = os.path.join(REPO_ROOT, 'data_curation', 'atomic')
    if args.complexes_pkl is None:
        args.complexes_pkl = os.path.join(
            _atomic_data_dir, f"precomputed_complexes_{_connectivity_suffix()}.pkl"
        )
    if args.test_complexes_pkl is None:
        args.test_complexes_pkl = os.path.join(
            _atomic_data_dir, f"precomputed_test_{_connectivity_suffix()}.pkl"
        )

    model_class_name, optuna_key = MODEL_TYPE_MAP[args.model_type]

    # ------------------------------------------------------------------
    # 1. Load optimal hyperparameters
    # ------------------------------------------------------------------
    if optuna_key is None:
        # EGNX uses hardcoded hyperparams (no optuna file available)
        model_kwargs = dict(_EGNX_MODEL_KWARGS)
        if args.model_type == 'EGX2':
            model_kwargs['num_layers'] = 7
        lr = _EGNX_LR
        weight_decay = _EGNX_WD
    else:
        model_kwargs, lr, weight_decay = load_optuna_config(optuna_key, args.optuna_dir)

    # SNF2 uses the same SGNN_v2 class with a Gaussian RBF distance encoder
    if args.model_type == 'SNF2':
        model_kwargs['distance_encoder'] = 'gaussian'

    print(f"Model: {args.model_type} ({model_class_name})")
    print(f"Fold:  {args.fold}")
    print(f"Hyperparameters: {json.dumps(model_kwargs, indent=2)}")
    print(f"lr={lr:.6f}, weight_decay={weight_decay:.6f}")

    # ------------------------------------------------------------------
    # 2. Load dataset & complexes
    # ------------------------------------------------------------------
    print("\nLoading dataset...")
    df = pd.read_pickle(args.data_pkl)
    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}

    # Train/val complexes
    if os.path.exists(args.complexes_pkl):
        print(f"Loading precomputed complexes from {args.complexes_pkl}")
        with open(args.complexes_pkl, 'rb') as f:
            complexes = pickle.load(f)
        print(f"  Loaded {len(complexes)} complexes")
    else:
        complexes = precompute_complexes(
            df, element_to_idx=element_to_idx,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
            fully_connected=args.fully_connected,
            max_dim=args.max_dim,
            output_path=args.complexes_pkl, verbose=True,
        )

    # ------------------------------------------------------------------
    # 3. Build fold split
    # ------------------------------------------------------------------
    print(f"\nBuilding fold {args.fold}...")
    train_idx, val_idx = get_fold(df, args.fold)

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    print(f"  Train: {len(train_df)} molecules, Val: {len(val_df)} molecules")

    # Keep copies of the ORIGINAL (unregularized) dataframes for later
    # when we need to produce unregularized predictions.
    train_df_orig = train_df.copy()
    val_df_orig = val_df.copy()

    # ------------------------------------------------------------------
    # 4. Compute stats & regularize
    # ------------------------------------------------------------------
    molecular_stats = compute_molecular_stats(train_df)
    atomic_stats = compute_per_atom_stats(train_df)

    regularized_train = apply_z_regularization(train_df, molecular_stats, atomic_stats)
    regularized_val = apply_z_regularization(val_df, molecular_stats, atomic_stats)

    # Element weights are now computed inside compute_per_atom_stats as
    # the 'weight' column: w_el = sqrt(total/count_el), mean-normalized.
    element_weight_map = {el: float(atomic_stats.at[el, 'weight'])
                          for el in ALL_ELEMENTS if el in atomic_stats.index}
    print(f"  Element weights (fold {args.fold}): {element_weight_map}")

    # ------------------------------------------------------------------
    # 5. Prepare padded batches
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
    # 6. Create model & optimizer
    # ------------------------------------------------------------------
    model = create_model(model_class_name, model_kwargs, seed=args.seed+args.fold)
    n_params = count_parameters(model)
    print(f"\nModel parameters: {n_params:,}")

    tx = make_optimizer(
        lr=lr, weight_decay=weight_decay,
        use_reduce_on_plateau=True,
        plateau_accumulation_size=len(train_batches),
    )
    optimizer = nnx.Optimizer(model, tx)

    # ------------------------------------------------------------------
    # 7. Set up output directories
    # ------------------------------------------------------------------
    fold_dir = os.path.join(args.output_dir, args.model_type, f"fold_{args.fold}")
    checkpoint_dir = os.path.join(fold_dir, "checkpoints")
    loss_dir = os.path.join(fold_dir, "loss")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    start_epoch = 0
    if args.resume is not None:
        model = load_checkpoint(
            model, os.path.join(checkpoint_dir, f"model_epoch{args.resume}"),
        )
        opt_ckpt = os.path.join(checkpoint_dir, f"optimizer_epoch{args.resume}")
        if os.path.exists(opt_ckpt):
            from qtnet.jax_models.train_utils import _state_to_numpy, _CHECKPOINTER, _resolve_checkpoint_path
            ref_opt_state = _state_to_numpy(nnx.state(optimizer))
            restored_opt = _CHECKPOINTER.restore(
                _resolve_checkpoint_path(opt_ckpt), item=ref_opt_state,
            )
            nnx.update(optimizer, restored_opt)
            print(f"Loaded optimizer state from {opt_ckpt}")
        start_epoch = args.resume

    # Save config for reproducibility
    aug_key = jax.random.PRNGKey(args.seed + args.fold + 7777)
    config = {
        'model_type': args.model_type,
        'model_class': model_class_name,
        'model_kwargs': model_kwargs,
        'lr': lr,
        'weight_decay': weight_decay,
        'fold': args.fold,
        'seed': args.seed+args.fold,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'n_params': n_params,
        'cutoff': args.cutoff,
        'max_neighbors': args.max_neighbors,
        'fully_connected': args.fully_connected,
        'rotate_every': args.rotate_every,
        'augmentation_key': jax.random.key_data(aug_key).tolist(),
        'element_weights': element_weight_map if not args.no_element_weights else None,
        'no_element_weights': args.no_element_weights,
        'mse_loss': args.mse_loss,
    }
    with open(os.path.join(fold_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------------
    # Build per-species weight array for the loss function
    if args.no_element_weights:
        element_weight_array = None
    else:
        element_weight_array = jnp.array(
            [element_weight_map[el] for el in ALL_ELEMENTS], dtype=jnp.float32
        )  # shape (num_species,), indexed by species index

    print(f"\nTraining for {args.epochs - start_epoch} epochs (total={args.epochs})...")
    train_history, val_history = train_multitask(
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
    # 9. Load best model for prediction
    # ------------------------------------------------------------------
    best_path = os.path.join(checkpoint_dir, "model_best_so_far")
    if os.path.exists(best_path):
        print("\nLoading best model for predictions...")
        model = load_checkpoint(model, best_path)
    else:
        print("\nNo best-model checkpoint found; using final model for predictions.")

    # ------------------------------------------------------------------
    # 10. Validation-set predictions (unregularized)
    # ------------------------------------------------------------------
    print("Generating validation predictions...")
    val_pred_df = _collect_predictions(
        model, val_batches, val_df_orig, complexes, atomic_stats,
    )
    val_pred_df['fold'] = args.fold
    val_pred_df['model'] = args.model_type
    val_preds_path = os.path.join(fold_dir, "val_preds.pkl")
    val_pred_df.to_pickle(val_preds_path)
    print(f"  Saved {len(val_pred_df)} atom predictions to {val_preds_path}")

    # ------------------------------------------------------------------
    # 11. Test-set predictions (unregularized)
    # ------------------------------------------------------------------
    print("\nPreparing test set...")
    test_df = pd.read_pickle(args.test_pkl)

    # Load or compute test complexes
    if os.path.exists(args.test_complexes_pkl):
        print(f"Loading precomputed test complexes from {args.test_complexes_pkl}")
        with open(args.test_complexes_pkl, 'rb') as f:
            test_complexes = pickle.load(f)
        print(f"  Loaded {len(test_complexes)} test complexes")
    else:
        print("Precomputing test complexes...")
        test_complexes = precompute_complexes(
            test_df, element_to_idx=element_to_idx,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
            fully_connected=args.fully_connected,
            max_dim=args.max_dim,
            output_path=args.test_complexes_pkl, verbose=True,
        )

    # Regularize test data using *training* stats (same fold)
    test_df_orig = test_df.copy()
    regularized_test = apply_z_regularization(test_df.copy(), molecular_stats, atomic_stats)

    test_batches = prepare_padded_batches(
        test_complexes, regularized_test, TARGET_COLUMNS,
        batch_size=args.batch_size, verbose=True, as_numpy=True,
    )

    print("Generating test predictions...")
    test_pred_df = _collect_predictions(
        model, test_batches, test_df_orig, test_complexes, atomic_stats,
    )
    test_pred_df['fold'] = args.fold
    test_pred_df['model'] = args.model_type
    test_preds_path = os.path.join(fold_dir, "test_preds.pkl")
    test_pred_df.to_pickle(test_preds_path)
    print(f"  Saved {len(test_pred_df)} atom predictions to {test_preds_path}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DONE")
    print(f"  Model:  {args.model_type} ({model_class_name})")
    print(f"  Fold:   {args.fold}")
    print(f"  Output: {fold_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
