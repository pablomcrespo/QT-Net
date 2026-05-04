#!/usr/bin/env python3
"""
Run QTNet ensemble inference on QM9 to produce per-atom AIM properties.

Loads all trained ensemble members from experiments/inference/{model_type}/model_*/,
runs each on qm9_filtered.pkl, denormalises each member's predictions using its
own per-atom stats, then averages across members.

The output (qm9_inferred.pkl) has all original qm9_filtered columns plus ten
new per-atom list-columns matching the aimel_clustered_molecular.pkl schema:
  N, LI, Mu_X, Mu_Y, Mu_Z, Q_XY, Q_XZ, Q_YZ, Q_aniso, Q_ZZ

Ensemble std columns (N_std, LI_std, …) are also written when more than one
member is present, for downstream uncertainty estimates.

Usage:
  python infer_QTAIM_QM9.py --model-type SGN2
  python infer_QTAIM_QM9.py --model-type both   # average over SGN2 + EGNX ensembles
"""

import argparse
import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from flax import nnx

# ---------------------------------------------------------------------------
# Repo / project root (mirrors train_QTNet_ensemble.py)
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
from qtnet.jax_models.representations import (
    precompute_complexes,
    prepare_padded_batches,
)
from qtnet.jax_models.train_utils import load_checkpoint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_ELEMENTS = ['H', 'C', 'N', 'O']
NUM_SPECIES = len(ALL_ELEMENTS)

TARGET_COLUMNS = [
    'N', 'LI', 'Mu_X', 'Mu_Y', 'Mu_Z',
    'Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ',
]

MODEL_CLASSES = {
    'SGNN_v2':            SGNN_v2,
    'EquivariantGNN_Flex': EquivariantGNN_Flex,
}

_MODEL_TYPE_TO_CLASS = {
    'SGN2': 'SGNN_v2',
    'EGNX': 'EquivariantGNN_Flex',
}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _create_model(model_class_name: str, model_kwargs: dict, seed: int):
    cls = MODEL_CLASSES[model_class_name]
    return cls(num_species=NUM_SPECIES, rngs=nnx.Rngs(seed), **model_kwargs)


# ---------------------------------------------------------------------------
# Per-member inference helpers
# ---------------------------------------------------------------------------

def _unregularize_predictions(
    pred_array: np.ndarray,
    atoms: list,
    per_atom_stats: pd.DataFrame,
) -> np.ndarray:
    """Invert z-regularization for per-atom AIM predictions.

    pred_array: (n_atoms, 10) in TARGET_COLUMNS order:
        [N, LI, Mu_X, Mu_Y, Mu_Z, Q_XY, Q_XZ, Q_YZ, Q_aniso, Q_ZZ]

    Returns array of the same shape in physical (unregularized) units.
    """
    out = pred_array.copy()
    for i, at in enumerate(atoms):
        if at not in per_atom_stats.index:
            warnings.warn(f"Atom type '{at}' missing from per_atom_stats; "
                          "leaving normalised value")
            continue
        row = per_atom_stats.loc[at]

        n_std, n_mean = float(row['N_std']), float(row['N_mean'])
        if not (np.isnan(n_std) or n_std == 0):
            out[i, 0] = out[i, 0] * n_std + n_mean

        li_std, li_mean = float(row['LI_std']), float(row['LI_mean'])
        if not (np.isnan(li_std) or li_std == 0):
            out[i, 1] = out[i, 1] * li_std + li_mean

        mu_rms = float(row['Mu_rms'])
        if not (np.isnan(mu_rms) or mu_rms == 0):
            out[i, 2:5] = out[i, 2:5] * mu_rms

        q_rms = float(row['Q_rms'])
        if not (np.isnan(q_rms) or q_rms == 0):
            out[i, 5:10] = out[i, 5:10] * q_rms

    return out


@nnx.jit
def _predict_batch(model, batch):
    return model(batch)


def _run_member_inference(
    model,
    batches: list,
    qm9: pd.DataFrame,
    per_atom_stats: pd.DataFrame,
    atom_col: str,
) -> list:
    """Run one ensemble member over all batches.

    Returns a list of (n_atoms, 10) float32 arrays, one per molecule,
    in the same order as qm9.iterrows().
    """
    import jax.numpy as jnp

    raw_preds = []
    for batch in batches:
        preds = _predict_batch(model, batch)
        pred_concat = np.asarray(jnp.concatenate(
            [preds['scalars'], preds['vectors'], preds['tensors']], axis=-1
        ))
        num_cells = np.asarray(batch.cochain_batches[0].num_cells)
        offset = 0
        for nc in num_cells:
            nc = int(nc)
            raw_preds.append(pred_concat[offset:offset + nc])
            offset += nc

    # Denormalise each molecule with its per-atom stats
    result = []
    for i, (idx, row) in enumerate(qm9.iterrows()):
        if i >= len(raw_preds):
            break
        atoms = list(row[atom_col])
        n_atoms = len(atoms)
        pred = raw_preds[i][:n_atoms]
        unreg = _unregularize_predictions(pred, atoms, per_atom_stats)
        result.append(unreg)

    return result


# ---------------------------------------------------------------------------
# Member discovery
# ---------------------------------------------------------------------------

def _discover_members(ensemble_root: str, model_type: str) -> list:
    """Return sorted list of (member_dir, config, atomic_stats) tuples."""
    pattern = os.path.join(ensemble_root, model_type, 'model_*')
    dirs = sorted(
        glob.glob(pattern),
        key=lambda d: int(os.path.basename(d).split('_')[-1]),
    )
    if not dirs:
        raise FileNotFoundError(
            f"No ensemble members found at {pattern}\n"
            f"Train them first with train_QTNet_ensemble.py"
        )

    members = []
    for d in dirs:
        cfg_path = os.path.join(d, 'config.json')
        stats_path = os.path.join(d, 'stats.json')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config.json in {d}")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Missing stats.json in {d}")

        with open(cfg_path) as f:
            cfg = json.load(f)
        with open(stats_path) as f:
            stats_payload = json.load(f)

        atomic_stats = pd.DataFrame.from_dict(stats_payload['atomic_stats'])
        members.append((d, cfg, atomic_stats))

    print(f"  Found {len(members)} members for {model_type}")
    return members


def _assert_consistent_connectivity(members: list):
    """All members of one architecture must share the same graph settings."""
    keys = ['cutoff', 'max_neighbors', 'fully_connected', 'max_dim']
    ref_cfg = members[0][1]
    for d, cfg, _ in members[1:]:
        for k in keys:
            if cfg.get(k) != ref_cfg.get(k):
                raise ValueError(
                    f"Connectivity mismatch between members:\n"
                    f"  model_0: {k}={ref_cfg.get(k)}\n"
                    f"  {os.path.basename(d)}: {k}={cfg.get(k)}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run QTNet ensemble inference on QM9 to produce per-atom AIM properties.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_QTAIM_QM9.py --model-type SGN2
  python infer_QTAIM_QM9.py --model-type EGNX
  python infer_QTAIM_QM9.py --model-type both   # average SGN2 + EGNX
        """,
    )
    parser.add_argument('--model-type', default='SGN2',
                        choices=['SGN2', 'EGNX', 'both'],
                        help="Which ensemble(s) to use for inference")
    parser.add_argument('--ensemble-root', type=str,
                        default=os.path.join(PROJECT_ROOT, 'experiments', 'inference'))
    parser.add_argument('--qm9-pkl', type=str,
                        default=os.path.join(
                            REPO_ROOT, 'data_curation', 'molecular',
                            'qm9_filtered.pkl'))
    parser.add_argument('--output-pkl', type=str,
                        default=os.path.join(
                            REPO_ROOT, 'data_curation', 'molecular',
                            'qm9_inferred.pkl'))
    parser.add_argument('--qm9-complexes-pkl', type=str, default=None,
                        help='Cache path for QM9 cutoff-graph complexes; '
                             'auto-named from connectivity settings if omitted')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--no-std', action='store_true',
                        help='Skip writing ensemble std columns')

    args = parser.parse_args()

    model_types = ['SGN2', 'EGNX'] if args.model_type == 'both' else [args.model_type]

    # ------------------------------------------------------------------
    # 1. Discover all ensemble members for the requested architectures
    # ------------------------------------------------------------------
    print("=" * 72)
    print("QTNet ensemble inference on QM9")
    print("=" * 72)

    all_members = {}  # model_type -> [(dir, cfg, atomic_stats)]
    for mt in model_types:
        print(f"\nDiscovering {mt} members...")
        members = _discover_members(args.ensemble_root, mt)
        _assert_consistent_connectivity(members)
        all_members[mt] = members

    # Use connectivity settings from the first member of the first type
    ref_cfg = all_members[model_types[0]][0][1]

    # ------------------------------------------------------------------
    # 2. Load QM9 frame
    # ------------------------------------------------------------------
    print(f"\nLoading QM9: {args.qm9_pkl}")
    qm9 = pd.read_pickle(args.qm9_pkl)
    print(f"  {len(qm9)} molecules")

    atom_col = 'elements'
    element_to_idx = {e: i for i, e in enumerate(ALL_ELEMENTS)}

    # ------------------------------------------------------------------
    # 3. Precompute or load cutoff-graph complexes for QM9
    # ------------------------------------------------------------------
    max_neighbors = ref_cfg.get('max_neighbors', 5)
    max_neighbors_arg = max_neighbors if max_neighbors > 0 else None
    fully_connected = ref_cfg.get('fully_connected', False)
    cutoff = ref_cfg.get('cutoff', 8.0)
    max_dim = ref_cfg.get('max_dim', 1)

    if args.qm9_complexes_pkl is None:
        suffix = 'fc' if fully_connected else f'cutoff{cutoff}_max{max_neighbors}'
        args.qm9_complexes_pkl = os.path.join(
            REPO_ROOT, 'data_curation', 'molecular',
            f'precomputed_complexes_qm9_{suffix}.pkl',
        )

    if os.path.exists(args.qm9_complexes_pkl):
        print(f"\nLoading QM9 complexes from {args.qm9_complexes_pkl}")
        with open(args.qm9_complexes_pkl, 'rb') as f:
            qm9_complexes = pickle.load(f)
        print(f"  Loaded {len(qm9_complexes)} complexes")
    else:
        print(f"\nPrecomputing QM9 cutoff-graph complexes...")
        print(f"  cutoff={cutoff}  max_neighbors={max_neighbors}  "
              f"fully_connected={fully_connected}  max_dim={max_dim}")
        qm9_complexes = precompute_complexes(
            qm9,
            element_to_idx=element_to_idx,
            cutoff=cutoff,
            max_neighbors=max_neighbors_arg,
            fully_connected=fully_connected,
            max_dim=max_dim,
            output_path=args.qm9_complexes_pkl,
            verbose=True,
        )

    # Restrict QM9 to rows that have a valid complex
    valid_indices = set(qm9_complexes.keys())
    n_before = len(qm9)
    qm9 = qm9.loc[qm9.index.isin(valid_indices)]
    if len(qm9) < n_before:
        print(f"  Restricted to {len(qm9)} molecules with valid complexes "
              f"(dropped {n_before - len(qm9)})")

    # ------------------------------------------------------------------
    # 4. Build batches (no targets — inference only)
    # ------------------------------------------------------------------
    print("\nBuilding padded batches (no targets)...")
    batches = prepare_padded_batches(
        qm9_complexes, qm9,
        target_columns=[],
        batch_size=args.batch_size,
        verbose=True,
        as_numpy=True,
    )
    print(f"  {len(batches)} batches")

    # ------------------------------------------------------------------
    # 5. Run each ensemble member; collect denormalised predictions
    # ------------------------------------------------------------------
    # member_preds: list of (n_molecules, n_atoms_i, 10) per member.
    # Since molecules have varying atom counts, we store a list-of-lists.
    all_member_preds = []  # outer: members, inner: per-molecule (n_atoms,10)

    for mt in model_types:
        for member_idx, (member_dir, cfg, atomic_stats) in enumerate(all_members[mt]):
            print(f"\n[{mt} / member {member_idx}] loading checkpoint...")
            model = _create_model(cfg['model_class'], cfg['model_kwargs'],
                                  seed=cfg['seed'])
            best_path = os.path.join(member_dir, 'checkpoints', 'model_best_so_far')
            if not os.path.exists(best_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {best_path}\n"
                    f"Train this member first."
                )
            load_checkpoint(model, best_path)

            print(f"  Running inference on {len(qm9)} molecules...")
            mol_preds = _run_member_inference(
                model, batches, qm9, atomic_stats, atom_col,
            )
            all_member_preds.append(mol_preds)
            print(f"  Done ({len(mol_preds)} molecules)")

    n_members = len(all_member_preds)
    print(f"\nAveraging predictions across {n_members} ensemble member(s)...")

    # ------------------------------------------------------------------
    # 6. Average (and optionally std) across members, then write output
    # ------------------------------------------------------------------
    out = qm9.copy()

    # Initialise accumulator columns as object dtype (lists)
    for col in TARGET_COLUMNS:
        out[col] = None
        out[col] = out[col].astype(object)
    if n_members > 1 and not args.no_std:
        for col in TARGET_COLUMNS:
            out[f'{col}_std'] = None
            out[f'{col}_std'] = out[f'{col}_std'].astype(object)

    for i, idx in enumerate(qm9.index):
        # Stack member predictions for this molecule: (n_members, n_atoms, 10)
        mol_stack = np.stack(
            [all_member_preds[m][i] for m in range(n_members)],
            axis=0,
        )  # (n_members, n_atoms, 10)

        mean_pred = mol_stack.mean(axis=0)  # (n_atoms, 10)

        for ci, col in enumerate(TARGET_COLUMNS):
            out.at[idx, col] = mean_pred[:, ci].tolist()

        if n_members > 1 and not args.no_std:
            std_pred = mol_stack.std(axis=0)  # (n_atoms, 10)
            for ci, col in enumerate(TARGET_COLUMNS):
                out.at[idx, f'{col}_std'] = std_pred[:, ci].tolist()

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output_pkl)), exist_ok=True)
    out.to_pickle(args.output_pkl)
    print(f"\nSaved {len(out)} molecules to {args.output_pkl}")
    print(f"  New columns: {TARGET_COLUMNS}")
    if n_members > 1 and not args.no_std:
        print(f"  Std columns: {[c + '_std' for c in TARGET_COLUMNS]}")
    print("=" * 72)
    print("Done.")


if __name__ == '__main__':
    main()
