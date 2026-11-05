#!/usr/bin/env python3
"""
Environment-holdout molecular training (GeoScalarGNNMolecular, blind / informed).

Trains one model for one (repeat, cv_fold, fraction, variant) of the 5×5
chemotype × atomic-environment cross-validation (``data_utils.
create_env_chemotype_cv_splits``) on AIMEl, and writes predictions on the four
test panels of that fold.

The fold holds out TWO sets at once:
  * chemotype group ``cv_fold`` of repeat ``repeat``'s partition (Morgan-FP
    chemotypes, ``fp_cluster``), and
  * environment fold ``(repeat + cv_fold) % 5`` of the whole-QM9 6-Bohr
    atomic-environment classification — every molecule carrying one of its
    labels leaves the training pool.
Their 2×2 cross gives the panels, all scored:
    A  held-out chemotype, seen environments
    B  held-out chemotype, reserved environments
    C  seen chemotype,     reserved environments
    D  seen chemotype,     seen environments (in-distribution reference)
The training pool is capped at the same size in every fold, so all 25 folds
train on the same number of molecules at a given fraction.

Variants differ ONLY in the 10 per-atom QT-Net inputs (N, LI, Mu, Q):
  blind     one-hot(Z) + geometry
  informed  one-hot(Z) + geometry + per-element-normalised QTAIM features
Both are trained with a fresh Haar-uniform SO(3) rotation of the training
batches every epoch (the geometry filters read frame-dependent pos / G, and the
informed inputs Mu / Q are frame-dependent).

Outputs (``experiments/molecular/env_holdout/<variant>/fold_<5*repeat+cv_fold>/frac_<f>/``,
the layout of the paper's ``experiments/molecular/<variant>/`` runs; only config.json is
committed, everything else stays next to it):
  config.json       every setting, seed, the target and atomic-feature
                    normalisation statistics (fit on train only), sizes,
                    held-out chemotypes and environment labels, results
  train_ids.json    QM9 ids of the training and validation molecules
  val_preds.pkl / test_preds.pkl
                    ``index`` (QM9 id), ``panel`` (test only), pred_/target_
                    for alpha, gap, U0, Cv in physical units
  checkpoints/, loss/

Usage:
    python train_molecular_env_holdout.py --repeat 0 --cv-fold 0 --fraction 1.0 --variant blind
    python train_molecular_env_holdout.py --repeat 2 --cv-fold 4 --fraction 0.1 --variant informed --resume
"""

import argparse
import functools
import json
import os
import pickle
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

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

import flax.nnx as nnx

from qtnet.data_utils import (
    MOLECULAR_PROPERTIES_PRED,
    create_env_chemotype_cv_splits,
    env_fold_masks,
)
from qtnet.jax_models.atom_feature_norm import (
    ELEMENTS,
    atom_feature_norm_stats,
    normalize_atom_feature_batches,
)
from qtnet.jax_models.models_molecular_geo import GeoScalarGNNMolecular
from qtnet.jax_models.representations import prepare_padded_batches
from qtnet.jax_models.train_utils import (
    count_parameters,
    get_lr_scale,
    load_checkpoint,
    make_optimizer,
    train_molecular_multitask_augmented,
)

MOLECULAR_PROPERTIES = list(MOLECULAR_PROPERTIES_PRED)   # ['alpha', 'gap', 'U0', 'Cv']
VARIANTS = ('blind', 'informed')
FRACTIONS = (0.01, 0.05, 0.1, 1.0)   # the paper's molecular fractions
MODEL_SEED_BASE = 900000        # model-init seeds: disjoint from split seeds
ROT_SEED_BASE = 700000          # SO(3) rotation seeds: a third disjoint range

MOLECULAR_DIR = os.path.join(REPO_ROOT, 'data_curation', 'molecular')
CLUSTER_DIR = os.path.join(REPO_ROOT, 'data_curation', 'atomic', 'cluster_analysis')

# Settings whose change makes an existing checkpoint belong to a different
# experiment: a --resume onto such a checkpoint is refused.
RESUME_IDENTITY = ('variant', 'repeat', 'cv_fold', 'fold', 'fraction', 'model_seed',
                   'rotation_seed', 'rotate_every', 'batch_size', 'model_kwargs',
                   'n_train', 'n_val', 'n_test', 'complexes_cutoff',
                   'complexes_max_neighbors', 'env_fold', 'pool_size')


def _frac_idx(fraction: float) -> int:
    return FRACTIONS.index(fraction) if fraction in FRACTIONS else int(round(fraction * 100))


def model_seed_for(variant: str, fold: int, fraction: float) -> int:
    return MODEL_SEED_BASE + VARIANTS.index(variant) * 10000 + fold * 100 + _frac_idx(fraction)


def rotation_seed_for(variant: str, fold: int, fraction: float) -> int:
    return ROT_SEED_BASE + VARIANTS.index(variant) * 10000 + fold * 100 + _frac_idx(fraction)


def _git_commit() -> str:
    try:
        return subprocess.run(['git', '-C', REPO_ROOT, 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_frame(pkl_file: str, chemotypes_file: str) -> pd.DataFrame:
    """AIMEl frame, positional index, with the Morgan-FP chemotype joined on the QM9 id."""
    df = pd.read_pickle(pkl_file).reset_index(drop=True)
    df['index'] = df['index'].astype(np.int64)
    chem = pd.read_pickle(chemotypes_file).set_index('index')['fp_cluster']
    missing = ~df['index'].isin(chem.index)
    if missing.any():
        raise KeyError(f"{int(missing.sum())} molecules have no chemotype label")
    df['fp_cluster'] = chem.loc[df['index']].to_numpy()
    return df


def load_env_folds(env_labels_file: str, env_folds_file: str, qm9_index):
    labels = pd.read_pickle(env_labels_file).set_index('index')['atom_env_labels_6b']
    with open(env_folds_file) as f:
        spec = json.load(f)
    folds = spec['folds']
    masks = env_fold_masks(qm9_index, labels, [f_['labels'] for f_ in folds])
    return folds, masks


def load_complexes(path: str, df: pd.DataFrame, rows: np.ndarray):
    """Complexes of ``rows`` (positional), re-keyed from QM9 id to row position."""
    with open(path, 'rb') as f:
        blob = pickle.load(f)
    cx = blob['complexes']
    ids = df['index'].to_numpy()[rows]
    missing = [int(q) for q in ids if int(q) not in cx]
    if missing:
        raise KeyError(f"{len(missing)} molecules missing from {path} (e.g. {missing[:3]})")
    out = {int(r): cx[int(q)] for r, q in zip(rows, ids)}
    meta = {k: v for k, v in blob.items() if k != 'complexes'}
    return out, meta


def norm_stats_from_train(df_train: pd.DataFrame) -> Dict:
    stats = {}
    for prop in MOLECULAR_PROPERTIES:
        vals = df_train[prop].to_numpy(dtype=np.float64)
        std = float(vals.std())
        stats[prop] = {'mean': float(vals.mean()), 'std': std if std > 1e-8 else 1.0}
    return stats


def normalize_targets(df: pd.DataFrame, stats: Dict) -> np.ndarray:
    out = np.zeros((len(df), len(MOLECULAR_PROPERTIES)), dtype=np.float32)
    for i, prop in enumerate(MOLECULAR_PROPERTIES):
        out[:, i] = (df[prop].to_numpy(dtype=np.float32) - stats[prop]['mean']) / stats[prop]['std']
    return out


def make_batches(complexes, df_sub, batch_size, feature_stats):
    batches = prepare_padded_batches(complexes, df_sub, target_columns=[],
                                     batch_size=batch_size, verbose=False, as_numpy=True)
    if feature_stats is not None:
        normalize_atom_feature_batches(batches, feature_stats, ELEMENTS)
    return batches


def create_model(model_kwargs: Dict, use_atom_features: bool, seed: int,
                 num_species: int) -> nnx.Module:
    return GeoScalarGNNMolecular(
        num_species=num_species, num_outputs=len(MOLECULAR_PROPERTIES),
        use_atom_features=use_atom_features, rngs=nnx.Rngs(seed), **model_kwargs)


def collect_predictions(model, batches, df_sub: pd.DataFrame, stats: Dict,
                        panel: pd.Series = None) -> pd.DataFrame:
    """Denormalised predictions; rows follow ``df_sub`` order (the batch order)."""

    @functools.partial(nnx.jit, static_argnames=('num_graphs',))
    def _predict(model, batch, num_graphs):
        return model(batch, graph_idx=batch.cochain_batches[0].owner_cochains,
                     num_graphs=num_graphs)

    preds: List[np.ndarray] = []
    for b in batches:
        n = int(b.num_complexes[0])
        preds.append(np.asarray(_predict(model, b, n)['predictions'])[:n])
    preds = np.concatenate(preds, axis=0)
    if len(preds) != len(df_sub):
        raise AssertionError(f"{len(preds)} predictions for {len(df_sub)} molecules")
    out = pd.DataFrame({'index': df_sub['index'].to_numpy()})
    if panel is not None:
        out['panel'] = panel.loc[df_sub.index].to_numpy()
    for i, prop in enumerate(MOLECULAR_PROPERTIES):
        out[f'pred_{prop}'] = preds[:, i] * stats[prop]['std'] + stats[prop]['mean']
        out[f'target_{prop}'] = df_sub[prop].to_numpy(dtype=float)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repeat', type=int, required=True)
    p.add_argument('--cv-fold', type=int, required=True)
    p.add_argument('--fraction', type=float, required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    # data
    p.add_argument('--pkl-file', default=os.path.join(MOLECULAR_DIR, 'aimel_clustered_molecular.pkl'))
    p.add_argument('--complexes', default=os.path.join(MOLECULAR_DIR, 'precomputed_geo_aimel.pkl'))
    p.add_argument('--chemotypes', default=os.path.join(MOLECULAR_DIR, 'qm9_chemotypes.pkl.gz'))
    p.add_argument('--env-labels', default=os.path.join(CLUSTER_DIR, 'qm9_env_labels_6b.pkl.gz'))
    p.add_argument('--env-folds', default=os.path.join(CLUSTER_DIR, 'qm9_env_folds_6b.json'))
    p.add_argument('--hp-file', default=os.path.join(MOLECULAR_DIR, 'GeoScalarGNNMolecular_hp.json'))
    # Same layout as the paper's runs (experiments/molecular/<variant>/fold_<n>/frac_<f>/):
    # config.json is committed; predictions, checkpoints and losses sit next to it, uncommitted.
    p.add_argument('--output-dir', default=os.path.join(REPO_ROOT, 'experiments', 'molecular', 'env_holdout'))
    # CV
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--n-repeats', type=int, default=5)
    p.add_argument('--base-seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--id-ref-frac', type=float, default=0.03)
    p.add_argument('--pool-cap', default='min', help="'min' (default), an int, or 'none'")
    # training
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--rotate-every', type=int, default=1)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--print-every', type=int, default=10)
    p.add_argument('--resume', action='store_true',
                   help='continue from checkpoints/train_state (refused if the stored config differs)')
    args = p.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise ValueError(f"--fraction must be in (0, 1]; got {args.fraction}")
    fold = args.repeat * args.n_splits + args.cv_fold
    use_atom_features = args.variant == 'informed'
    pool_cap = None if str(args.pool_cap).lower() == 'none' else (
        'min' if args.pool_cap == 'min' else int(args.pool_cap))

    print('=' * 80)
    print(f"GeoScalarGNNMolecular ({args.variant})  repeat={args.repeat} cv_fold={args.cv_fold} "
          f"(fold {fold})  fraction={args.fraction}", flush=True)
    print('=' * 80)

    # ---- data + split -------------------------------------------------
    df = load_frame(args.pkl_file, args.chemotypes)
    env_folds, masks = load_env_folds(args.env_labels, args.env_folds, df['index'])
    split = None
    for s in create_env_chemotype_cv_splits(
            df, masks, 'fp_cluster', n_splits=args.n_splits, n_repeats=args.n_repeats,
            base_seed=args.base_seed, training_fractions=[args.fraction],
            val_fraction=args.val_fraction, id_ref_frac=args.id_ref_frac, pool_cap=pool_cap):
        if s['fold'] == fold:
            split = s
            break
    if split is None:
        raise ValueError(f"fold {fold} not found")
    panel = pd.Series('', index=df.index, dtype=object)
    for name, idx in split['panels'].items():
        panel.iloc[idx] = name
    train_idx, val_idx, test_idx = split['train_idx'], split['val_idx'], split['test_idx']
    df_train, df_val, df_test = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    env_spec = env_folds[split['env_fold']]
    heldout_chemotypes = sorted(set(df['fp_cluster'].to_numpy()[
        np.concatenate([split['panels']['A'], split['panels']['B']])]))
    print(f"  env fold {split['env_fold']} (seed {env_spec.get('seed')}: {env_spec['labels']}) | "
          f"{len(heldout_chemotypes)} held-out chemotypes")
    print("  panels " + " ".join(f"{k}={len(v):,}" for k, v in split['panels'].items())
          + f" | pool {split['pool_size']:,} | train {len(train_idx):,} val {len(val_idx):,}", flush=True)

    # ---- hyperparameters, seeds, statistics ---------------------------
    with open(args.hp_file) as f:
        hp = json.load(f)
    model_kwargs, lr, wd = dict(hp['model_kwargs']), float(hp['lr']), float(hp['weight_decay'])
    model_seed = model_seed_for(args.variant, fold, args.fraction)
    rot_seed = rotation_seed_for(args.variant, fold, args.fraction)
    norm_stats = norm_stats_from_train(df_train)
    feature_stats = atom_feature_norm_stats(df_train, ELEMENTS) if use_atom_features else None

    # ---- complexes + batches ------------------------------------------
    rows = np.unique(np.concatenate([train_idx, val_idx, test_idx]))
    complexes, cx_meta = load_complexes(args.complexes, df, rows)
    num_species = len(cx_meta['element_to_idx'])

    run_dir = os.path.join(args.output_dir, args.variant, f'fold_{fold}', f'frac_{args.fraction}')
    ckpt_dir = os.path.abspath(os.path.join(run_dir, 'checkpoints'))
    loss_dir = os.path.abspath(os.path.join(run_dir, 'loss'))
    os.makedirs(run_dir, exist_ok=True)

    config = {
        'model_class': 'GeoScalarGNNMolecular',
        'variant': args.variant,
        'use_atom_features': use_atom_features,
        'repeat': args.repeat, 'cv_fold': args.cv_fold, 'fold': fold, 'fraction': args.fraction,
        'n_splits': args.n_splits, 'n_repeats': args.n_repeats, 'base_seed': args.base_seed,
        'val_fraction': args.val_fraction, 'id_ref_frac': args.id_ref_frac,
        'pool_cap': args.pool_cap, 'pool_size': split['pool_size'],
        'group_col': 'fp_cluster', 'heldout_chemotypes': heldout_chemotypes,
        'env_fold': split['env_fold'], 'env_fold_seed': env_spec.get('seed'),
        'env_fold_labels': env_spec['labels'],
        'panel_sizes': {k: int(len(v)) for k, v in split['panels'].items()},
        'model_kwargs': model_kwargs, 'learning_rate': lr, 'weight_decay': wd,
        'hp_source': hp.get('source'),
        'model_seed': model_seed, 'rotation_seed': rot_seed, 'rotate_every': args.rotate_every,
        'augment': 'SO(3) Haar, augment_molecular_batches',
        'epochs': args.epochs, 'batch_size': args.batch_size, 'eval_batch_size': args.eval_batch_size,
        'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)), 'n_test': int(len(test_idx)),
        'target_columns': MOLECULAR_PROPERTIES,
        'complexes': os.path.basename(args.complexes),
        'complexes_cutoff': cx_meta.get('cutoff'), 'complexes_max_neighbors': cx_meta.get('max_neighbors'),
        'atom_feature_keys': cx_meta.get('atom_feature_keys') if use_atom_features else [],
        'norm_stats': norm_stats, 'atom_feature_norm_stats': feature_stats,
        'code_commit': _git_commit(),
    }
    cfg_path = os.path.join(run_dir, 'config.json')
    if args.resume and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            prev = json.load(f)
        diff = {k: (prev.get(k), config[k]) for k in RESUME_IDENTITY if prev.get(k) != config[k]}
        if diff:
            raise SystemExit(
                f"REFUSING TO RESUME {run_dir}: the checkpoint was written under different "
                "settings, so continuing would silently mix two experiments.\n"
                + "\n".join(f"  {k}: checkpoint={a!r} now={b!r}" for k, (a, b) in diff.items()))
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, 'train_ids.json'), 'w') as f:
        json.dump({'train': [int(i) for i in df_train['index']],
                   'val': [int(i) for i in df_val['index']]}, f)

    train_batches = make_batches(complexes, df_train, args.batch_size, feature_stats)
    val_batches = make_batches(complexes, df_val, args.eval_batch_size, feature_stats)
    train_targets = normalize_targets(df_train, norm_stats)
    val_targets = normalize_targets(df_val, norm_stats)

    # ---- model + optimizer --------------------------------------------
    model = create_model(model_kwargs, use_atom_features, model_seed, num_species)
    print(f"  model: {count_parameters(model):,} params | lr {lr:.2e} wd {wd:.2e} | {model_kwargs}",
          flush=True)
    tx = make_optimizer(lr=lr, weight_decay=wd, use_reduce_on_plateau=True,
                        plateau_accumulation_size=len(train_batches))
    # The loop calls optimizer.update(grads, value=...): flax >= 0.11 keeps that
    # signature only on ModelAndOptimizer, older flax has it on nnx.Optimizer.
    optimizer = getattr(nnx, 'ModelAndOptimizer', nnx.Optimizer)(model, tx)

    # ---- train ----------------------------------------------------------
    t0 = time.perf_counter()
    train_hist, val_hist = train_molecular_multitask_augmented(
        model, optimizer, train_batches, train_targets, val_batches, val_targets,
        target_names=MOLECULAR_PROPERTIES, epochs=args.epochs, rotation_seed=rot_seed,
        rotate_every=args.rotate_every, save_every=args.save_every,
        checkpoint_dir=ckpt_dir, loss_dir=loss_dir, verbose=True,
        print_every=args.print_every, resume=args.resume)
    seconds = time.perf_counter() - t0

    # ---- predict from the best-validation checkpoint -------------------
    best = os.path.join(ckpt_dir, 'model_best_so_far')
    predicted_from = 'final'
    if os.path.exists(best):
        model = load_checkpoint(model, best)
        predicted_from = 'best_checkpoint'
    config['results'] = {
        'best_val_total': float(min(val_hist['total'])) if val_hist and val_hist['total'] else None,
        'final_val_total': float(val_hist['total'][-1]) if val_hist and val_hist['total'] else None,
        'final_train_total': float(train_hist['total'][-1]) if train_hist['total'] else None,
        'best_epoch': int(np.argmin(val_hist['total'])) + 1 if val_hist and val_hist['total'] else None,
        'final_lr_scale': float(get_lr_scale(optimizer)),
        'seconds': round(seconds, 1),
        'predicted_from': predicted_from,
    }
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=2)

    val_batches = make_batches(complexes, df_val, args.eval_batch_size, feature_stats)
    test_batches = make_batches(complexes, df_test, args.eval_batch_size, feature_stats)
    collect_predictions(model, val_batches, df_val, norm_stats).to_pickle(
        os.path.join(run_dir, 'val_preds.pkl'))
    test_preds = collect_predictions(model, test_batches, df_test, norm_stats, panel)
    test_preds.to_pickle(os.path.join(run_dir, 'test_preds.pkl'))
    print(f"  wrote {len(test_preds):,} test predictions "
          f"({test_preds['panel'].value_counts().to_dict()}) → {run_dir}", flush=True)


if __name__ == '__main__':
    main()
