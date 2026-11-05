#!/usr/bin/env python3
"""
Deploy the environment-holdout models on the QM9 remainder — no retraining.

Every run written by ``scripts/molecular/jax/train_molecular_env_holdout.py``
(trained on AIMEl with ground-truth QTAIM features) predicts the 98,096 QM9
molecules outside AIMEl.  The blind variant needs only geometry; the informed
variant reads the per-atom N, LI, Mu, Q inferred by the QT-Net ensemble
(``qm9_inferred.pkl``, from ``infer_QTAIM_QM9.py``), normalised with the
statistics STORED in the run's config.json (fit on that run's AIMEl training
split) — never re-fit on the remainder, which would move the inputs out from
under the model.

Each prediction row carries, relative to THAT run's actual training set:
  chemotype      'seen' (Morgan-FP chemotype in the run's training set),
                 'heldout' (one of the fold's held-out chemotypes) or
                 'never_in_aimel' (a chemotype with no AIMEl molecule)
  reserved_env   the molecule carries an environment of the fold's reserved
                 environment fold (zero training atoms by construction)
  n_unseen_env_atoms, min_env_train_atoms
                 labelled atoms whose environment has zero training atoms, and
                 the training support of the molecule's rarest environment
  remainder_only_env
                 carries one of the environments with no AIMEl atom at all
The same remainder molecule is thus predicted by the 5 folds that reserved its
environments and the 20 that did not — the within-molecule environment contrast.

Output: ``remainder_preds.pkl`` and ``remainder_meta.json`` written INTO each run directory,
next to its config.json and test_preds.pkl (uncommitted, like the other predictions).

Usage:
    python predict_env_holdout_from_inferred.py --variant blind
    python predict_env_holdout_from_inferred.py --variant informed --folds 0:1:2 --fractions 1.0
"""

import argparse
import functools
import glob
import json
import os
import pickle
import sys
import time
from collections import Counter

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
            return os.path.abspath(os.path.join(start_dir, '..', '..'))
        cur = parent


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
for _root, _dirs, _ in os.walk(REPO_ROOT):
    if 'qtnet' in _dirs:
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break

import flax.nnx as nnx

from qtnet.jax_models.atom_feature_norm import ELEMENTS, normalize_atom_feature_batches
from qtnet.jax_models.models_molecular_geo import ATOM_FEATURE_KEYS, GeoScalarGNNMolecular
from qtnet.jax_models.representations import prepare_padded_batches
from qtnet.jax_models.train_utils import load_checkpoint

MOLECULAR_DIR = os.path.join(REPO_ROOT, 'data_curation', 'molecular')
CLUSTER_DIR = os.path.join(REPO_ROOT, 'data_curation', 'atomic', 'cluster_analysis')


def molecule_env_table(qm9_index, labels: pd.Series) -> list:
    """Per molecule: its labelled (non-noise) atom environments, in atom order."""
    return [[l for l in labels.loc[int(q)] if not l.endswith('_-1')] for q in qm9_index]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', choices=('blind', 'informed'), required=True)
    ap.add_argument('--folds', default='', help='colon-separated global folds (default: all found)')
    ap.add_argument('--fractions', default='', help='colon-separated fractions (default: all found)')
    ap.add_argument('--runs-dir', default=os.path.join(REPO_ROOT, 'experiments', 'molecular', 'env_holdout'))
    ap.add_argument('--frame', default=os.path.join(MOLECULAR_DIR, 'qm9_inferred.pkl'))
    ap.add_argument('--complexes', default=os.path.join(MOLECULAR_DIR, 'precomputed_geo_qm9_inferred.pkl'))
    ap.add_argument('--aimel', default=os.path.join(MOLECULAR_DIR, 'aimel_clustered_molecular.pkl'))
    ap.add_argument('--chemotypes', default=os.path.join(MOLECULAR_DIR, 'qm9_chemotypes.pkl.gz'))
    ap.add_argument('--env-labels', default=os.path.join(CLUSTER_DIR, 'qm9_env_labels_6b.pkl.gz'))
    ap.add_argument('--env-folds', default=os.path.join(CLUSTER_DIR, 'qm9_env_folds_6b.json'))
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--skip-existing', action='store_true')
    a = ap.parse_args()

    runs = sorted(glob.glob(os.path.join(a.runs_dir, a.variant, 'fold_*', 'frac_*', 'config.json')))
    want_f = {int(x) for x in a.folds.split(':') if x}
    want_x = {float(x) for x in a.fractions.split(':') if x}
    cfgs = []
    for p in runs:
        cfg = json.load(open(p))
        if (want_f and cfg['fold'] not in want_f) or (want_x and cfg['fraction'] not in want_x):
            continue
        if not os.path.exists(os.path.join(os.path.dirname(p), 'checkpoints', 'model_best_so_far')):
            print(f"  skip {p}: no model_best_so_far")
            continue
        cfgs.append((os.path.dirname(p), cfg))
    print(f"[deploy {a.variant}] {len(cfgs)} runs", flush=True)
    if not cfgs:
        return

    # ---- remainder frame, strata inputs ---------------------------------
    df = pd.read_pickle(a.frame).reset_index(drop=True)
    df['index'] = df['index'].astype(np.int64)
    chem = pd.read_pickle(a.chemotypes).set_index('index')['fp_cluster']
    df['fp_cluster'] = chem.loc[df['index']].to_numpy()
    aimel_ids = pd.read_pickle(a.aimel)['index'].astype(np.int64).to_numpy()
    aimel_chem = set(chem.loc[aimel_ids])
    labels = pd.read_pickle(a.env_labels).set_index('index')['atom_env_labels_6b']
    lab_table = pd.read_pickle(a.env_labels)
    aimel_label_atoms = Counter(l for L in lab_table.loc[lab_table.in_aimel, 'atom_env_labels_6b'] for l in L)
    env_folds = json.load(open(a.env_folds))['folds']
    mol_envs = molecule_env_table(df['index'], labels)
    remainder_only = {l for L in mol_envs for l in L if aimel_label_atoms[l] == 0}
    df['remainder_only_env'] = [bool(remainder_only.intersection(L)) for L in mol_envs]
    targets = [c for c in cfgs[0][1]['target_columns']]

    # ---- complexes + raw batches, built once ------------------------------
    with open(a.complexes, 'rb') as f:
        blob = pickle.load(f)
    cx = blob['complexes']
    missing = [int(q) for q in df['index'] if int(q) not in cx]
    if missing:
        raise KeyError(f"{len(missing)} remainder molecules missing from {a.complexes}")
    complexes = {i: cx[int(q)] for i, q in zip(df.index, df['index'])}
    graph = (blob.get('cutoff'), blob.get('max_neighbors'))
    del blob, cx
    t0 = time.perf_counter()
    batches = prepare_padded_batches(complexes, df, target_columns=[], batch_size=a.batch_size,
                                     verbose=False, as_numpy=True)
    raw = ([{k: b.cochain_batches[0].static[k].copy() for k in ATOM_FEATURE_KEYS} for b in batches]
           if a.variant == 'informed' else None)
    print(f"  {len(df):,} molecules in {len(batches)} batches ({time.perf_counter() - t0:.0f}s)", flush=True)

    @functools.partial(nnx.jit, static_argnames=('num_graphs',))
    def _predict(model, batch, num_graphs):
        return model(batch, graph_idx=batch.cochain_batches[0].owner_cochains, num_graphs=num_graphs)

    for run_dir, cfg in cfgs:
        rel = os.path.relpath(run_dir, a.runs_dir)
        out_dir = run_dir
        out_p = os.path.join(out_dir, 'remainder_preds.pkl')
        if a.skip_existing and os.path.exists(out_p):
            continue
        if (cfg.get('complexes_cutoff'), cfg.get('complexes_max_neighbors')) != graph:
            raise ValueError(f"{rel}: trained on graphs {cfg.get('complexes_cutoff')} Bohr / "
                             f"{cfg.get('complexes_max_neighbors')} NN, deployment complexes are {graph}")
        # restore raw features, then apply THIS run's stored statistics
        if raw is not None:
            for b, r in zip(batches, raw):
                for k, v in r.items():
                    b.cochain_batches[0].static[k] = v.copy()
            normalize_atom_feature_batches(batches, cfg['atom_feature_norm_stats'], ELEMENTS)
        model = GeoScalarGNNMolecular(num_species=len(ELEMENTS), num_outputs=len(targets),
                                      use_atom_features=cfg['use_atom_features'],
                                      rngs=nnx.Rngs(cfg['model_seed']), **cfg['model_kwargs'])
        model = load_checkpoint(model, os.path.abspath(os.path.join(run_dir, 'checkpoints', 'model_best_so_far')))
        preds = np.concatenate([np.asarray(_predict(model, b, int(b.num_complexes[0]))['predictions'])
                                [:int(b.num_complexes[0])] for b in batches])
        if len(preds) != len(df):
            raise AssertionError(f"{len(preds)} predictions for {len(df)} molecules")

        # strata relative to this run's training set
        with open(os.path.join(run_dir, 'train_ids.json')) as f:
            train_ids = json.load(f)['train']
        train_counts = Counter(l for q in train_ids for l in labels.loc[int(q)] if not l.endswith('_-1'))
        train_chem = set(chem.loc[np.asarray(train_ids, dtype=np.int64)])
        held = set(cfg['heldout_chemotypes'])
        reserved = set(env_folds[cfg['env_fold']]['labels'])
        out = pd.DataFrame({'index': df['index'].to_numpy(), 'fp_cluster': df['fp_cluster'].to_numpy()})
        out['chemotype'] = np.where(out['fp_cluster'].isin(train_chem), 'seen',
                                    np.where(out['fp_cluster'].isin(held), 'heldout',
                                             np.where(~out['fp_cluster'].isin(aimel_chem), 'never_in_aimel', 'other')))
        out['reserved_env'] = [bool(reserved.intersection(L)) for L in mol_envs]
        out['n_unseen_env_atoms'] = [sum(train_counts[l] == 0 for l in L) for L in mol_envs]
        out['min_env_train_atoms'] = [min((train_counts[l] for l in L), default=-1) for L in mol_envs]
        out['remainder_only_env'] = df['remainder_only_env'].to_numpy()
        for i, prop in enumerate(targets):
            out[f'pred_{prop}'] = preds[:, i] * cfg['norm_stats'][prop]['std'] + cfg['norm_stats'][prop]['mean']
            out[f'target_{prop}'] = df[prop].to_numpy(dtype=float)
        os.makedirs(out_dir, exist_ok=True)
        out.to_pickle(out_p)
        meta = {k: cfg[k] for k in ('variant', 'repeat', 'cv_fold', 'fold', 'fraction', 'env_fold', 'n_train')}
        meta.update({'n_molecules': int(len(out)), 'chemotype_counts': out['chemotype'].value_counts().to_dict(),
                     'n_reserved_env': int(out['reserved_env'].sum()),
                     'features': 'QT-Net ensemble inferred (qm9_inferred.pkl)' if cfg['use_atom_features'] else 'none'})
        with open(os.path.join(out_dir, 'remainder_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  {rel}: {len(out):,} predictions | {meta['chemotype_counts']} | reserved-env {meta['n_reserved_env']:,}",
              flush=True)


if __name__ == '__main__':
    main()
