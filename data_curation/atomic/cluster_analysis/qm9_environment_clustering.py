#!/usr/bin/env python3
"""
Whole-QM9 classification of atomic environments at 6 Bohr (the environment
axis of the environment-holdout study).

QT-Net's recipe (SOAP → per-element PCA → HDBSCAN, ``atomic_env_split.py``)
applied to ALL 128,908 QM9 molecules (AIMEl ∪ remainder, 2,333,341 atoms) at
a 6-Bohr SOAP cutoff:

  1. SOAP (dscribe; n_max 8, l_max 6, sigma 0.7, species H/C/N/O, r_cut 6 Bohr)
     for every atom, streamed to per-element float32 memmaps;
  2. per-element PCA to 0.99 explained variance (exact, from a streamed
     covariance): H 48 / C 76 / N 67 / O 52 dimensions;
  3. per-element HDBSCAN (euclidean, eom) with the committed parameters
     H 2500/100, C 850/50, N 300/5, O 250/10 (min_cluster_size/min_samples),
     chosen from sweeps for healthy cluster counts and noise:
     → 38 / 85 / 91 / 103 = 317 labels, noise 22.9 / 14.1 / 20.8 / 19.7 %.

Labels are ``<element>_<cluster>``; HDBSCAN noise is ``<element>_-1``.  Molecules
are processed in the order of ``qm9_classification_order.json.gz`` (it fixes
the cluster numbering) with the QM9 conformer of the 23 AIMEl molecules whose
QTAIM structure is a different conformer (``load_qm9_classification_frame``).

Every expensive step is cached in ``--cache-dir`` (SOAP memmaps, PCA, one label
file per (element, mcs, ms, dims) HDBSCAN fit — the C fit alone takes ~8.6 h),
so re-running only recomputes what is missing.

Output: ``qm9_env_labels_6b_rerun.pkl.gz`` — ``index`` (QM9 id), ``atom_env_labels_6b``
(per-atom label list, atom order of the frames), ``in_aimel``.  The shipped
``qm9_env_labels_6b.pkl.gz`` defines the splits and is not overwritten by default.

Usage:
    python qm9_environment_clustering.py --soap-only                # step 1, all elements
    python qm9_environment_clustering.py --sweep --elements N      # parameter sweep for one element
    python qm9_environment_clustering.py                           # commit: labels for all atoms
"""

import argparse
import os
import pickle
import sys
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
for _root, _dirs, _ in os.walk(REPO_ROOT):
    if 'qtnet' in _dirs:
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from qtnet.data_utils import load_qm9_classification_frame

ELEMENTS = ['H', 'C', 'N', 'O']
RADIUS_BOHR = 6.0
MOLECULAR_DIR = os.path.join(REPO_ROOT, 'data_curation', 'molecular')

# Committed HDBSCAN parameters (min_cluster_size, min_samples) at 6 Bohr / PCA 0.99.
COMMITTED = {'H': (2500, 100), 'C': (850, 50), 'N': (300, 5), 'O': (250, 10)}


def soap_params(radius: float) -> Dict:
    # r_cut in the units of the coordinates (Bohr); passed EXPLICITLY — the default
    # of atomic_env_split.SOAP_PARAMS is the paper's AIMEl-only setting.
    return {'r_cut': float(radius), 'n_max': 8, 'l_max': 6, 'sigma': 0.7,
            'species': ELEMENTS, 'periodic': False, 'sparse': False}


def stream_soap_to_memmaps(frame: pd.DataFrame, cache_dir: str, params: Dict) -> Dict:
    """SOAP for every atom → per-element float32 memmaps.  Returns
    ``meta[el] = {path, n_atoms, n_features, index}``; ``index`` lists
    (mol_pos, atom_idx) in memmap row order, ``mol_pos`` = row of ``frame``."""
    from ase import Atoms
    from atomic_env_split import compute_soap_descriptors

    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, 'soap_memmap_meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as fh:
            print(f"[cache] SOAP memmaps: {meta_path}")
            return pickle.load(fh)
    counts = {el: 0 for el in ELEMENTS}
    for symbols in frame['atom']:
        for s in symbols:
            counts[str(s)] += 1
    print('[soap] atoms per element:', {k: f'{v:,}' for k, v in counts.items()}, flush=True)
    writers, cursors, index, n_feat = {}, {el: 0 for el in ELEMENTS}, {el: [] for el in ELEMENTS}, None
    for mol_pos in range(len(frame)):
        row = frame.iloc[mol_pos]
        symbols = [str(s) for s in row['atom']]
        pos = np.column_stack([row['position_x'], row['position_y'], row['position_z']])
        soap = compute_soap_descriptors(Atoms(symbols=symbols, positions=pos), params)
        if n_feat is None:
            n_feat = soap.shape[1]
            writers = {el: np.memmap(os.path.join(cache_dir, f'soap_{el}.f32'), dtype=np.float32,
                                     mode='w+', shape=(counts[el], n_feat)) for el in ELEMENTS}
        for atom_idx, (el, vec) in enumerate(zip(symbols, soap)):
            writers[el][cursors[el]] = vec.astype(np.float32)
            index[el].append((mol_pos, atom_idx))
            cursors[el] += 1
        if mol_pos % 4000 == 0:
            print(f"[soap] {mol_pos + 1:,}/{len(frame):,} molecules", flush=True)
    for el in ELEMENTS:
        writers[el].flush()
    meta = {el: {'path': os.path.join(cache_dir, f'soap_{el}.f32'), 'n_atoms': counts[el],
                 'n_features': n_feat, 'index': index[el]} for el in ELEMENTS}
    with open(meta_path, 'wb') as fh:
        pickle.dump(meta, fh)
    return meta


def _memmap(m: Dict) -> np.memmap:
    return np.memmap(m['path'], dtype=np.float32, mode='r', shape=(m['n_atoms'], m['n_features']))


def pca_project(element: str, meta: Dict, cache_dir: str, variance: float = 0.99):
    """Exact PCA to ``variance`` from a streamed covariance (cached).
    Returns (X_pca, n_components, variance_retained_percent)."""
    cache = os.path.join(cache_dir, f'pca_{element}_v{variance}.pkl')
    mm = _memmap(meta[element])
    if os.path.exists(cache):
        with open(cache, 'rb') as fh:
            d = pickle.load(fh)
    else:
        n, dim = mm.shape
        s, ss = np.zeros(dim), np.zeros((dim, dim))
        for i in range(0, n, 20_000):
            X = np.asarray(mm[i:i + 20_000], dtype=np.float64)
            s += X.sum(0); ss += X.T @ X
        mean = s / n
        cov = (ss - n * np.outer(mean, mean)) / (n - 1)
        evals, evecs = np.linalg.eigh(cov)
        evals, evecs = np.clip(evals[::-1], 0, None), evecs[:, ::-1]
        ratio = np.cumsum(evals) / evals.sum()
        k = int(np.searchsorted(ratio, variance) + 1)
        d = {'mean': mean, 'W': evecs[:, :k], 'k': k, 'var_retained': float(ratio[k - 1] * 100)}
        with open(cache, 'wb') as fh:
            pickle.dump(d, fh)
    X_pca = np.empty((mm.shape[0], d['W'].shape[1]), dtype=np.float32)
    for i in range(0, mm.shape[0], 50_000):
        X_pca[i:i + 50_000] = ((np.asarray(mm[i:i + 50_000], dtype=np.float64) - d['mean']) @ d['W']).astype(np.float32)
    return X_pca, d['k'], d['var_retained']


def hdbscan_labels(X_pca: np.ndarray, mcs: int, ms: int, cache_dir: str, element: str) -> np.ndarray:
    """HDBSCAN (euclidean, eom), cached per (element, mcs, ms, dims)."""
    cache = os.path.join(cache_dir, f'labels_{element}_mcs{mcs}_ms{ms}_d{X_pca.shape[1]}_eom.npz')
    if os.path.exists(cache):
        print(f"[cache] {os.path.basename(cache)}", flush=True)
        return np.load(cache)['labels']
    import hdbscan
    labels = hdbscan.HDBSCAN(min_cluster_size=int(mcs), min_samples=int(ms), metric='euclidean',
                             cluster_selection_method='eom', core_dist_n_jobs=-1
                             ).fit_predict(X_pca.astype(np.float64))
    np.savez_compressed(cache, labels=labels)
    return labels


def label_stats(labels: np.ndarray) -> Dict:
    sizes = pd.Series(labels[labels != -1]).value_counts()
    return {'n_clusters': int(sizes.size), 'noise_frac': float(np.mean(labels == -1)),
            'max_cluster_frac': float(sizes.max() / len(labels)) if len(sizes) else 0.0}


def sweep(meta, cache_dir, elements, grid: List[tuple]):
    for el in elements:
        X, k, var = pca_project(el, meta, cache_dir)
        for mcs, ms in grid:
            st = label_stats(hdbscan_labels(X, mcs, ms, cache_dir, el))
            print(f"[sweep {el}] mcs={mcs} ms={ms} -> {st['n_clusters']} clusters, "
                  f"noise {100*st['noise_frac']:.1f} %, largest {100*st['max_cluster_frac']:.1f} %", flush=True)


def commit(frame, meta, cache_dir, elements, params: Dict[str, tuple], out_path: str):
    mol_labels = [np.array([''] * len(a), dtype=object) for a in frame['atom']]
    in_aimel = frame['in_aimel'].to_numpy()
    report = []
    for el in elements:
        X, k, var = pca_project(el, meta, cache_dir)
        mcs, ms = params[el]
        labels = hdbscan_labels(X, mcs, ms, cache_dir, el)
        st = label_stats(labels)
        print(f"[commit {el}] mcs={mcs} ms={ms} pca={k} ({var:.2f} %) -> {st['n_clusters']} clusters, "
              f"noise {100*st['noise_frac']:.1f} %", flush=True)
        idx = meta[el]['index']
        mol_pos = np.fromiter((t[0] for t in idx), dtype=np.int64, count=len(idx))
        atom_idx = np.fromiter((t[1] for t in idx), dtype=np.int64, count=len(idx))
        for mp, ai, lab in zip(mol_pos, atom_idx, labels):
            mol_labels[mp][ai] = f'{el}_{int(lab)}'
        atom_aim = in_aimel[mol_pos]
        for lab in sorted(set(int(l) for l in labels) - {-1}):
            m = labels == lab
            report.append({'label': f'{el}_{lab}', 'n_aimel': int((m & atom_aim).sum()),
                           'n_remainder': int((m & ~atom_aim).sum())})
    if any('' in L for L in mol_labels):
        raise AssertionError("some atoms received no label (element list incomplete?)")
    out = pd.DataFrame({'index': frame['index'].to_numpy(),
                        'atom_env_labels_6b': [list(L) for L in mol_labels],
                        'in_aimel': in_aimel})
    out.to_pickle(out_path)
    rep = pd.DataFrame(report)
    novel = rep[rep.n_aimel == 0].label.tolist()
    print(f"[commit] {len(rep)} labels; {len(rep) - len(novel)} with AIMEl atoms, "
          f"{len(novel)} only in the remainder: {novel}")
    print(f"[commit] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--aimel', default=os.path.join(MOLECULAR_DIR, 'aimel_clustered_molecular.pkl'))
    ap.add_argument('--remainder', default=os.path.join(MOLECULAR_DIR, 'qm9_filtered.pkl'))
    ap.add_argument('--order', default=os.path.join(MOLECULAR_DIR, 'qm9_classification_order.json.gz'))
    ap.add_argument('--cache-dir', default=os.path.join(SCRIPT_DIR, 'cache_6b'))
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'qm9_env_labels_6b_rerun.pkl.gz'),
                    help='output labels (default does NOT overwrite the shipped qm9_env_labels_6b.pkl.gz)')
    ap.add_argument('--elements', default=','.join(ELEMENTS))
    ap.add_argument('--soap-only', action='store_true', help='compute + cache the SOAP memmaps, then exit')
    ap.add_argument('--sweep', action='store_true', help='print an HDBSCAN parameter sweep, write nothing')
    ap.add_argument('--sweep-mcs', default='', help='colon-separated min_cluster_size list')
    ap.add_argument('--sweep-ms', default='', help='colon-separated min_samples list')
    ap.add_argument('--params', default='',
                    help='override committed params, colon-separated EL=mcs/ms, e.g. N=300/5:O=250/10')
    a = ap.parse_args()

    elements = [e for e in a.elements.split(',') if e]
    frame = load_qm9_classification_frame(a.aimel, a.remainder, a.order)
    print(f"[data] {len(frame):,} molecules ({int(frame['in_aimel'].sum()):,} AIMEl) | "
          f"SOAP r_cut {RADIUS_BOHR} Bohr | cache {a.cache_dir}", flush=True)
    meta = stream_soap_to_memmaps(frame, a.cache_dir, soap_params(RADIUS_BOHR))
    if a.soap_only:
        return
    if a.sweep:
        mcs = [int(x) for x in a.sweep_mcs.split(':') if x]
        ms = [int(x) for x in a.sweep_ms.split(':') if x]
        sweep(meta, a.cache_dir, elements, [(m, s) for m in mcs for s in ms if s <= m])
        return
    params = dict(COMMITTED)
    for tok in filter(None, a.params.split(':')):
        el, v = tok.split('=')
        params[el] = tuple(int(x) for x in v.split('/'))
    commit(frame, meta, a.cache_dir, elements, params, a.out)


if __name__ == '__main__':
    main()
