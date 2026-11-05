"""
How different are the atomic environments of the QM9 remainder from AIMEl's, measured with
the atomic-environment classification of the environment-holdout study.

The reference is the shipped whole-QM9 classification ``qm9_env_labels_6b.pkl.gz`` (SOAP 6 Bohr -> per-element
PCA to 99 % variance -> HDBSCAN, fit ONCE on AIMEl ∪ remainder; see
``data_curation/atomic/cluster_analysis/qm9_environment_clustering.py``).  AIMEl and the
remainder therefore share one label space, and novelty can be read directly off it:

* **label-support novelty** (:func:`support_novelty`, primary): an atom's environment is
  as novel as it is rare in AIMEl.  Support = number of AIMEl atoms carrying the atom's
  label, counted leave-one-molecule-out for AIMEl atoms (the null for "rare if this
  molecule were not in AIMEl", which is exactly the situation of a remainder molecule).
  ``novelty_pct`` = % of AIMEl atoms of the element whose own support is at least as
  large; ``rare99`` = support below AIMEl's 1st percentile, so at most 1 % of AIMEl atoms
  are flagged by construction (the achieved rate is reported).  Labels with no AIMEl atom
  at all are *remainder-only* environments.  HDBSCAN noise (``<el>_-1``) is not an
  environment: it is counted apart and never called rare, as in
  ``predict_env_holdout_from_inferred.molecule_env_table``.
* **6-Bohr distance novelty** (:func:`knn_novelty_pca`, continuous complement): the
  calibrated distance to the 15th-nearest AIMEl atom in the classification's own PCA
  space (:func:`soap_pca_6b`), unscaled.

Everything is joined on the QM9 id (``index``) and the atom position within the molecule
(``atom_idx``), never on a row position.

Typical notebook usage (``analysis/molecular/atomic_envs_novelty_6b.ipynb``):
    import compare_atomic_envs_6b as cae6
    labels = cae6.load_env_labels()
    atoms = cae6.atom_label_table(labels)
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
import time
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CLUSTER_DIR = os.path.join(REPO_ROOT, 'data_curation', 'atomic', 'cluster_analysis')
MOLECULAR_DIR = os.path.join(REPO_ROOT, 'data_curation', 'molecular')
for _p in (os.path.join(REPO_ROOT, 'src'), CLUSTER_DIR, MOLECULAR_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LABELS_PKL = os.path.join(CLUSTER_DIR, 'qm9_env_labels_6b.pkl.gz')
FOLDS_JSON = os.path.join(CLUSTER_DIR, 'qm9_env_folds_6b.json')
ORDER_JSON = os.path.join(MOLECULAR_DIR, 'qm9_classification_order.json.gz')
CHEMOTYPES_PKL = os.path.join(MOLECULAR_DIR, 'qm9_chemotypes.pkl.gz')
AIMEL_PKL = os.path.join(MOLECULAR_DIR, 'aimel_clustered_molecular.pkl')
REMAINDER_PKL = os.path.join(MOLECULAR_DIR, 'qm9_filtered.pkl')
INFERRED_PKL = os.path.join(MOLECULAR_DIR, 'qm9_inferred.pkl')
# The classification's own SOAP/PCA/HDBSCAN cache (written by qm9_environment_clustering.py;
# present on the machine that ran it, not shipped).
CACHE_6B = os.path.join(CLUSTER_DIR, 'cache_6b')
# This analysis' caches (6-Bohr PCA coordinates, t-SNE embeddings); regenerated on demand.
CACHE_DIR = os.path.join(HERE, 'env_novelty_6b_cache')

ELEMENTS = ['H', 'C', 'N', 'O']
RADIUS_BOHR = 6.0
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def load_env_labels(path: str = LABELS_PKL) -> pd.DataFrame:
    """``index`` (QM9 id), ``atom_env_labels_6b`` (per-atom labels), ``in_aimel``."""
    labels = pd.read_pickle(path)
    labels['index'] = labels['index'].astype(np.int64)
    if labels['index'].duplicated().any():
        raise ValueError(f"{path}: duplicated QM9 ids")
    return labels


def load_env_folds(path: str = FOLDS_JSON) -> list:
    with open(path) as fh:
        return json.load(fh)['folds']


def load_frame(aimel: str = AIMEL_PKL, remainder: str = REMAINDER_PKL,
               order: str = ORDER_JSON) -> pd.DataFrame:
    """Whole QM9 as the classifications saw it (``qtnet.data_utils.load_qm9_classification_frame``)."""
    from qtnet.data_utils import load_qm9_classification_frame
    return load_qm9_classification_frame(aimel, remainder, order)


def atom_label_table(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per QM9 atom: ``index, atom_idx, label, element, is_noise, in_aimel``."""
    lens = labels['atom_env_labels_6b'].map(len).to_numpy()
    flat = pd.Series(np.concatenate([np.asarray(v, dtype=object)
                                     for v in labels['atom_env_labels_6b']]))
    out = pd.DataFrame({
        'index': np.repeat(labels['index'].to_numpy(np.int64), lens),
        'atom_idx': np.concatenate([np.arange(n, dtype=np.int32) for n in lens]),
        'label': flat.to_numpy(dtype=object),
        'element': flat.str.split('_', n=1).str[0].to_numpy(dtype=object),
        'is_noise': flat.str.endswith('_-1').to_numpy(),
        'in_aimel': np.repeat(labels['in_aimel'].to_numpy(bool), lens),
    })
    return out


def check_atom_order(frame: pd.DataFrame, atoms: pd.DataFrame) -> int:
    """Every label's element prefix must equal the frame's atom symbol at the same
    (index, atom_idx).  Returns the number of atoms checked; raises on any mismatch."""
    lens = frame['atom'].map(len).to_numpy()
    ref = pd.DataFrame({
        'index': np.repeat(frame['index'].to_numpy(np.int64), lens),
        'atom_idx': np.concatenate([np.arange(n, dtype=np.int32) for n in lens]),
        'symbol': np.concatenate([np.asarray([str(s) for s in a], dtype=object) for a in frame['atom']]),
    })
    m = atoms[['index', 'atom_idx', 'element']].merge(ref, on=['index', 'atom_idx'], how='outer',
                                                      indicator=True)
    if (m['_merge'] != 'both').any():
        raise AssertionError(f"{int((m['_merge'] != 'both').sum())} atoms present in only one table")
    bad = int((m['element'] != m['symbol']).sum())
    if bad:
        raise AssertionError(f"{bad} atoms whose label element differs from the frame's symbol")
    return len(m)


def label_env_fold(folds: list) -> Dict[str, int]:
    """label -> environment fold it belongs to (labels outside every fold are absent)."""
    return {lab: int(f['fold']) for f in folds for lab in f['labels']}


def label_census(atoms: pd.DataFrame, folds: Optional[list] = None, pseudo: float = 0.5) -> pd.DataFrame:
    """Per environment label: AIMEl and remainder atom counts, their shares of the
    element's atoms (noise included in the denominator), log2 enrichment of the
    remainder share over the AIMEl share (pseudo-count ``pseudo``), remainder-only flag
    and environment fold."""
    tot = atoms.groupby(['element', 'in_aimel']).size().unstack('in_aimel')
    lab = atoms[~atoms['is_noise']]
    c = lab.groupby(['element', 'label', 'in_aimel']).size().unstack('in_aimel', fill_value=0)
    c = c.reindex(columns=[True, False], fill_value=0).reset_index()
    c.columns = ['element', 'label', 'n_aimel', 'n_remainder']
    n_a = c['element'].map(tot[True]).to_numpy(float)
    n_r = c['element'].map(tot[False]).to_numpy(float)
    c['aimel_share'] = c['n_aimel'] / n_a
    c['remainder_share'] = c['n_remainder'] / n_r
    c['log2_enrichment'] = np.log2(((c['n_remainder'] + pseudo) / n_r) / ((c['n_aimel'] + pseudo) / n_a))
    c['remainder_only'] = c['n_aimel'] == 0
    fold_of = label_env_fold(folds) if folds is not None else {}
    c['env_fold'] = c['label'].map(fold_of).fillna(-1).astype(int)
    c['_k'] = c['label'].str.split('_').str[1].astype(int)
    c['_e'] = c['element'].map({e: i for i, e in enumerate(ELEMENTS)})
    return c.sort_values(['_e', '_k']).drop(columns=['_k', '_e']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Label-support novelty (primary)
# ---------------------------------------------------------------------------

def support_novelty(atoms: pd.DataFrame, q: float = 1.0):
    """Per-atom AIMEl support of the atom's environment and its calibrated novelty.

    Adds to a copy of ``atoms``: ``support`` (AIMEl atoms with the same label, leave-one-
    molecule-out for AIMEl atoms; NaN for noise), ``novelty_pct`` (% of the element's
    labelled AIMEl atoms whose support is >= this atom's; NaN for noise), ``rare99``
    (support below the q-th percentile of the AIMEl supports; False for noise) and
    ``novel`` (support == 0: the label has no other AIMEl molecule).

    The threshold is the ``floor(q/100·n)``-th smallest AIMEl support, so the fraction of
    AIMEl atoms strictly below it — the false-positive rate on AIMEl — is at most q %.

    In the calibration table ``remainder_rare%`` is over all of the element's remainder atoms (noise
    counted as not rare, the denominator of the per-molecule counts); ``remainder_rare%_labelled``
    is over labelled atoms, the denominator of ``aimel_fpr%``.

    Returns ``(atoms_with_novelty, calibration_table)``.
    """
    out = atoms.copy()
    lab = ~out['is_noise'].to_numpy()
    n_aimel = out[lab & out['in_aimel']].groupby('label').size()
    support = out['label'].map(n_aimel).fillna(0).to_numpy(float)
    own = out.groupby(['index', 'label'])['atom_idx'].transform('size').to_numpy(float)
    support = np.where(out['in_aimel'].to_numpy(), support - own, support)
    support[~lab] = np.nan
    out['support'] = support
    out['novelty_pct'] = np.nan
    out['rare99'] = False

    rows = []
    for el in ELEMENTS:
        e = (out['element'] == el).to_numpy()
        a = e & lab & out['in_aimel'].to_numpy()
        s_a = np.sort(support[a])
        n = len(s_a)
        thr = float(s_a[int(np.floor(q / 100.0 * n))])
        el_lab = e & lab
        s = support[el_lab]
        out.loc[el_lab, 'novelty_pct'] = (n - np.searchsorted(s_a, s, side='left')) / n * 100.0
        out.loc[el_lab, 'rare99'] = s < thr
        r = e & ~out['in_aimel'].to_numpy()
        r_lab = r & lab
        rows.append({
            'element': el,
            'n_labels': int(out.loc[e & lab, 'label'].nunique()),
            'n_labels_remainder_only': int(out.loc[r_lab & (support == 0), 'label'].nunique()),
            'n_aimel_atoms': int((e & out['in_aimel'].to_numpy()).sum()),
            'n_remainder_atoms': int(r.sum()),
            'aimel_noise%': round(100 * float(np.mean(~lab[e & out['in_aimel'].to_numpy()])), 1),
            'remainder_noise%': round(100 * float(np.mean(~lab[r])), 1),
            'support_threshold': thr,
            'aimel_fpr%': round(100 * float(np.mean(s_a < thr)), 2),
            'remainder_rare%': round(100 * float(np.mean(out['rare99'].to_numpy()[r])), 2),
            'remainder_rare%_labelled': round(100 * float(np.mean(out['rare99'].to_numpy()[r_lab])), 2),
            'remainder_only%': round(100 * float(np.mean(support[r_lab] == 0) * r_lab.sum() / r.sum()), 2),
            'aimel_median_pct': round(float(np.median(out['novelty_pct'].to_numpy()[a])), 1),
            'remainder_median_pct': round(float(np.median(out['novelty_pct'].to_numpy()[r_lab])), 1),
        })
    out['novel'] = out['support'] == 0
    return out, pd.DataFrame(rows)


def molecule_novelty_6b(atom_nov: pd.DataFrame, frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-molecule degree of novelty (``n_rare``, ``max_novelty_pct``, ``has_rare``), keyed by
    the QM9 id ``index``.

    Also: ``n_noise``, ``n_labelled``, ``n_novel`` (atoms whose label has no other AIMEl
    molecule), ``min_aimel_support`` (support of the molecule's rarest labelled environment;
    -1 when every atom is noise), ``n_heavy``.  ``smiles`` is attached from ``frame``.
    """
    a = atom_nov
    g = a.groupby('index', sort=False)
    out = pd.DataFrame({
        'in_aimel': g['in_aimel'].first(),
        'n_atoms': g.size(),
        'n_heavy': (a['element'] != 'H').groupby(a['index'], sort=False).sum(),
        'n_noise': g['is_noise'].sum(),
        'n_rare': g['rare99'].sum().astype(int),
        'n_novel': g['novel'].sum().astype(int),
        'max_novelty_pct': g['novelty_pct'].max(),
        'mean_novelty_pct': g['novelty_pct'].mean().round(1),
        'min_aimel_support': g['support'].min(),
    })
    out['n_labelled'] = out['n_atoms'] - out['n_noise']
    out['frac_rare'] = (out['n_rare'] / out['n_atoms']).round(3)
    out['has_rare'] = out['n_rare'] > 0
    out['has_novel'] = out['n_novel'] > 0
    out['min_aimel_support'] = out['min_aimel_support'].fillna(-1)
    out = out.reset_index()
    if frame is not None and 'smiles' in frame.columns:
        out = out.merge(frame[['index', 'smiles']], on='index', how='left')
    return out.sort_values(['n_rare', 'max_novelty_pct'], ascending=False).reset_index(drop=True)


def molecule_baseline(mol_nov: pd.DataFrame, rare_hi: int = 3, count_col: str = 'n_rare',
                      pct_col: str = 'max_novelty_pct') -> pd.DataFrame:
    """% of molecules flagged, AIMEl (the chance level of the per-atom calibration) vs remainder.
    ``count_col``: the per-molecule count of rare atoms ('n_rare' = label support, 'n_rare_dist' = 6-Bohr
    distance); ``pct_col``: the per-molecule largest novelty percentile of the same score."""
    rows = []
    for name, sub in (('AIMEl (calibration)', mol_nov[mol_nov['in_aimel']]),
                      ('QM9 remainder', mol_nov[~mol_nov['in_aimel']])):
        rows.append({'set': name, 'n_molecules': len(sub),
                     '>=1 rare %': round(100 * (sub[count_col] > 0).mean(), 1),
                     f'>={rare_hi} rare %': round(100 * (sub[count_col] >= rare_hi).mean(), 1),
                     '>=1 novel %': round(100 * sub['has_novel'].mean(), 1),
                     'all-noise %': round(100 * (sub['n_labelled'] == 0).mean(), 2),
                     f'median {pct_col}': round(float(sub[pct_col].median()), 1)})
    return pd.DataFrame(rows)


def env_fold_novelty(mol_nov: pd.DataFrame, labels: pd.DataFrame, folds: list, count_col: str = 'n_rare') -> pd.DataFrame:
    """Per environment fold of the holdout study: its labels and the molecules carrying them,
    with the novelty of the remainder molecules among them (``count_col`` > 0: at least one rare atom)."""
    from qtnet.data_utils import env_fold_masks
    lab = labels.set_index('index')['atom_env_labels_6b']
    masks = env_fold_masks(labels['index'], lab, [f['labels'] for f in folds])
    nov = mol_nov.set_index('index').loc[labels['index']]
    in_aimel = labels['in_aimel'].to_numpy(bool)
    rows = []
    for f, m in zip(folds, masks):
        r = m & ~in_aimel
        rows.append({'env_fold': f['fold'], 'seed': f.get('seed'), 'n_labels': len(f['labels']),
                     'n_aimel_mols': int((m & in_aimel).sum()), 'n_remainder_mols': int(r.sum()),
                     'remainder: >=1 rare %': round(100 * float((nov[count_col].to_numpy()[r] > 0).mean()), 1),
                     'remainder: >=1 novel %': round(100 * float(nov['has_novel'].to_numpy()[r].mean()), 1),
                     'labels': ' '.join(f['labels'])})
    return pd.DataFrame(rows)


def chemotype_table(mol_nov: pd.DataFrame, chemotypes: pd.DataFrame, count_col: str = 'n_rare') -> pd.DataFrame:
    """Per Morgan-FP chemotype: AIMEl / remainder molecules and the remainder's novelty (``count_col`` > 0:
    at least one rare atom)."""
    m = mol_nov.merge(chemotypes, on='index', how='left')
    m['has_rare'] = m[count_col] > 0
    g = m.groupby('fp_cluster')
    out = pd.DataFrame({
        'n_aimel': g['in_aimel'].sum().astype(int),
        'n_remainder': (~m['in_aimel']).groupby(m['fp_cluster']).sum().astype(int),
    })
    rem = m[~m['in_aimel']].groupby('fp_cluster')
    out['remainder: >=1 rare %'] = (100 * rem['has_rare'].mean()).round(1)
    out['remainder: >=1 novel %'] = (100 * rem['has_novel'].mean()).round(1)
    out['aimel_share'] = out['n_aimel'] / out['n_aimel'].sum()
    out['remainder_share'] = out['n_remainder'] / out['n_remainder'].sum()
    out['never_in_aimel'] = out['n_aimel'] == 0
    out = out.reset_index()
    out['_k'] = out['fp_cluster'].str.split('_').str[1].astype(int)
    return out.sort_values('_k').drop(columns='_k').reset_index(drop=True)


# ---------------------------------------------------------------------------
# The classification's descriptor space: 6-Bohr SOAP -> per-element PCA (0.99)
# ---------------------------------------------------------------------------

def _pca_from_covariance(s: np.ndarray, ss: np.ndarray, n: int, variance: float) -> Dict:
    """Same maths as ``qm9_environment_clustering.pca_project``."""
    mean = s / n
    cov = (ss - n * np.outer(mean, mean)) / (n - 1)
    evals, evecs = np.linalg.eigh(cov)
    evals, evecs = np.clip(evals[::-1], 0, None), evecs[:, ::-1]
    ratio = np.cumsum(evals) / evals.sum()
    k = int(np.searchsorted(ratio, variance) + 1)
    return {'mean': mean, 'W': evecs[:, :k], 'k': k, 'var_retained': float(ratio[k - 1] * 100)}


def _soap_chunks(frame: pd.DataFrame, chunk: int):
    """Yield, per chunk of molecules, {element: (SOAP rows float64, QM9 ids, atom_idx)}."""
    from ase import Atoms
    from atomic_env_split import compute_soap_descriptors
    import qm9_environment_clustering as qec

    params = qec.soap_params(RADIUS_BOHR)
    cols = [frame[c].tolist() for c in ('index', 'atom', 'position_x', 'position_y', 'position_z')]
    n = len(frame)
    for start in range(0, n, chunk):
        parts = {el: ([], [], []) for el in ELEMENTS}
        for i in range(start, min(n, start + chunk)):
            qid, symbols = cols[0][i], [str(s) for s in cols[1][i]]
            pos = np.column_stack([cols[2][i], cols[3][i], cols[4][i]])
            X = compute_soap_descriptors(Atoms(symbols=symbols, positions=pos), params)
            sym = np.asarray(symbols)
            for el in ELEMENTS:
                idx = np.flatnonzero(sym == el)
                if len(idx):
                    parts[el][0].append(X[idx])
                    parts[el][1].append(np.full(len(idx), qid, dtype=np.int64))
                    parts[el][2].append(idx.astype(np.int32))
        yield {el: (np.vstack(p[0]).astype(np.float64), np.concatenate(p[1]), np.concatenate(p[2]))
               for el, p in parts.items() if p[0]}


def _pca_streamed(frame: pd.DataFrame, variance: float, chunk: int) -> Dict:
    """Two passes over the whole frame: exact streamed covariance -> PCA, then projection."""
    t0 = time.time()
    acc = {}
    for i, parts in enumerate(_soap_chunks(frame, chunk)):
        for el, (X, _, _) in parts.items():
            if el not in acc:
                acc[el] = [np.zeros(X.shape[1]), np.zeros((X.shape[1], X.shape[1])), 0]
            acc[el][0] += X.sum(0)
            acc[el][1] += X.T @ X
            acc[el][2] += len(X)
        if i % 20 == 0:
            print(f"[soap6b] pass 1: {min((i + 1) * chunk, len(frame)):,}/{len(frame):,} molecules "
                  f"({time.time() - t0:.0f}s)", flush=True)
    pcas = {el: _pca_from_covariance(s, ss, n, variance) for el, (s, ss, n) in acc.items()}
    out = {el: {'X': [], 'index': [], 'atom_idx': []} for el in ELEMENTS}
    for i, parts in enumerate(_soap_chunks(frame, chunk)):
        for el, (X, qid, aidx) in parts.items():
            d = pcas[el]
            out[el]['X'].append(((X - d['mean']) @ d['W']).astype(np.float32))
            out[el]['index'].append(qid)
            out[el]['atom_idx'].append(aidx)
        if i % 20 == 0:
            print(f"[soap6b] pass 2: {min((i + 1) * chunk, len(frame)):,}/{len(frame):,} molecules "
                  f"({time.time() - t0:.0f}s)", flush=True)
    res = {}
    for el in ELEMENTS:
        X = np.vstack(out[el]['X'])
        res[el] = {'X': X, 'index': np.concatenate(out[el]['index']),
                   'atom_idx': np.concatenate(out[el]['atom_idx']),
                   'evals': X.var(axis=0, ddof=1).astype(np.float64),
                   'var_retained': pcas[el]['var_retained']}
    return res


def _pca_from_classification_cache(cache_6b: str, variance: float) -> Dict:
    """Re-use the classification's own SOAP memmaps and PCA (exact)."""
    import qm9_environment_clustering as qec

    with open(os.path.join(cache_6b, 'soap_memmap_meta.pkl'), 'rb') as fh:
        meta = pickle.load(fh)
    with gzip.open(ORDER_JSON, 'rt') as fh:
        order = np.asarray(json.load(fh)['qm9_index'], dtype=np.int64)
    res = {}
    for el in ELEMENTS:
        X, k, var = qec.pca_project(el, meta, cache_6b, variance)
        idx = meta[el]['index']
        res[el] = {'X': X.astype(np.float32),
                   'index': order[np.fromiter((t[0] for t in idx), dtype=np.int64, count=len(idx))],
                   'atom_idx': np.fromiter((t[1] for t in idx), dtype=np.int32, count=len(idx)),
                   'evals': X.var(axis=0, ddof=1).astype(np.float64), 'var_retained': var}
    return res


def soap_pca_6b(frame: Optional[pd.DataFrame] = None, variance: float = 0.99, chunk: int = 1000,
                cache_path: Optional[str] = None, cache_6b: str = CACHE_6B) -> Dict:
    """Every QM9 atom in the environment classification's descriptor space.

    6-Bohr SOAP (``qm9_environment_clustering.soap_params``, the classification's
    ``compute_soap_descriptors``) -> per-element PCA to ``variance`` from the exact
    streamed covariance over all 128,908 molecules, i.e. the classification's own space
    (H 48 / C 76 / N 67 / O 52 dimensions).  Uses the classification cache ``cache_6b``
    when present; otherwise recomputes SOAP (two passes, ~1.5 min each of SOAP plus the
    covariance products) and keeps only the PCA coordinates.

    Returns ``{element: {'X' (n, k) float32 PCA coordinates, 'index' QM9 id, 'atom_idx',
    'evals' per-component variance over all QM9 atoms, 'var_retained', 'source'}}``;
    whiten with :func:`whiten`.  Cached as ``.npz`` in :data:`CACHE_DIR`.
    """
    cache_path = cache_path or os.path.join(CACHE_DIR, f'soap6b_pca_v{variance}.npz')
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        src = str(z['source'])
        return {el: {'X': z[f'{el}_X'], 'index': z[f'{el}_index'], 'atom_idx': z[f'{el}_atom_idx'],
                     'evals': z[f'{el}_evals'], 'var_retained': float(z[f'{el}_var']), 'source': src}
                for el in ELEMENTS}
    if os.path.exists(os.path.join(cache_6b, 'soap_memmap_meta.pkl')):
        res, src = _pca_from_classification_cache(cache_6b, variance), 'classification cache_6b'
    else:
        frame = frame if frame is not None else load_frame()
        res, src = _pca_streamed(frame, variance, chunk), 'recomputed (exact streamed covariance)'
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, source=np.asarray(src),
             **{f'{el}_{k}': (np.asarray(v['var_retained']) if k == 'var' else v[k])
                for el, v in res.items() for k in ('X', 'index', 'atom_idx', 'evals', 'var')})
    for v in res.values():
        v['source'] = src
    return res


def whiten(pca: Dict, element: str) -> np.ndarray:
    """PCA coordinates divided by each component's whole-QM9 standard deviation, so
    Euclidean distance is the Mahalanobis distance w.r.t. whole-QM9 (a metric that is the
    same for every reference set)."""
    d = pca[element]
    return (d['X'] / np.sqrt(d['evals']).astype(np.float32)).astype(np.float32)


def pca_summary(pca: Dict) -> pd.DataFrame:
    return pd.DataFrame([{'element': el, 'n_atoms': len(pca[el]['X']), 'n_components': pca[el]['X'].shape[1],
                          'variance_retained_%': round(pca[el]['var_retained'], 2), 'source': pca[el]['source']}
                         for el in ELEMENTS])


def knn_distance(ref: np.ndarray, query: np.ndarray, ks: Sequence[int] = (1, 5, 15),
                 loo: bool = False, n_jobs: int = -1, chunk: int = 20000) -> np.ndarray:
    """Distance from each query row to its k-th nearest ``ref`` row, for each k in ``ks``
    (1-based).  ``loo=True`` when the queries are themselves in ``ref`` (drops the self
    match).  Returns (n_query, len(ks)) float32.  When ``ref`` has fewer rows than k the
    farthest available neighbour is used."""
    from sklearn.neighbors import NearestNeighbors

    kmax = min(max(ks) + int(loo), len(ref))
    nn = NearestNeighbors(n_neighbors=kmax, algorithm='brute', n_jobs=n_jobs).fit(ref)
    out = np.empty((len(query), len(ks)), dtype=np.float32)
    cols = [min(k + int(loo), kmax) - 1 for k in ks]
    for i in range(0, len(query), chunk):
        d = nn.kneighbors(query[i:i + chunk], return_distance=True)[0]
        out[i:i + chunk] = d[:, cols]
    return out


def knn_novelty_pca(pca: Dict, atoms: pd.DataFrame, knn: int = 15, pad: int = 30):
    """Distance of every QM9 atom to AIMEl, in the classification's own space: the 6-Bohr SOAP
    projected on the whole-QM9 principal components, unscaled (the coordinates HDBSCAN clustered).
    The same notion of distance as the results notebook's per-model novelty
    (``env_holdout_figures.run_distances``), with AIMEl, QT-Net's training set, as the reference:

      * d_k = Euclidean distance to the ``knn``-th nearest AIMEl atom of the same element; for an AIMEl
        atom, the atoms of its own molecule are skipped (leave-one-molecule-out, ``pad`` extra
        neighbours are searched for that), so it is scored as if its molecule were not in AIMEl;
      * the AIMEl atoms are the calibration: ``novelty_pct`` = % of the element's AIMEl atoms with a
        smaller d_k; ``rare99`` = beyond their p99 (1 % of AIMEl atoms by construction);
        ``ratio`` = d_k / p99 (> 1 = rare).

    Unscaled rather than whitened: dividing each component by its whole-QM9 spread shrinks the
    high-variance directions, and those are the ones along which, e.g., the remainder's 17,468 alkynes
    differ from AIMEl.  Returns ``(per atom: element, index, atom_idx, in_aimel, dk, novelty_pct, rare99,
    ratio; per-element table with the calibration p50 / p99 and the remainder's rare share)``."""
    from sklearn.neighbors import NearestNeighbors

    in_aimel = atoms.set_index('index')['in_aimel'].groupby(level=0).first()
    parts, rows = [], []
    for el in ELEMENTS:
        d = pca[el]
        X = d['X'].astype(np.float32)
        aim = in_aimel.loc[d['index']].to_numpy(bool)
        ia, ir = np.flatnonzero(aim), np.flatnonzero(~aim)
        mol_a = d['index'][ia]
        nn = NearestNeighbors(n_neighbors=knn + pad, algorithm='brute', n_jobs=-1).fit(X[ia])
        d_a = np.empty(len(ia), np.float32)
        for i in range(0, len(ia), 20000):
            q = np.arange(i, min(i + 20000, len(ia)))
            dist, nb = nn.kneighbors(X[ia[q]])
            dist = np.where(mol_a[nb] != mol_a[q][:, None], dist, np.inf)     # leave the atom's molecule out
            dist.sort(axis=1)
            d_a[q] = dist[:, knn - 1]
        d_r = np.empty(len(ir), np.float32)
        for i in range(0, len(ir), 20000):
            d_r[i:i + 20000] = nn.kneighbors(X[ir[i:i + 20000]])[0][:, knn - 1]
        cs = np.sort(d_a)
        p50, p99 = float(np.percentile(d_a, 50)), float(np.percentile(d_a, 99))
        for rows_, dk, flag in ((ia, d_a, True), (ir, d_r, False)):
            parts.append(pd.DataFrame({'element': el, 'index': d['index'][rows_], 'atom_idx': d['atom_idx'][rows_],
                                       'in_aimel': flag, 'dk': dk,
                                       'novelty_pct': np.searchsorted(cs, dk, side='right') / len(cs) * 100.0,
                                       'rare99': dk > p99, 'ratio': dk / p99}))
        rows.append({'element': el, 'knn': knn, 'space': 'PCA (unscaled)', 'n_aimel_atoms': len(ia),
                     'n_remainder_atoms': len(ir), 'calib_dk_p50': round(p50, 3), 'calib_dk_p99': round(p99, 3),
                     'remainder_dk_p50': round(float(np.median(d_r)), 3),
                     'aimel_rare99%': round(100 * float(np.mean(d_a > p99)), 2),
                     'remainder_rare99%': round(100 * float(np.mean(d_r > p99)), 2)})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(rows)


def novelty_agreement(a: pd.DataFrame, b: pd.DataFrame, names=('label', 'kNN')) -> pd.DataFrame:
    """Per element agreement of two per-atom novelty tables joined on (index, atom_idx):
    rare rates, their intersection, Jaccard and the Spearman correlation of the percentiles."""
    from scipy.stats import spearmanr

    m = a[['index', 'atom_idx', 'element', 'rare99', 'novelty_pct']].merge(
        b[['index', 'atom_idx', 'element', 'rare99', 'novelty_pct']],
        on=['index', 'atom_idx'], suffixes=('_a', '_b'))
    if len(m) != len(a) or len(m) != len(b):
        raise AssertionError(f"only {len(m):,} of {len(a):,} / {len(b):,} atoms matched on (index, atom_idx)")
    if (m['element_a'] != m['element_b']).any():
        raise AssertionError("the two tables disagree on the element of some atoms (atom order)")
    rows = []
    for el in ELEMENTS:
        s = m[m['element_a'] == el]
        ra, rb = s['rare99_a'].to_numpy(bool), s['rare99_b'].to_numpy(bool)
        ok = s['novelty_pct_a'].notna() & s['novelty_pct_b'].notna()
        rows.append({'element': el, 'n_atoms': len(s),
                     f'{names[0]}_rare%': round(100 * ra.mean(), 1),
                     f'{names[1]}_rare%': round(100 * rb.mean(), 1),
                     'both_rare%': round(100 * (ra & rb).mean(), 1),
                     'jaccard': round((ra & rb).sum() / max((ra | rb).sum(), 1), 2),
                     'spearman_pct': round(float(spearmanr(s.loc[ok, 'novelty_pct_a'],
                                                           s.loc[ok, 'novelty_pct_b'])[0]), 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# t-SNE views: chemotypes, and atomic environments per element
# ---------------------------------------------------------------------------

def _sample_two_sources(is_a: np.ndarray, n_a: int, n_b: int, seed: int):
    rng = np.random.default_rng(seed)
    ia, ib = np.flatnonzero(is_a), np.flatnonzero(~is_a)
    return (rng.choice(ia, min(n_a, len(ia)), replace=False),
            rng.choice(ib, min(n_b, len(ib)), replace=False))


def _tsne(X: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=seed).fit_transform(X)


def chemotype_embedding(frame: pd.DataFrame, chemotypes: pd.DataFrame, n_per_source: int = 20000,
                        perplexity: float = 50.0, seed: int = RANDOM_STATE,
                        cache_path: Optional[str] = None) -> pd.DataFrame:
    """t-SNE of the chemotype space: the chemotype recipe's ECFP4 (2048 bits) -> L2 ->
    PCA(50) fit on ALL QM9 molecules (``qm9_chemotype_clustering``), then a joint
    embedding of ``n_per_source`` AIMEl and remainder molecules.  Cached."""
    cache_path = cache_path or os.path.join(CACHE_DIR, f'tsne_chemotypes_{n_per_source}_p{perplexity:g}.pkl')
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    from qm9_chemotype_clustering import morgan_matrix

    X, valid = morgan_matrix(frame['smiles'].tolist(), radius=2, n_bits=2048)
    Z = PCA(n_components=50, svd_solver='randomized', random_state=seed).fit_transform(normalize(X))
    ids = frame['index'].to_numpy(np.int64)[valid]
    in_aimel = frame['in_aimel'].to_numpy(bool)[valid]
    sa, sb = _sample_two_sources(in_aimel, n_per_source, n_per_source, seed)
    sel = np.concatenate([sa, sb])
    print(f"[tsne:chemotypes] embedding {len(sel):,} molecules ({len(sa):,} AIMEl + {len(sb):,} remainder)...",
          flush=True)
    emb = _tsne(Z[sel], perplexity, seed)
    out = pd.DataFrame({'index': ids[sel], 'in_aimel': in_aimel[sel], 'x': emb[:, 0], 'y': emb[:, 1]})
    out = out.merge(chemotypes, on='index', how='left')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out.to_pickle(cache_path)
    return out


def env_embedding(pca: Dict, atom_nov: pd.DataFrame, element: str, n_per_source: int = 20000,
                  perplexity: float = 50.0, seed: int = RANDOM_STATE, space: str = 'pca',
                  cache_path: Optional[str] = None) -> pd.DataFrame:
    """Joint t-SNE of ``n_per_source`` AIMEl and remainder atoms of ``element`` in the 6-Bohr
    classification space (the input is the PCA coordinates; the labels only annotate).
    ``space='pca'``: unscaled, the coordinates HDBSCAN clustered and the space of the distance of
    :func:`knn_novelty_pca`; ``'whitened'``: each component divided by its whole-QM9 spread.  Cached."""
    suffix = '' if space == 'whitened' else f'_{space}'           # the whitened caches predate the argument
    cache_path = cache_path or os.path.join(CACHE_DIR, f'tsne_env_{element}_{n_per_source}_p{perplexity:g}{suffix}.pkl')
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    d = pca[element]
    keys = pd.DataFrame({'index': d['index'], 'atom_idx': d['atom_idx']})
    keys = keys.merge(atom_nov[['index', 'atom_idx', 'in_aimel', 'label', 'novelty_pct']],
                      on=['index', 'atom_idx'], how='left')
    if keys['in_aimel'].isna().any():
        raise AssertionError("PCA rows without a label row")
    is_a = keys['in_aimel'].to_numpy(bool)
    sa, sb = _sample_two_sources(is_a, n_per_source, n_per_source, seed)
    sel = np.concatenate([sa, sb])
    print(f"[tsne:{element}] embedding {len(sel):,} atoms ({len(sa):,} AIMEl + {len(sb):,} remainder)...",
          flush=True)
    X = whiten(pca, element) if space == 'whitened' else pca[element]['X'].astype(np.float32)
    emb = _tsne(X[sel], perplexity, seed)
    out = keys.iloc[sel].reset_index(drop=True).assign(x=emb[:, 0], y=emb[:, 1])
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out.to_pickle(cache_path)
    return out




# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

RUNS_DIR = os.path.join(REPO_ROOT, 'experiments', 'molecular', 'env_holdout')
# faint background of the t-SNE panel maps: blue (AIMEl denser) -> white -> red (QM9 remainder denser)
BG_COLORS = ('#1a4f9c', '#ffffff', '#e8604c')
BG_ALPHA = 0.3
# text sizes of QT-Net's published figures (analysis/inference/analyse_inference.py, _FONT)
DEFAULT_FONTS = {'title': 18, 'label': 16, 'tick': 12, 'legend': 12, 'annot': 12}


def _fonts(fonts: Optional[dict]) -> dict:
    return {**DEFAULT_FONTS, **(fonts or {})}


def _style():
    """The rcParams of QT-Net's published figures (``analyse_inference.py``), no bold, no top/right spines."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
                         'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12, 'xtick.labelsize': 12,
                         'ytick.labelsize': 12, 'legend.fontsize': 12, 'axes.labelweight': 'normal',
                         'axes.titleweight': 'normal', 'figure.titleweight': 'normal',
                         'axes.spines.top': False, 'axes.spines.right': False})


def _save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
        root, _ = os.path.splitext(save_path)
        fig.savefig(root + '.svg', bbox_inches='tight')
        print(f"[plot] saved {save_path} (+svg)")


def _square_extent(x: np.ndarray, y: np.ndarray, pad: float = 0.03):
    """A square (xmin, xmax, ymin, ymax) around the points, so both axes span the same length."""
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    h = (1 + pad) * max(np.ptp(x), np.ptp(y)) / 2
    return (cx - h, cx + h, cy - h, cy + h)


def _tsne_axis(ax, ext, xlabel: Optional[str] = 't-SNE 1', ylabel: Optional[str] = 't-SNE 2', fontsize: float = 16,
               weight: str = 'normal'):
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=weight)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=weight)


def _log2_share_ratio(is_b: np.ndarray, pseudo: float = 0.5):
    n_b = float(is_b.sum())
    n_a = float(len(is_b) - n_b)

    def f(v):
        nb = float(np.sum(v))
        return np.log2(((nb + pseudo) / n_b) / ((len(v) - nb + pseudo) / n_a))
    return f


def density_diff_panels(axes, emb: pd.DataFrame, *, titles=('AIMEl', 'QM9 remainder', 'Difference'),
                        cbar_labels=('AIMEl atoms', 'QM9 remainder atoms', r'Log$_2$ share ratio (QM9 remainder / AIMEl)'),
                        gridsize: int = 60, vlim: float = 3.0, mincnt_diff: int = 3, xlabel: str = 't-SNE 1',
                        ylabel: str = 't-SNE 2', ylabel_first: bool = True, fonts: Optional[dict] = None,
                        title_weight: str = 'normal', label_weight: str = 'normal'):
    """Three square panels on one embedding: AIMEl count, remainder count (log gradients) and the
    difference: per hexbin cell, log2 of the remainder's share over AIMEl's share (pseudo-count
    0.5), so red = populated by the remainder more than by AIMEl, blue = the reverse."""
    from matplotlib.colors import LogNorm

    f = _fonts(fonts)
    x, y = emb['x'].to_numpy(), emb['y'].to_numpy()
    a = emb['in_aimel'].to_numpy(bool)
    ext = _square_extent(x, y)
    fig = axes[0].figure
    cb = dict(location='bottom', shrink=0.8, pad=0.02, aspect=30)
    h1 = axes[0].hexbin(x[a], y[a], gridsize=gridsize, extent=ext, cmap='Blues', norm=LogNorm(), mincnt=1, linewidths=0.1)
    h2 = axes[1].hexbin(x[~a], y[~a], gridsize=gridsize, extent=ext, cmap='Reds', norm=LogNorm(), mincnt=1, linewidths=0.1)
    h3 = axes[2].hexbin(x, y, C=(~a).astype(float), reduce_C_function=_log2_share_ratio(~a), gridsize=gridsize,
                        extent=ext, cmap='RdBu_r', vmin=-vlim, vmax=vlim, mincnt=mincnt_diff, linewidths=0.1)
    for i, (ax, hb, ttl, lab) in enumerate(zip(axes, (h1, h2, h3), titles, cbar_labels)):
        c = fig.colorbar(hb, ax=ax, **cb)
        c.set_label(lab, fontsize=f['label'], fontweight=label_weight)
        c.ax.tick_params(labelsize=f['tick'])
        ax.set_title(ttl, fontsize=f['title'], fontweight=title_weight)
        _tsne_axis(ax, ext, xlabel, ylabel if (i == 0 and ylabel_first) else None, f['label'], label_weight)


def tsne_chemotype_view(emb: pd.DataFrame, *, titles=('Chemotypes', 'AIMEl', 'QM9 remainder', 'Difference'),
                        cbar_labels=('AIMEl molecules', 'QM9 remainder molecules',
                                     r'Log$_2$ share ratio (QM9 remainder / AIMEl)'),
                        gridsize: int = 60, xlabel: str = 't-SNE 1', ylabel: str = 't-SNE 2',
                        fonts: Optional[dict] = None, title_weight: str = 'normal', label_weight: str = 'normal',
                        chemotype_panel: bool = True, figsize=(20, 6.6), save_path: Optional[str] = None):
    """Chemotype t-SNE: coloured by chemotype (its number at each chemotype's median position), then
    AIMEl | QM9 remainder | difference; square panels.  ``chemotype_panel=False`` leaves out the first
    panel (``titles`` and the figure are then those of the last three)."""
    import matplotlib.pyplot as plt
    _style()
    f = _fonts(fonts)
    if not chemotype_panel:
        fig, axes = plt.subplots(1, 3, figsize=figsize, layout='constrained')
        fig.get_layout_engine().set(w_pad=0.02, wspace=0.01)
        density_diff_panels(axes, emb, titles=titles[-3:], cbar_labels=cbar_labels, gridsize=gridsize, xlabel=xlabel,
                            ylabel=ylabel, fonts=f, title_weight=title_weight, label_weight=label_weight)
        _save(fig, save_path)
        return fig
    cmap = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors) + \
        list(plt.get_cmap('tab20c').colors)
    fig, axes = plt.subplots(1, 4, figsize=figsize, layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, wspace=0.01)
    k = emb['fp_cluster'].str.split('_').str[1].astype(int).to_numpy()
    ext = _square_extent(emb['x'].to_numpy(), emb['y'].to_numpy())
    axes[0].scatter(emb['x'], emb['y'], c=[cmap[i % len(cmap)] for i in k], s=1.2, alpha=0.6, linewidths=0)
    for kk, g in emb.assign(k=k).groupby('k'):
        axes[0].text(g['x'].median(), g['y'].median(), str(kk), fontsize=f['annot'] - 4, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.6))
    axes[0].set_title(titles[0], fontsize=f['title'], fontweight=title_weight)
    _tsne_axis(axes[0], ext, xlabel, ylabel, f['label'], label_weight)
    # an empty colorbar slot keeps the first panel the same size as the others
    cax = fig.colorbar(plt.cm.ScalarMappable(cmap='Greys'), ax=axes[0], location='bottom', shrink=0.8, pad=0.02, aspect=30)
    cax.ax.set_visible(False)
    density_diff_panels(axes[1:], emb, titles=titles[1:], cbar_labels=cbar_labels, gridsize=gridsize, xlabel=xlabel,
                        ylabel=ylabel, ylabel_first=False, fonts=f, title_weight=title_weight, label_weight=label_weight)
    _save(fig, save_path)
    return fig


def tsne_env_view(emb: pd.DataFrame, element: str, *, title_fmt: str = '{panel}',
                  panels=('AIMEl', 'QM9 remainder', 'Difference'),
                  cbar_labels=('AIMEl atoms', 'QM9 remainder atoms', r'Log$_2$ share ratio (QM9 remainder / AIMEl)'),
                  gridsize: int = 60, xlabel: str = 't-SNE 1', ylabel: str = 't-SNE 2', fonts: Optional[dict] = None,
                  title_weight: str = 'normal', label_weight: str = 'normal', figsize=(15.5, 6.6),
                  save_path: Optional[str] = None):
    """Environment t-SNE of one element: AIMEl | QM9 remainder | difference; square panels.  ``title_fmt`` takes {element} and {panel};
    the default leaves the element to the caption."""
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, wspace=0.01)
    density_diff_panels(axes, emb, titles=[title_fmt.format(element=element, panel=t) for t in panels],
                        cbar_labels=cbar_labels, gridsize=gridsize, xlabel=xlabel, ylabel=ylabel, fonts=fonts,
                        title_weight=title_weight, label_weight=label_weight)
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Where the holdout's test panels sit: the five folds of one repeat
# ---------------------------------------------------------------------------

# one colour per fold, away from the atom colours (CPK / held-out colours) and the background: a purple between
# winter's blue and cool's magenta, winter's blue end, summer's yellow end darkened (mustard), summer's green,
# coolwarm's red end.  Each needs mid lightness, so that its light tint (panel A / C) reads as lighter.
FOLD_COLORS = ['#9900cc', '#0000ff', '#c9a400', '#008066', '#b40426']
HELD_OUT_LIGHT = 0.4     # how much lighter the 'light' panel is drawn (mixed with white)


def load_fold_panels(folds: Sequence[int], runs_dir: str = RUNS_DIR, variant: str = 'blind',
                     fraction: float = 1.0) -> pd.DataFrame:
    """Test panels of the given folds (``index``, ``panel``, ``fold``, ``repeat``, ``cv_fold``,
    ``env_fold``, ``env_labels``), panels A/B/C only, rebuilt from each run's ``config.json``, the
    labels and the chemotypes, so no run output is needed (the test set is the same for both
    variants and every fraction)."""
    a = load_env_labels()
    a = a[a['in_aimel']].merge(pd.read_pickle(CHEMOTYPES_PKL), on='index')
    parts = []
    for f in folds:
        d = os.path.join(runs_dir, variant, f'fold_{f}', f'frac_{fraction}')
        cfg = json.load(open(os.path.join(d, 'config.json')))
        held = a['fp_cluster'].isin(cfg['heldout_chemotypes'])
        env = set(cfg['env_fold_labels'])
        res = a['atom_env_labels_6b'].map(lambda L: bool(env & set(L)))
        t = a[['index']].assign(panel=np.select([held & ~res, held & res, ~held & res], ['A', 'B', 'C'], ''))
        t = t[t['panel'] != '']
        parts.append(t.assign(fold=f, repeat=cfg['repeat'], cv_fold=cfg['cv_fold'], env_fold=cfg['env_fold'],
                              env_labels=[cfg['env_fold_labels']] * len(t)))
    return pd.concat(parts, ignore_index=True)


def _shade(color, amount: float):
    """``color`` mixed with white by ``amount`` (0 = the colour, 1 = white)."""
    from matplotlib.colors import to_rgb
    c = np.asarray(to_rgb(color))
    return tuple(c + (1 - c) * amount)


def _background(ax, emb: pd.DataFrame, ext, gridsize: int = 60, vlim: float = 3.0):
    """Faint diff map: where AIMEl (blue) and the QM9 remainder (red) are denser."""
    from matplotlib.colors import LinearSegmentedColormap
    x, y = emb['x'].to_numpy(), emb['y'].to_numpy()
    rem = ~emb['in_aimel'].to_numpy(bool)
    ax.hexbin(x, y, C=rem.astype(float), reduce_C_function=_log2_share_ratio(rem), gridsize=gridsize, extent=ext,
              cmap=LinearSegmentedColormap.from_list('aimel_qm9', BG_COLORS), vmin=-vlim, vmax=vlim, mincnt=3,
              linewidths=0, alpha=BG_ALPHA, zorder=0)


def _group_hexbin(ax, x, y, group, dark, ext, colors: Sequence[str], gridsize: int = 60,
                  light: float = HELD_OUT_LIGHT, n_shades: int = 64):
    """One hexbin layer for several groups: each cell is coloured by the group with the most items in
    it, from the group colour (all of the cell's items of that group 'dark') to a light tint (none)."""
    from matplotlib.colors import ListedColormap
    code = 2 * np.asarray(group, int) + np.asarray(dark, int)
    n = len(colors)

    def reduce(v):
        v = np.asarray(v, int)
        g = int(np.bincount(v // 2, minlength=n).argmax())
        return g + 0.999 * (1.0 - float((v[v // 2 == g] % 2).mean()))

    cmap = ListedColormap([_shade(colors[g], light * t) for g in range(n) for t in np.linspace(0, 1, n_shades)])
    return ax.hexbin(x, y, C=code.astype(float), reduce_C_function=reduce, gridsize=gridsize, extent=ext, cmap=cmap,
                     vmin=0, vmax=n, mincnt=1, linewidths=0.15, edgecolors='white', zorder=2)


def _fold_legend(fig, folds: pd.DataFrame, colors, fold_fmt: str, shade_entries: Sequence, bg_labels: Sequence[str],
                 fonts: dict, ncol: int, gap: float = 0.0):
    """Legend below the figure: one entry per fold (``fold_fmt`` with {fold}, {cv_fold}, {env_fold},
    {env_labels}), the shading entries ``(label, colour)`` and the two background entries (an empty
    label leaves its entry out); ``gap`` (pt) is blank space between the axes and the legend."""
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker='h', ls='', color=colors[k], ms=12,
                label=fold_fmt.format(fold=r.fold, cv_fold=r.cv_fold, env_fold=r.env_fold, env_labels=', '.join(r.env_labels)))
         for k, r in enumerate(folds.itertuples())]
    h += [Line2D([0], [0], marker='h', ls='', color=c, ms=12, label=lab) for lab, c in shade_entries]
    h += [Line2D([0], [0], marker='s', ls='', color=_shade(BG_COLORS[i], 1 - BG_ALPHA), ms=11, label=lab)
          for i, lab in zip((0, 2), bg_labels) if lab]
    fig.legend(handles=h, loc='outside lower center', ncol=ncol, frameon=False, fontsize=fonts['legend'],
               columnspacing=1.2, handletextpad=0.3, title=' ' if gap else None, title_fontsize=gap or None)


def _fold_table(panels: pd.DataFrame) -> pd.DataFrame:
    return panels.drop_duplicates('fold')[['fold', 'repeat', 'cv_fold', 'env_fold', 'env_labels']].sort_values('fold')


def tsne_panel_map(emb: pd.DataFrame, panels: pd.DataFrame, *, colors: Sequence[str] = FOLD_COLORS, gridsize: int = 45,
                   title: Optional[str] = None,
                   fold_fmt: str = 'Fold {fold}', dark_label: str = 'Dark: mostly panel B',
                   light_label: str = 'Light: mostly panel A',
                   bg_labels: Sequence[str] = ('AIMEl denser', 'QM9 remainder denser'), legend_ncol: int = 3,
                   xlabel: str = 't-SNE 1', ylabel: str = 't-SNE 2', fonts: Optional[dict] = None,
                   title_weight: str = 'normal', label_weight: str = 'normal', figsize=(9.0, 10.4), save_path=None):
    """Chemotype t-SNE of how the chemotype axis partitions the space: hexbins of the held-out
    chemotype molecules (panels A ∪ B) of each fold in ``panels`` (the five folds of one repeat),
    coloured by fold and darker where panel B (environments also held out) dominates, over the faint
    AIMEl-vs-remainder difference map."""
    import matplotlib.pyplot as plt
    _style()
    f = _fonts(fonts)
    folds = _fold_table(panels)
    pos = {fo: k for k, fo in enumerate(folds['fold'])}
    x, y = emb['x'].to_numpy(), emb['y'].to_numpy()
    ext = _square_extent(x, y)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    _background(ax, emb, ext, gridsize)
    pts = emb[emb['in_aimel']][['index', 'x', 'y']].merge(
        panels.loc[panels['panel'].isin(['A', 'B']), ['index', 'panel', 'fold']], on='index')
    _group_hexbin(ax, pts['x'], pts['y'], pts['fold'].map(pos), pts['panel'] == 'B', ext, colors, gridsize)
    _tsne_axis(ax, ext, xlabel, ylabel, f['label'], label_weight)
    if title:
        ax.set_title(title, fontsize=f['title'], fontweight=title_weight)
    _fold_legend(fig, folds, colors, fold_fmt, [(dark_label, _shade('#555555', 0)), (light_label, _shade('#555555', HELD_OUT_LIGHT))],
                 bg_labels, f, ncol=legend_ncol)
    _save(fig, save_path)
    return fig


def tsne_env_panel_map(embs: Dict[str, pd.DataFrame], panels: pd.DataFrame, *, colors: Sequence[str] = FOLD_COLORS,
                       gridsize: int = 45, title_fmt: str = '{element}',
                       suptitle: Optional[str] = None,
                       fold_fmt: str = 'Fold {fold}: {env_labels}',
                       dark_label: str = 'Dark: mostly in panel-B molecules',
                       light_label: str = 'Light: mostly in panel-C molecules',
                       bg_labels: Sequence[str] = ('AIMEl denser', 'QM9 remainder denser'), legend_ncol: int = 3,
                       legend_gap: float = 14.0, xlabel: str = 't-SNE 1', ylabel: str = 't-SNE 2',
                       fonts: Optional[dict] = None, title_weight: str = 'bold', label_weight: str = 'normal',
                       w_pad: float = 0.15, h_pad: float = 0.15, wspace: float = 0.06, hspace: float = 0.06,
                       figsize=(13.0, 15.0), save_path=None):
    """Per element (2 × 2 square environment t-SNEs): hexbins of the sampled AIMEl atoms whose own
    label is reserved by each fold's environment fold, coloured by fold (same colours as
    :func:`tsne_panel_map`) and darker where the atoms belong to panel-B molecules.  The element is
    each panel's title (``title_weight``, bold by default); ``w_pad`` / ``h_pad`` (inches) and
    ``wspace`` / ``hspace`` (fraction of the panel) separate the four panels."""
    import matplotlib.pyplot as plt
    _style()
    f = _fonts(fonts)
    folds = _fold_table(panels)
    fig, axes = plt.subplots(2, 2, figsize=figsize, layout='constrained')
    fig.get_layout_engine().set(w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace)
    for ax, el in zip(axes.ravel(), ELEMENTS):
        emb = embs[el]
        ext = _square_extent(emb['x'].to_numpy(), emb['y'].to_numpy())
        _background(ax, emb, ext, gridsize)
        parts = []
        for k, r in enumerate(folds.itertuples()):
            a = emb[emb['in_aimel'] & emb['label'].isin(r.env_labels)]
            p = panels[panels['fold'] == r.fold][['index', 'panel']]
            parts.append(a.merge(p, on='index', how='left').assign(group=k))
        pts = pd.concat(parts, ignore_index=True)
        if pts['panel'].isna().any() or not pts['panel'].isin(['B', 'C']).all():
            raise AssertionError(f"{el}: reserved-environment atoms outside panels B/C")
        _group_hexbin(ax, pts['x'], pts['y'], pts['group'], pts['panel'] == 'B', ext, colors, gridsize)
        _tsne_axis(ax, ext, xlabel if el in ('N', 'O') else None, ylabel if el in ('H', 'N') else None, f['label'],
                   label_weight)
        ax.set_title(title_fmt.format(element=el), fontsize=f['title'], fontweight=title_weight)
    if suptitle:
        fig.suptitle(suptitle, fontsize=f['title'], fontweight=title_weight)
    _fold_legend(fig, folds, colors, fold_fmt, [(dark_label, _shade('#555555', 0)), (light_label, _shade('#555555', HELD_OUT_LIGHT))],
                 bg_labels, f, ncol=legend_ncol, gap=legend_gap)
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# QT-Net dipole error and novelty; outliers in 3D
# ---------------------------------------------------------------------------

CPK = {'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D'}
# per-element colours of the held-out environments in the atomic leakage analysis
# (analysis/atomic/leakage_analysis/leakage_vs_radius.py, HOLDOUT_LABEL_COLORS)
FAR_COLORS = {'H': '#37B221', 'C': '#4A5568', 'N': '#047DD9', 'O': '#FF6363'}
# blue -> salmon -> red: the novelty colours of the molecular figures (no rare / 1-2 / >=3 rare atoms)
NOVELTY_SCALE = ('#3b4cc0', '#f39c7d', '#b40426')
COVALENT_RADIUS_A = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66}
BOHR_TO_A = 0.529177210903
EBOHR_TO_DEBYE = 2.541746473   # e·Bohr -> Debye, as in scripts/inference/molecule_dipoles_from_inferred.py


def distance_novelty_atoms(atom_knn: pd.DataFrame, knn_calib: pd.DataFrame) -> pd.DataFrame:
    """Per remainder atom (output of :func:`knn_novelty_pca`): ``ratio`` = its
    15th-NN distance to AIMEl divided by the AIMEl calibration p99 of its element (> 1 = rare)."""
    p99 = knn_calib.set_index('element')['calib_dk_p99']
    return atom_knn.assign(ratio=atom_knn['dk'] / atom_knn['element'].map(p99))


def molecule_distance_novelty(atom_dist: pd.DataFrame) -> pd.DataFrame:
    """Per molecule: ``n_rare_dist`` (atoms beyond AIMEl's p99 in 6-Bohr distance, noise
    atoms included), ``frac_rare_dist`` (their share of the molecule's atoms), ``mean_ratio`` /
    ``max_ratio`` (mean / largest atom distance ÷ p99) and ``max_dist_pct``."""
    g = atom_dist.groupby('index')
    out = pd.DataFrame({'n_rare_dist': g['rare99'].sum().astype(int), 'frac_rare_dist': g['rare99'].mean(),
                        'mean_ratio': g['ratio'].mean(), 'max_ratio': g['ratio'].max(),
                        'max_dist_pct': g['novelty_pct'].max()})
    if 'in_aimel' in atom_dist:
        out['in_aimel'] = g['in_aimel'].first()
    return out.reset_index()


def dipole_errors(df_mu: pd.DataFrame, mol_nov: pd.DataFrame, mol_dist: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per remainder molecule: reference and QT-Net dipole, absolute error, its label novelty
    (:func:`molecule_novelty_6b`) and, when given, its 6-Bohr distance novelty
    (:func:`molecule_distance_novelty`); joined on the QM9 id."""
    cols = ['index', 'n_atoms', 'n_rare', 'n_novel', 'n_noise', 'max_novelty_pct', 'min_aimel_support']
    cols += ['smiles'] if 'smiles' in mol_nov else []
    m = df_mu[['index', 'mu', 'mu_inferred']].merge(mol_nov[cols], on='index', how='inner', validate='one_to_one')
    if mol_dist is not None:
        m = m.merge(mol_dist, on='index', how='left', validate='one_to_one')
    m['abs_err'] = (m['mu_inferred'] - m['mu']).abs()
    m['frac_rare'] = m['n_rare'] / m['n_atoms']                 # share of the molecule's atoms with a rare label
    return m


def dipole_group_table(err: pd.DataFrame, count_col: str = 'n_rare_dist', rare_hi: int = 3, n_boot: int = 1000,
                       seed: int = 0, exclude_col: Optional[str] = None) -> pd.DataFrame:
    """MAE of molecules with 0 / 1–(rare_hi−1) / ≥rare_hi rare atoms (``count_col``) with a
    molecule-bootstrap 95 % CI, the group's share of molecules and of the total absolute error;
    with ``exclude_col`` (boolean), also the group's count and MAE without those molecules."""
    rng = np.random.default_rng(seed)
    n = err[count_col].to_numpy()
    groups = {'No rare': n == 0, f'1–{rare_hi - 1} rare': (n > 0) & (n < rare_hi), f'≥{rare_hi} rare': n >= rare_hi}
    total = err['abs_err'].sum()
    rows = []
    for g, m in groups.items():
        e = err.loc[m, 'abs_err'].to_numpy()
        boots = e[rng.integers(0, len(e), size=(n_boot, len(e)))].mean(1)
        rows.append({'group': g, 'n': len(e), 'share_of_molecules_%': 100 * len(e) / len(err), 'MAE': e.mean(),
                     'MAE_lo': np.quantile(boots, 0.025), 'MAE_hi': np.quantile(boots, 0.975),
                     'median_abs_err': float(np.median(e)), 'share_of_total_abs_err_%': 100 * e.sum() / total})
        if exclude_col:
            keep = m & ~err[exclude_col].to_numpy(bool)
            rows[-1].update({f'n_{exclude_col}': int((m & ~keep).sum()),
                             f'MAE_without_{exclude_col}': float(err.loc[keep, 'abs_err'].mean())})
    return pd.DataFrame(rows).assign(grouped_by=count_col)


def _three_ticks(vmin: float, vmax: float, log: bool):
    """About three colour-bar ticks: decades 10^0, 10^2, ..., 10^max (log), or round numbers (linear)."""
    if log:
        hi = int(np.floor(np.log10(vmax)))
        return [10.0 ** e for e in sorted(set(range(0, hi + 1, 2)) | {hi})]
    from matplotlib.ticker import MaxNLocator
    t = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(vmin, vmax)
    return [v for v in t if vmin - 1e-9 <= v <= vmax + 1e-9]


def _rare_groups(n: np.ndarray, rare_hi: int, labels: Optional[Sequence[str]] = None):
    """Masks of molecules with 0 / 1–(rare_hi−1) / ≥rare_hi rare atoms, with their labels."""
    labels = labels or ('No rare', f'1–{rare_hi - 1} rare', f'≥{rare_hi} rare')
    return list(zip(labels, (n == 0, (n > 0) & (n < rare_hi), n >= rare_hi)))


def _group_marginal(ax, t: np.ndarray, p: np.ndarray, groups, colors, max_bins: int = 60, lw: float = 1.4):
    """Top/right marginals with one step histogram (counts, log) per molecule group, laid out like
    ``analyse_inference._add_marginal`` so the main axes keep the same size."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes('top', size='22%', pad=0.00, sharex=ax)
    ax_right = divider.append_axes('right', size='22%', pad=0.00, sharey=ax)
    edges = np.histogram_bin_edges(t, bins='fd')
    if len(edges) - 1 > max_bins:
        edges = np.histogram_bin_edges(t, bins=max_bins)
    for (_, m), c in zip(groups, colors):
        ax_top.hist(t[m], bins=edges, histtype='step', lw=lw, color=c)
        ax_right.hist(p[m], bins=edges, histtype='step', lw=lw, color=c, orientation='horizontal')
    ax_top.set_yscale('log'); ax_right.set_xscale('log')
    for a in (ax_top, ax_right):
        a.grid(False); a.minorticks_off()
        for sp in a.spines.values():
            sp.set_visible(False)
    ax_top.set_yticks([]); ax_right.set_xticks([])
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_right.tick_params(axis='y', labelleft=False)


def dipole_figure(err: pd.DataFrame, metric_df: Optional[pd.DataFrame] = None, *, right: str = 'parity',
                  color_col: str = 'max_ratio', count_col: str = 'n_rare_dist', rare_hi: int = 3,
                  group_labels: Optional[Sequence[str]] = None,
                  group_fmt: str = '{label}: MAE {mae:.3f} D (n = {n:,})', legend_title: Optional[str] = None,
                  highlights: Sequence[dict] = (), highlight_min_err: float = 1.0,
                  highlight_fmt: str = '{label}: {n_off:,} of {n:,} with |Δμ| > {thr:g} D', highlight_lw: float = 1.0,
                  highlight_alpha: float = 1.0,
                  legend_fontsize: Optional[float] = None,
                  bins: int = 80, cmap_count: str = 'winter',
                  novelty_colors: Sequence[str] = NOVELTY_SCALE, novelty_range: Optional[tuple] = None,
                  xlabel: str = r'Target molecular $|\mathbf{\mu}|$ [D]',
                  ylabel: str = r'Molecular $|\mathbf{\mu}|$ from QT-Net [D]',
                  count_label: str = 'Count', novelty_label: str = 'Max distance\nto AIMEl / p99',
                  xlabel_distance: str = 'Max atom distance to AIMEl / p99 (6 Bohr)',
                  ylabel_error: str = r'$|\Delta \mu|$ [D]', marginal_color: str = '#27d687',
                  titles: Sequence[str] = ('', ''), fonts: Optional[dict] = None, title_weight: str = 'normal',
                  label_weight: str = 'normal', cbar_box: Sequence[float] = (0.46, 0.10, 0.50, 0.045),
                  cbar_label_size: Optional[float] = None, legend_kw: Optional[dict] = None, figsize=(12.4, 6.0),
                  save_path=None):
    """1 × 2 QT-Net dipole figure on the remainder, sharing the y axis.  ``cbar_box``: (x, y, width,
    height) of both inset colour bars, in axes fractions; ``cbar_label_size``: their labels' size
    (default: the tick size); ``legend_kw``: overrides of the right panel's legend arguments (e.g. a
    smaller ``labelspacing`` to keep it clear of the data).

    Left: parity as in ``analyse_inference.dipole_parity`` (hexbin counts, marginals, R² with its
    bootstrap CI from ``metric_df``).  Right, ``right='parity'``: the same parity, each cell coloured
    by the mean of ``color_col`` over its molecules (default: the largest 6-Bohr distance to AIMEl
    among the molecule's atoms, in units of AIMEl's p99), on a linear blue → salmon → red scale
    (``novelty_colors``; ``novelty_range`` = colour limits, default the 1st–99th percentile).  Its
    marginals are split by the molecule's number
    of rare atoms (``count_col``: 0 / 1–(rare_hi−1) / ≥rare_hi, in ``novelty_colors``), and the legend
    gives each group's MAE (``group_fmt``: {label}, {mae}, {n}, and {mae_wo} = the MAE without the
    highlighted molecules).  ``highlights``: chemical groups to outline on the right panel, each a
    dict with ``col`` (boolean column of ``err``), ``label`` and ``color``, optionally ``lw`` and
    ``alpha`` (default ``highlight_lw`` / ``highlight_alpha``), plus any field used by
    ``highlight_fmt``; the hexbin cells holding molecules of the group with an absolute error above
    ``highlight_min_err`` (0 = all of them) are outlined in its colour, with a legend entry
    (``highlight_fmt``: the dict's fields and {thr}, {n_off}, {n}, {mae}).
    ``right='error_vs_distance'``: absolute error against ``color_col`` (hexbin counts)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
    import analyse_inference as ai

    _style()
    f = _fonts(fonts)
    fig, (ax, axr) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace': 0.08})
    t, p = err['mu'].to_numpy(float), err['mu_inferred'].to_numpy(float)
    lo_, hi_ = min(t.min(), p.min()), max(t.max(), p.max())
    lim = (lo_ - 0.03 * (hi_ - lo_), hi_ + 0.03 * (hi_ - lo_))

    def _parity(a, groups=None, **hb_kw):
        hb = a.hexbin(t, p, gridsize=bins, mincnt=1, extent=(*lim, *lim), linewidths=0.2, **hb_kw)
        a.plot(lim, lim, 'k--', lw=0.8, alpha=0.5)
        a.set_xlim(lim); a.set_ylim(lim); a.set_aspect('equal', adjustable='box')
        a.grid(True, alpha=0.25, lw=0.4); a.set_axisbelow(True)
        a.tick_params(axis='both', labelsize=f['tick'])
        a.set_xlabel(xlabel, fontsize=f['label'], fontweight=label_weight)
        if groups is None:
            ai._add_marginal(a, t, p, color=marginal_color, marginal_log=True)
        else:
            _group_marginal(a, t, p, groups, novelty_colors)
        return hb

    def _inset_cbar(a, hb, label, ticks, fmt=None):
        cax = a.inset_axes(list(cbar_box))
        cb = plt.colorbar(hb, cax=cax, orientation='horizontal', ticks=ticks, format=fmt)
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=f['tick'] - 1, length=2, pad=2)
        cax.set_title(label, fontsize=cbar_label_size or f['tick'], pad=3)

    hb = _parity(ax, cmap=cmap_count, norm=LogNorm())
    vmax_count = float(hb.get_array().max())
    _inset_cbar(ax, hb, count_label, _three_ticks(1, vmax_count, log=True),
                fmt=plt.FuncFormatter(lambda v, _: f'$10^{{{int(round(np.log10(v)))}}}$'))
    ann = ai._format_dipole_metric(metric_df, 'mu_inferred', 'R2', t, p)
    ax.text(0.04, 0.96, f'$R^2$ = {ann}', transform=ax.transAxes, ha='left', va='top', fontsize=f['annot'],
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', alpha=0.92))
    ax.set_ylabel(ylabel, fontsize=f['label'], fontweight=label_weight)

    c = err[color_col].to_numpy(float)
    vmin, vmax = novelty_range or (float(np.nanquantile(c, 0.01)), float(np.nanquantile(c, 0.99)))
    cmap = LinearSegmentedColormap.from_list('novelty', list(novelty_colors))
    if right == 'parity':
        from matplotlib.colors import to_rgba
        from matplotlib.lines import Line2D

        groups = _rare_groups(err[count_col].to_numpy(int), rare_hi, group_labels)
        hb2 = _parity(axr, groups, C=c, reduce_C_function=np.mean, cmap=cmap, norm=Normalize(vmin, vmax))
        _inset_cbar(axr, hb2, novelty_label, _three_ticks(vmin, vmax, log=False))
        axr.tick_params(labelleft=False)
        hl = np.zeros(len(err), bool)
        for h in highlights:
            hl |= err[h['col']].to_numpy(bool)
        mae = lambda m: float(np.abs(p[m] - t[m]).mean())
        handles = [Line2D([0], [0], color=col, lw=2.5) for col in novelty_colors]
        texts = [group_fmt.format(label=lab, mae=mae(m), mae_wo=mae(m & ~hl), n=int(m.sum())) for lab, m in groups]
        for h in highlights:
            grp = err[h['col']].to_numpy(bool)
            off = grp & (np.abs(p - t) > highlight_min_err)
            if off.any():
                pc = axr.hexbin(t[off], p[off], gridsize=bins, mincnt=1, extent=(*lim, *lim), zorder=4)
                pc.set_array(None)
                pc.set_facecolor('none')
                pc.set_edgecolor(to_rgba(h['color'], h.get('alpha', highlight_alpha)))
                pc.set_linewidth(h.get('lw', highlight_lw))
            handles.append(Line2D([0], [0], marker='h', ls='', markerfacecolor='none',
                                  markeredgecolor=to_rgba(h['color'], h.get('alpha', highlight_alpha)),
                                  markeredgewidth=1.4, markersize=9))
            texts.append(highlight_fmt.format(**{**h, 'thr': highlight_min_err, 'n_off': int(off.sum()),
                                                 'n': int(grp.sum()), 'mae': mae(grp)}))
        lfs = legend_fontsize or f['legend'] - 1
        leg = axr.legend(handles, texts, **{**dict(loc='upper left', frameon=True, framealpha=0.92, edgecolor='#cccccc',
                                                   fontsize=lfs, handlelength=1.4, handletextpad=0.5, borderpad=0.4,
                                                   labelspacing=0.3, title=legend_title, title_fontsize=lfs),
                                              **(legend_kw or {})})
        leg._legend_box.align = 'left'
    elif right == 'error_vs_distance':
        e = err['abs_err'].to_numpy(float)
        hb2 = axr.hexbin(c, e, gridsize=bins, cmap=cmap_count, norm=LogNorm(), mincnt=1, yscale='log', linewidths=0.2)
        axr.axvline(1.0, color='k', ls='--', lw=0.8)
        axr.grid(True, alpha=0.25, lw=0.4)
        axr.tick_params(axis='both', labelsize=f['tick'])
        axr.set_xlabel(xlabel_distance, fontsize=f['label'], fontweight=label_weight)
        axr.set_ylabel(ylabel_error, fontsize=f['label'], fontweight=label_weight)
        _inset_cbar(axr, hb2, count_label, _three_ticks(1, float(hb2.get_array().max()), log=True),
                    fmt=plt.FuncFormatter(lambda v, _: f'$10^{{{int(round(np.log10(v)))}}}$'))
    else:
        raise ValueError(right)
    for a, ttl in zip((ax, axr), titles):
        if ttl:
            a.set_title(ttl, fontsize=f['title'], fontweight=title_weight)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        root, _ = os.path.splitext(save_path)
        fig.savefig(root + '.svg', bbox_inches='tight')
    return fig


def dipole_outliers(err: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """The ``n`` remainder molecules with the largest absolute dipole error, with both novelty scores
    and their cumulative share of the total absolute error."""
    out = err.sort_values('abs_err', ascending=False).head(n).reset_index(drop=True)
    out['cum_share_of_total_abs_err_%'] = 100 * out['abs_err'].cumsum() / err['abs_err'].sum()
    cols = ['index', 'smiles', 'n_atoms', 'mu', 'mu_inferred', 'abs_err', 'n_rare', 'n_noise', 'n_rare_dist',
            'mean_ratio', 'max_ratio', 'cum_share_of_total_abs_err_%']
    return out[[c for c in cols if c in out]]


def formal_charges(smiles: Sequence[str]) -> pd.DataFrame:
    """Per SMILES: net formal charge and number of atoms with a formal charge (RDKit)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    rows = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        fc = [a.GetFormalCharge() for a in m.GetAtoms()] if m is not None else []
        rows.append((sum(fc) if m is not None else np.nan, sum(c != 0 for c in fc) if m is not None else np.nan))
    return pd.DataFrame(rows, columns=['net_charge', 'n_charged_atoms'])


def charge_separation_table(err: pd.DataFrame, frame: pd.DataFrame, top: int = 100) -> pd.DataFrame:
    """Charge-separated (zwitterion-like: atoms with a formal charge, net charge 0) vs other
    molecules: count and share in AIMEl and in the remainder; on the remainder the dipole MAE,
    share of the total absolute error, count among the ``top`` largest errors, median number of rare
    atoms by the label and by the distance score, share with ≥ 3 distance-rare atoms and the share of
    their atoms that are HDBSCAN noise.  Charges are read from the SMILES of ``frame``."""
    ch = pd.concat([frame[['index', 'in_aimel']].reset_index(drop=True),
                    formal_charges(frame['smiles'].tolist())], axis=1)
    assert ch['net_charge'].notna().all() and (ch['net_charge'] == 0).all(), "a QM9 molecule with net charge != 0"
    ch['charge_separated'] = ch['n_charged_atoms'] > 0
    e = err[['index', 'abs_err']].merge(ch[['index', 'charge_separated']], on='index', how='left', validate='one_to_one')
    r = err[['n_rare', 'n_rare_dist', 'n_noise', 'n_atoms']].reset_index(drop=True)
    top_ids = set(e.nlargest(top, 'abs_err')['index'])
    rows = []
    for flag, name in ((True, 'charge-separated'), (False, 'other')):
        sel = e['charge_separated'] == flag
        rows.append({'molecules': name,
                     'n_AIMEl': int((ch['in_aimel'] & (ch['charge_separated'] == flag)).sum()),
                     '%_of_AIMEl': 100 * float((ch.loc[ch['in_aimel'], 'charge_separated'] == flag).mean()),
                     'n_remainder': int(sel.sum()), '%_of_remainder': 100 * float(sel.mean()),
                     'MAE_D': float(e.loc[sel, 'abs_err'].mean()),
                     'share_of_total_abs_err_%': 100 * float(e.loc[sel, 'abs_err'].sum() / e['abs_err'].sum()),
                     f'n_in_top_{top}_errors': int(e.loc[sel, 'index'].isin(top_ids).sum()),
                     'median_rare_atoms_label': float(r.loc[sel, 'n_rare'].median()),
                     'median_rare_atoms_distance': float(r.loc[sel, 'n_rare_dist'].median()),
                     '%_with_>=3_distance_rare': 100 * float((r.loc[sel, 'n_rare_dist'] >= 3).mean()),
                     'noise_%_of_atoms': 100 * float(r.loc[sel, 'n_noise'].sum() / r.loc[sel, 'n_atoms'].sum())})
    return pd.DataFrame(rows)


# a 2-oxabicyclo[1.1.1]pentane with an OH and an alkyl chain on the one-carbon bridge (bare bridgeheads)
OXABICYCLOPENTANOL_SMARTS = '[CX4;R]([OX2H1])([CX4;!R])1[CX4H1;R]2[CX4H2;R][CX4H1;R]1O2'


ALKYNE_SMARTS = '[C]#[C]'
# the chemical groups behind the largest QT-Net dipole errors: column of outlier_groups() -> description
OUTLIER_GROUPS = {
    'protonated_zwitterion': 'zwitterion whose cation is a protonated sp3 amine (NH₃⁺ / NH₂⁺ / NH⁺)',
    'other_zwitterion': 'zwitterion with another cation (iminium, amidinium, …)',
    'oxabicyclopentanol': '2-oxabicyclo[1.1.1]pentane with an OH and an alkyl chain on the bridge carbon',
    'alkyne': 'a C≡C triple bond (zwitterions excluded)',
}


def outlier_groups(smiles: Sequence[str]) -> pd.DataFrame:
    """Boolean columns per SMILES: ``zwitterion`` (formal charges, net 0) and the groups of
    :data:`OUTLIER_GROUPS`: ``protonated_zwitterion``, ``other_zwitterion``, ``oxabicyclopentanol``
    (:data:`OXABICYCLOPENTANOL_SMARTS`) and ``alkyne`` (:data:`ALKYNE_SMARTS`, zwitterions excluded, so the
    groups do not overlap)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    smiles = list(smiles)
    zw = formal_charges(smiles)['n_charged_atoms'].to_numpy() > 0
    sp3 = np.zeros(len(smiles), bool)
    if zw.any():
        sp3[zw] = zwitterion_types([s for s, z in zip(smiles, zw) if z])['pair'].str.startswith('sp3').to_numpy()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    match = lambda smarts: np.array([m.HasSubstructMatch(Chem.MolFromSmarts(smarts)) for m in mols], bool)
    return pd.DataFrame({'zwitterion': zw, 'protonated_zwitterion': sp3, 'other_zwitterion': zw & ~sp3,
                         'oxabicyclopentanol': match(OXABICYCLOPENTANOL_SMARTS), 'alkyne': match(ALKYNE_SMARTS) & ~zw})


def outlier_group_table(err: pd.DataFrame, frame: pd.DataFrame, min_err: float = 1.0) -> pd.DataFrame:
    """Per group of :data:`OUTLIER_GROUPS` (``err`` carries the :func:`outlier_groups` columns): its
    molecules in AIMEl and in the remainder, the QT-Net dipole MAE on the remainder, the group's
    share of the total absolute and squared error, how many are off by more than ``min_err`` D, and
    the R² of the remainder without the group."""
    aim = outlier_groups(frame.loc[frame['in_aimel'], 'smiles']).sum()
    d = (err['mu_inferred'] - err['mu']).to_numpy()
    t = err['mu'].to_numpy()
    r2 = lambda m: 1 - (d[m] ** 2).sum() / ((t[m] - t[m].mean()) ** 2).sum()
    rows = []
    for g, desc in OUTLIER_GROUPS.items():
        m = err[g].to_numpy(bool)
        rows.append({'group': g, 'description': desc, 'n_AIMEl': int(aim[g]), 'n_remainder': int(m.sum()),
                     '%_of_remainder': 100 * m.mean(), 'MAE_D': float(np.abs(d[m]).mean()),
                     'MAE_others_D': float(np.abs(d[~m]).mean()),
                     'share_of_abs_err_%': 100 * np.abs(d[m]).sum() / np.abs(d).sum(),
                     'share_of_sq_err_%': 100 * (d[m] ** 2).sum() / (d ** 2).sum(),
                     f'n_off_by_>{min_err:g}_D': int((m & (np.abs(d) > min_err)).sum()),
                     'R2_without_group': r2(~m)})
    return pd.DataFrame(rows)


def error_threshold_table(err: pd.DataFrame, thresholds: Sequence[float] = (0.5, 1, 2, 3, 5)) -> pd.DataFrame:
    """Which chemical groups make up the molecules off by more than each threshold (D): the count, the
    share with any C≡C bond (zwitterions included), and the counts of zwitterions, oxabicyclopentanols
    and of molecules in none of the groups."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    q = Chem.MolFromSmarts(ALKYNE_SMARTS)
    cc = err['smiles'].map(lambda smi: Chem.MolFromSmiles(smi).HasSubstructMatch(q)).to_numpy()
    zw, cage = err['zwitterion'].to_numpy(bool), err['oxabicyclopentanol'].to_numpy(bool)
    rows = []
    for thr in thresholds:
        b = err['abs_err'].to_numpy() > thr
        rows.append({'off_by_more_than_D': thr, 'n': int(b.sum()), 'with_C≡C_%': 100 * float(cc[b].mean()),
                     'zwitterions': int((b & zw).sum()), 'oxabicyclopentanols': int((b & cage).sum()),
                     'none_of_these': int((b & ~cc & ~zw & ~cage).sum())})
    return pd.DataFrame(rows)


def pick_group_molecule(err: pd.DataFrame, group: Optional[str] = None, rank: int = 0) -> int:
    """QM9 id of the molecule with the ``rank``-th largest QT-Net dipole error (0 = the largest)
    within ``group``, a column of :func:`outlier_groups`:

    * ``'protonated_zwitterion'``: zwitterion whose cation is a protonated sp3 amine (NH₃⁺ / NH₂⁺ / NH⁺);
    * ``'other_zwitterion'``: zwitterion with another cation (iminium, amidinium, …);
    * ``'oxabicyclopentanol'``: 2-oxabicyclo[1.1.1]pentane with an OH and an alkyl chain on the bridge carbon;
    * ``'alkyne'``: a C≡C triple bond (zwitterions excluded);
    * ``None``: any molecule."""
    if group is not None and group not in OUTLIER_GROUPS:
        raise ValueError(f"group must be None or one of {list(OUTLIER_GROUPS)}")
    sub = err if group is None else err[err[group].to_numpy(bool)]
    return int(sub.sort_values('abs_err', ascending=False)['index'].iloc[rank])


def render_molecule(err: pd.DataFrame, atom_nov: pd.DataFrame, inferred: pd.DataFrame,
                    atom_dist: Optional[pd.DataFrame] = None, *, group: Optional[str] = 'protonated_zwitterion',
                    rank: int = 0, index: Optional[int] = None, interactive: bool = True, static: bool = True,
                    color_by: str = 'distance', near_colors: Optional[dict] = None, far_colors: Optional[dict] = None,
                    threshold_pct: float = 99.0, labels: Optional[str] = None, label_hydrogens: bool = False,
                    width: int = 560, height: int = 440, title_fmt: str = 'QM9 {index}', fonts: Optional[dict] = None,
                    title_weight: str = 'normal', save_path: Optional[str] = None) -> dict:
    """Pick a molecule and render it: the per-atom environment table
    (:func:`molecule_environment_table`), the interactive py3Dmol view (:func:`view_molecule_3d`) and
    the static image (:func:`plot_molecule_3d`).

    group : which chemical group to draw from, by its ``rank``-th largest dipole error (0 = largest):
        ``'protonated_zwitterion'`` zwitterion whose cation is a protonated sp3 amine (NH₃⁺ / NH₂⁺ / NH⁺);
        ``'other_zwitterion'``      zwitterion with another cation (iminium, amidinium, …);
        ``'oxabicyclopentanol'``    2-oxabicyclo[1.1.1]pentane with an OH and an alkyl chain on the bridge carbon;
        ``'alkyne'``                a C≡C triple bond (zwitterions excluded);
        ``None``                    any molecule.
    index : a QM9 id to render instead (overrides ``group`` / ``rank``).

    Returns ``{'index', 'env', 'view', 'fig'}``; ``view`` / ``fig`` are None when not requested."""
    idx = int(index) if index is not None else pick_group_molecule(err, group, rank)
    env = molecule_environment_table(idx, atom_nov, inferred, atom_dist)
    style = dict(color_by=color_by, near_colors=near_colors, far_colors=far_colors, threshold_pct=threshold_pct,
                 labels=labels, label_hydrogens=label_hydrogens)
    view = view_molecule_3d(env, idx, inferred, width=width, height=height, **style) if interactive else None
    fig = (plot_molecule_3d(env, idx, inferred, title=title_fmt.format(index=idx), fonts=fonts,
                            title_weight=title_weight, save_path=save_path, **style) if static else None)
    return {'index': idx, 'env': env, 'view': view, 'fig': fig}


def ensemble_spread(inferred: pd.DataFrame) -> pd.DataFrame:
    """Per molecule: the largest ensemble standard deviation of an atom's inferred dipole contribution
    (norm of Mu_X/Y/Z_std, in D) and its percentile over the frame, i.e. how much the QT-Net
    ensemble members disagree on the molecule's most uncertain atom."""
    sd = [np.linalg.norm(np.column_stack([x, y, z]).astype(float), axis=1).max() * EBOHR_TO_DEBYE
          for x, y, z in zip(inferred['Mu_X_std'], inferred['Mu_Y_std'], inferred['Mu_Z_std'])]
    out = pd.DataFrame({'index': inferred['index'].astype(np.int64).to_numpy(), 'max_atom_mu_std': sd})
    out['max_atom_mu_std_pct'] = 100 * out['max_atom_mu_std'].rank(pct=True)
    return out


def zwitterion_types(smiles: Sequence[str]) -> pd.DataFrame:
    """Per charge-separated SMILES: ``pair`` (sp3 ammonium or sp2 cation, × carboxylate or other
    anion) and ``separation_bonds``, the shortest bond path between a positive and a negative atom."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    carboxylate = Chem.MolFromSmarts('[CX3](=O)[O-]')
    rows = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        pos = [a for a in m.GetAtoms() if a.GetFormalCharge() > 0]
        neg = [a for a in m.GetAtoms() if a.GetFormalCharge() < 0]
        D = Chem.GetDistanceMatrix(m)
        sp3 = any(a.GetSymbol() == 'N' and a.GetHybridization() == Chem.HybridizationType.SP3 for a in pos)
        rows.append({'pair': ('sp3 ammonium' if sp3 else 'sp2 cation') + ' + ' +
                             ('carboxylate' if m.HasSubstructMatch(carboxylate) else 'other anion'),
                     'separation_bonds': int(min(D[a.GetIdx(), b.GetIdx()] for a in pos for b in neg))})
    return pd.DataFrame(rows)


def zwitterion_table(err: pd.DataFrame, frame: pd.DataFrame, flag_col: str = 'zwitterion',
                     min_err: float = 1.0) -> pd.DataFrame:
    """Charge-separated molecules by type (:func:`zwitterion_types`): count in AIMEl and in the
    remainder, and on the remainder how many lie off the parity diagonal (absolute error >
    ``min_err`` D) and how many of those are under-predicted, with the median |μ|, absolute error,
    charge separation (bonds), largest atom distance to AIMEl / p99 and number of noise atoms."""
    z = err[err[flag_col]].reset_index(drop=True)
    z = pd.concat([z, zwitterion_types(z['smiles'].tolist())], axis=1)
    z['off'] = z['abs_err'] > min_err
    z['under'] = z['off'] & (z['mu_inferred'] < z['mu'])
    aim = frame.loc[frame['in_aimel'], 'smiles']
    aim = aim[formal_charges(aim.tolist())['n_charged_atoms'].to_numpy() > 0]
    n_aim = zwitterion_types(aim.tolist())['pair'].value_counts()
    t = z.groupby('pair').agg(n_remainder=('index', 'size'), n_off_diagonal=('off', 'sum'),
                              n_under_predicted=('under', 'sum'), median_mu=('mu', 'median'),
                              median_abs_err=('abs_err', 'median'), median_separation_bonds=('separation_bonds', 'median'),
                              median_max_ratio=('max_ratio', 'median'), median_noise_atoms=('n_noise', 'median'))
    t.insert(0, 'n_AIMEl', n_aim.reindex(t.index).fillna(0).astype(int))
    t.insert(3, 'off_diagonal_%', 100 * t['n_off_diagonal'] / t['n_remainder'])
    return t.reset_index()


def molecule_environment_table(index: int, atom_nov: pd.DataFrame, inferred: pd.DataFrame,
                               atom_dist: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Every atom of one QM9 molecule (by QM9 id): element, 6-Bohr environment label, its AIMEl
    support and label-novelty percentile with the rare / remainder-only / noise flags; the 6-Bohr
    distance to AIMEl (percentile, ratio to p99, rare) when ``atom_dist`` is given; and the
    QT-Net-inferred atomic properties (N, LI, |Mu| with the ensemble standard deviations)."""
    a = atom_nov[atom_nov['index'] == index].sort_values('atom_idx')
    row = inferred[inferred['index'].astype(np.int64) == index]
    if a.empty or row.empty:
        raise KeyError(f"QM9 id {index} not found")
    row = row.iloc[0]
    if [str(s) for s in row['elements']] != a['element'].tolist():
        raise AssertionError(f"{index}: element order of qm9_inferred and of the labels differ")
    out = a[['atom_idx', 'element', 'label', 'is_noise', 'support', 'novelty_pct', 'rare99', 'novel']].reset_index(drop=True)
    out = out.rename(columns={'novelty_pct': 'label_pct', 'rare99': 'label_rare'})
    if atom_dist is not None:
        d = atom_dist[atom_dist['index'] == index][['atom_idx', 'novelty_pct', 'ratio', 'rare99']]
        out = out.merge(d.rename(columns={'novelty_pct': 'dist_pct', 'ratio': 'dist_ratio', 'rare99': 'dist_rare'}),
                        on='atom_idx', how='left')
    for c in ('N', 'LI'):
        out[c] = np.asarray(row[c], float)
        out[f'{c}_std'] = np.asarray(row[f'{c}_std'], float)
    mu = np.column_stack([np.asarray(row[f'Mu_{k}'], float) for k in 'XYZ'])
    mu_std = np.column_stack([np.asarray(row[f'Mu_{k}_std'], float) for k in 'XYZ'])
    out['|Mu|'] = np.linalg.norm(mu, axis=1)
    out['Mu_std'] = np.linalg.norm(mu_std, axis=1)
    return out


def _atom_colors(env: pd.DataFrame, color_by: str = 'distance', near_colors: Optional[dict] = None,
                 far_colors: Optional[dict] = None, threshold_pct: float = 99.0):
    """Per-atom hex colours.  'distance' (default): the CPK colour of the element when the atom's
    6-Bohr distance percentile to the training distribution (``dist_pct``) is at most
    ``threshold_pct``, the element's held-out colour of the leakage analysis above it (atoms without a
    distance stay CPK).  'element': CPK only."""
    near = {**CPK, **(near_colors or {})}
    far = {**FAR_COLORS, **(far_colors or {})}
    if color_by == 'element' or 'dist_pct' not in env:
        return [near[e] for e in env['element']]
    if color_by == 'distance':
        return [far[e] if (np.isfinite(q) and q > threshold_pct) else near[e] for e, q in zip(env['element'], env['dist_pct'])]
    raise ValueError(color_by)


def _atom_texts(env: pd.DataFrame, labels: Optional[str], label_hydrogens: bool = False) -> list:
    """'label' (environment label), 'distance' (6-Bohr distance percentile), 'index', or None."""
    if labels is None:
        return [''] * len(env)
    if labels == 'label':
        t = list(env['label'])
    elif labels == 'distance':
        t = ['' if np.isnan(v) else f'{v:.0f}' for v in env.get('dist_pct', pd.Series(np.nan, index=env.index))]
    elif labels == 'index':
        t = [f'{e}{i}' for e, i in zip(env['element'], env['atom_idx'])]
    else:
        raise ValueError(labels)
    return [s if (label_hydrogens or e != 'H') else '' for s, e in zip(t, env['element'])]


def _molecule_xyz(index: int, inferred: pd.DataFrame) -> tuple:
    row = inferred[inferred['index'].astype(np.int64) == index].iloc[0]
    pos = BOHR_TO_A * np.column_stack([np.asarray(row[f'position_{k}'], float) for k in 'xyz'])
    return [str(s) for s in row['elements']], pos


def view_molecule_3d(env: pd.DataFrame, index: int, inferred: pd.DataFrame, *, color_by: str = 'distance',
                     near_colors: Optional[dict] = None, far_colors: Optional[dict] = None, threshold_pct: float = 99.0,
                     labels: Optional[str] = None, label_hydrogens: bool = False, label_size: int = 10,
                     width: int = 560, height: int = 440):
    """Interactive py3Dmol view of one QM9 molecule (QT-Net frame, Å).  ``env`` is its
    :func:`molecule_environment_table`; atoms are CPK-coloured, or in their element's held-out colour
    when farther from the training distribution than ``threshold_pct`` (see :func:`_atom_colors`);
    optional text labels.  Returns the view (``.show()``)."""
    import py3Dmol
    els, pos = _molecule_xyz(index, inferred)
    xyz = f'{len(els)}\nQM9 {index}\n' + '\n'.join(f'{e} {x:.5f} {y:.5f} {z:.5f}' for e, (x, y, z) in zip(els, pos))
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz, 'xyz')
    for i, c in enumerate(_atom_colors(env, color_by, near_colors, far_colors, threshold_pct)):
        view.setStyle({'index': i}, {'stick': {'radius': 0.14, 'color': c},
                                     'sphere': {'scale': 0.28 if els[i] != 'H' else 0.2, 'color': c}})
    for (x, y, z), txt in zip(pos, _atom_texts(env, labels, label_hydrogens)):
        if txt:
            view.addLabel(txt, {'position': {'x': float(x), 'y': float(y), 'z': float(z)}, 'fontSize': label_size,
                                'fontColor': 'black', 'backgroundColor': 'white', 'backgroundOpacity': 0.6})
    view.zoomTo()
    return view


def plot_molecule_3d(env: pd.DataFrame, index: int, inferred: pd.DataFrame, *, color_by: str = 'distance',
                     near_colors: Optional[dict] = None, far_colors: Optional[dict] = None, threshold_pct: float = 99.0,
                     labels: Optional[str] = None, label_hydrogens: bool = False, title: Optional[str] = None,
                     fonts: Optional[dict] = None, title_weight: str = 'normal', elev: float = 20, azim: float = 35,
                     figsize=(5.6, 5.0), save_path=None):
    """Static ball-and-stick image of the same molecule (matplotlib 3D), for viewers without
    JavaScript; same colours as :func:`view_molecule_3d`.  Bonds: atom pairs closer than 1.15 × the
    sum of their covalent radii."""
    import matplotlib.pyplot as plt
    f = _fonts(fonts)
    els, pos = _molecule_xyz(index, inferred)
    cols = _atom_colors(env, color_by, near_colors, far_colors, threshold_pct)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    for i in range(len(els)):
        for j in range(i + 1, len(els)):
            if np.linalg.norm(pos[i] - pos[j]) < 1.15 * (COVALENT_RADIUS_A[els[i]] + COVALENT_RADIUS_A[els[j]]):
                ax.plot(*zip(pos[i], pos[j]), color='#777777', lw=2.0, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=cols, s=[90 if e == 'H' else 260 for e in els],
               edgecolors='k', linewidths=0.6, depthshade=False, zorder=2)
    for (x, y, z), txt in zip(pos, _atom_texts(env, labels, label_hydrogens)):
        if txt:
            ax.text(x, y, z, f'  {txt}', fontsize=f['annot'] - 1)
    rng = (pos.max(0) - pos.min(0)).max() / 2 + 0.5
    ctr = (pos.max(0) + pos.min(0)) / 2
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), ctr):
        setter(c - rng, c + rng)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title if title is not None else f'QM9 {index}', fontsize=f['title'], fontweight=title_weight)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def write_molecule_html(env: pd.DataFrame, index: int, inferred: pd.DataFrame, path: str, *,
                        color_by: str = 'distance', near_colors: Optional[dict] = None,
                        far_colors: Optional[dict] = None, threshold_pct: float = 99.0,
                        rotate: Sequence[tuple] = (), width: int = 800, height: int = 800) -> str:
    """The interactive view of :func:`view_molecule_3d` as a standalone page (white background, no
    labels), for a screenshot (:func:`screenshot_html`, or by hand from a browser).  ``rotate``:
    ``(degrees, axis)`` pairs applied in order, axis 'x', 'y' or 'z' of the screen."""
    view = view_molecule_3d(env, index, inferred, color_by=color_by, near_colors=near_colors,
                            far_colors=far_colors, threshold_pct=threshold_pct, width=width, height=height)
    view.setBackgroundColor('white')
    for deg, axis in rotate:
        view.rotate(deg, axis)
    view.zoomTo()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    view.write_html(path)
    return path


CHROMIUM_CANDIDATES = ('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                       '/Applications/Chromium.app/Contents/MacOS/Chromium',
                       '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
                       '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
                       'google-chrome', 'chromium', 'chromium-browser')


def screenshot_html(html: str, png: str, *, browser: Optional[str] = None, width: int = 800, height: int = 800,
                    wait_ms: int = 8000) -> str:
    """PNG of a page rendered by a headless Chromium-based browser (``browser``: its executable;
    default the first of :data:`CHROMIUM_CANDIDATES` found).  WebGL runs in software (SwiftShader);
    ``wait_ms`` of virtual time lets 3Dmol.js load and draw.  A throwaway profile is used, so the
    browser's own profile is never touched."""
    import shutil
    import subprocess
    import tempfile
    exe = browser or next((c for c in CHROMIUM_CANDIDATES if os.path.exists(c) or shutil.which(c)), None)
    if exe is None:
        raise FileNotFoundError('no Chromium-based browser found; pass browser=, or take the screenshot by hand')
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([exe, '--headless=new', f'--user-data-dir={profile}', '--use-angle=swiftshader',
                        '--enable-unsafe-swiftshader', '--hide-scrollbars', f'--window-size={width},{height}',
                        f'--virtual-time-budget={wait_ms}', f'--screenshot={os.path.abspath(png)}',
                        'file://' + os.path.abspath(html)], check=True, capture_output=True, timeout=120)
    return png


def tile_images(paths: Sequence[str], save_path: str, *, gap: int = 40, margin: int = 10,
                height: Optional[int] = None) -> str:
    """One row of images, each cropped to its non-white content and scaled to a common height
    (default: the smallest cropped height), ``gap`` pixels apart: the screenshots of
    :func:`screenshot_html` as one figure, with no text."""
    from PIL import Image, ImageChops
    ims = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        bbox = ImageChops.difference(im, Image.new('RGB', im.size, 'white')).getbbox()
        ims.append(im.crop(bbox) if bbox else im)
    h = height or min(im.height for im in ims)
    ims = [im.resize((round(im.width * h / im.height), h), Image.LANCZOS) for im in ims]
    out = Image.new('RGB', (sum(im.width for im in ims) + gap * (len(ims) - 1) + 2 * margin, h + 2 * margin), 'white')
    x = margin
    for im in ims:
        out.paste(im, (x, margin))
        x += im.width + gap
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    out.save(save_path, dpi=(300, 300))
    return save_path
