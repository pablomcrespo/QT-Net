#!/usr/bin/env python3
"""
Morgan-fingerprint chemotypes of the whole QM9 (the chemotype axis of the
environment-holdout study).

Recipe: ECFP (binary, radius 2, 2048 bits) → L2-normalise → PCA(50,
randomized SVD) → k-means(50, n_init=10), seed 42, fit ONCE on all 128,908
molecules (AIMEl ∪ remainder, in the order recorded in
``qm9_classification_order.json.gz``).  Output: ``qm9_chemotypes.pkl.gz`` with
``index`` (QM9 id) and ``fp_cluster`` ('fp_<k>'; 'fp_-1' = SMILES that did not
parse), in that order.

Why this recipe (each choice measured on QM9):
  * k-means, not density clustering: ECFP space is near-uniform (mean pairwise
    Tanimoto ≈ 0.08), so HDBSCAN puts 60–100 % of molecules in noise; a
    holdout wants every molecule in exactly one holdable group.
  * 50 PCA dimensions from the plateau of η² (property variance explained by
    the cluster labels), which is flat from ~10 dimensions; there is no
    variance elbow (50 dims keep ~42 % of the variance).
  * 2048 bits: ~20 active features per molecule, so collisions are negligible.
The acceptance gate is η²(property | chemotype) against label-shuffled
chemotypes on AIMEl: the ratio must be ≫ 1 for the holdout to be
out-of-distribution with respect to the targets (a random or topology-based
split scores ≈ 1).  ECFP: Rogers & Hahn, J. Chem. Inf. Model. 50 (2010) 742.

THE SHIPPED LABELS DEFINE THE SPLIT, NOT A RE-RUN.  k-means keeps the best of
``n_init`` local optima, and near-tied optima are decided by floating-point
reduction order, i.e. by the BLAS / OpenMP thread count: re-running this script
with identical inputs and seed on our cluster gave adjusted Rand index 0.9996
(16 threads), 0.998 (1 thread) and 0.66 (8 threads) against the shipped
``qm9_chemotypes.pkl.gz``, which is the labelling every experiment used.  The
default output therefore does not overwrite it.

Usage:
    python qm9_chemotype_clustering.py
    python qm9_chemotype_clustering.py --sweep-dims 5,10,20,35,50,75,100   # η² vs dims, no output
"""

import argparse
import os
import sys
import time

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

from qtnet.data_utils import MOLECULAR_PROPERTIES_PRED, load_qm9_classification_frame

MOLECULAR_DIR = os.path.join(REPO_ROOT, 'data_curation', 'molecular')


def morgan_matrix(smiles, radius=2, n_bits=2048):
    """Binary ECFP → (n_valid, n_bits) float32, plus the mask of rows that parsed."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles), n_bits), dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)
    t0 = time.time()
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        X[i] = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        valid[i] = True
    print(f"  fingerprints: {valid.sum():,}/{len(smiles):,} parsed ({time.time() - t0:.0f}s)", flush=True)
    return X[valid], valid


def pca_kmeans(feats, n_components, k, seed):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=seed)
    Z = pca.fit_transform(feats)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Z)
    return labels, float(pca.explained_variance_ratio_.sum())


def eta_squared(y, labels):
    """SS_between / SS_total of ``y`` grouped by ``labels`` (label -1 ignored)."""
    y, labels = np.asarray(y, float), np.asarray(labels)
    m = labels != -1
    y, labels = y[m], labels[m]
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    ss_bet = sum((labels == g).sum() * (y[labels == g].mean() - grand) ** 2 for g in np.unique(labels))
    return float(ss_bet / ss_tot) if ss_tot > 0 else float('nan')


def eta2_gate(frame, labels, seed, shuffles=20):
    """η² of each target on AIMEl, observed vs label-shuffled."""
    rng = np.random.default_rng(seed)
    aim = frame['in_aimel'].to_numpy()
    lab = labels[aim]
    rows = []
    for p in MOLECULAR_PROPERTIES_PRED:
        y = frame.loc[aim, p].to_numpy()
        obs = eta_squared(y, lab)
        rand = float(np.mean([eta_squared(y, rng.permutation(lab)) for _ in range(shuffles)]))
        rows.append((p, obs, rand, obs / rand))
    return pd.DataFrame(rows, columns=['property', 'eta2', 'eta2_shuffled', 'ratio'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--aimel', default=os.path.join(MOLECULAR_DIR, 'aimel_clustered_molecular.pkl'))
    ap.add_argument('--remainder', default=os.path.join(MOLECULAR_DIR, 'qm9_filtered.pkl'))
    ap.add_argument('--order', default=os.path.join(MOLECULAR_DIR, 'qm9_classification_order.json.gz'))
    ap.add_argument('--out', default=os.path.join(MOLECULAR_DIR, 'qm9_chemotypes_rerun.pkl.gz'),
                    help='output labels (default does NOT overwrite the shipped qm9_chemotypes.pkl.gz)')
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--fp-size', type=int, default=2048)
    ap.add_argument('-k', '--n-clusters', type=int, default=50)
    ap.add_argument('--n-components', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sweep-dims', default=None,
                    help='comma-separated PCA dims: print η² vs dims and exit (justifies the cut)')
    a = ap.parse_args()

    from sklearn.preprocessing import normalize

    frame = load_qm9_classification_frame(a.aimel, a.remainder, a.order)
    print(f"QM9: {len(frame):,} molecules ({int(frame['in_aimel'].sum()):,} in AIMEl) | "
          f"ECFP r={a.radius} {a.fp_size} bits | PCA {a.n_components} | k-means {a.n_clusters}", flush=True)
    X, valid = morgan_matrix(frame['smiles'].tolist(), a.radius, a.fp_size)
    feats = normalize(X)

    def full_labels(lv):
        out = np.full(len(frame), -1, dtype=int)
        out[np.where(valid)[0]] = lv
        return out

    if a.sweep_dims:
        for d in [int(x) for x in a.sweep_dims.split(',')]:
            lv, var = pca_kmeans(feats, d, a.n_clusters, a.seed)
            g = eta2_gate(frame, full_labels(lv), a.seed, shuffles=5)
            print(f"  dims {d:>4}: variance kept {var:.3f} | η² " +
                  ' '.join(f"{r.property} {r.eta2:.3f}" for r in g.itertuples()), flush=True)
        return

    lv, var = pca_kmeans(feats, a.n_components, a.n_clusters, a.seed)
    labels = full_labels(lv)
    sizes = pd.Series(labels[labels != -1]).value_counts()
    aim_groups = set(labels[frame['in_aimel'].to_numpy()]) - {-1}
    print(f"  PCA keeps {var:.3f} of the variance | {sizes.size} chemotypes, size "
          f"{sizes.min()}–{sizes.max()} (median {int(sizes.median())}) | unparsed {int((labels == -1).sum())} | "
          f"chemotypes with no AIMEl molecule: {sorted(set(sizes.index) - aim_groups)}")
    gate = eta2_gate(frame, labels, a.seed)
    print("  η² acceptance gate on AIMEl (ratio ≫ 1 ⇒ the chemotype holdout is OOD for the targets):")
    print(gate.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = pd.DataFrame({'index': frame['index'].to_numpy(),
                        'fp_cluster': [f'fp_{int(k)}' for k in labels]})
    out.to_pickle(a.out)
    print(f"  wrote {a.out} ({len(out):,} rows)")


if __name__ == '__main__':
    main()
