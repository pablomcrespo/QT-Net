"""
Holdout novelty as a function of perception radius (SOAP cutoff).

The OOD holdout ``{H_10, C_11, N_13, O_10}`` was defined by clustering SOAP
descriptors at ``r_cut = 4.5`` Bohr, but the models build their message-passing
graph at a larger cutoff (EGNN ~5.25 Bohr, SG-8 8 Bohr). An environment that is
novel at 4.5 Bohr can have near-duplicates in training once the neighbourhood
widens -- i.e. the holdout "leaks" at the model's radius. We measure this
directly and *without clustering* (so the answer does not depend on HDBSCAN
tuning).

Novelty metric (per element, per radius ``r``):
    1. SOAP @ r          -- rotation/translation/permutation-invariant vector.
    2. PCA on the full dataset -- low-dim coords (the embedding the holdout was
       selected in); the retained variance is held constant across radii so only
       ``r`` changes.
    3. whiten by training std -- Euclidean distance becomes Mahalanobis distance
       w.r.t. the training distribution.
    4. d_k               -- distance to the 15th-nearest training atom (coverage).
    5. calibrate d_k against a held-out split of *training* atoms -> novelty_pct
       in [0, 100]: 50 = as isolated as the median training atom (effectively
       seen), ~100 = training's extreme tail (genuinely OOD).

``median_novelty_pct`` per held-out cluster is the leakage metric -- it falls
toward 50 as the holdout leaks. ``stratified_ccc_by_novelty`` then checks whether
the leaked atoms inflated the reported OOD accuracy.

Pipeline (heavy SOAP/PCA cached under ``cache_dir``):
    stream_soap_radius -> project_pca_radius -> leakage_at_radius -> leakage_curve
    stratified_ccc_by_novelty  -- accuracy on leaked vs novel atoms.

Reuses ``sanitizing_clusters`` for the streamed-covariance PCA, the holdout
loader, and shared constants.
"""

from __future__ import annotations

import os
import sys
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ase import Atoms

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sanitizing_clusters as sc  # noqa: E402
from atomic_env_split import compute_soap_descriptors  # noqa: E402
from result_analysis import (  # noqa: E402
    df_to_atom_table,
    concordance_correlation_coefficient,
)

ELEMENTS = sc.ELEMENTS
RANDOM_STATE = sc.RANDOM_STATE
HOLDOUT_TARGET_LABELS = sc.HOLDOUT_TARGET_LABELS
PROPERTIES = sc.PROPERTIES

# Radii to sweep (Bohr): 4.5 = holdout-defining radius, 5.25 = EGNN, 8.0 = SG-8.
DEFAULT_RADII: Tuple[float, ...] = (4.5, 5.25, 8.0)

# Per-element PCA component counts of the original 4.5-Bohr clustering. The
# 4.5-Bohr novelty reuses THIS projection so it matches the basis the holdout was
# defined in; other radii keep enough components for the same retained variance
# (below), so the sweep varies the radius, not the information budget.
ORIGINAL_PCA_COMPONENTS: Dict[str, int] = {"H": 20, "C": 35, "N": 30, "O": 20}
DEFAULT_FIXED_COMPONENTS: Dict[float, Dict[str, int]] = {4.5: ORIGINAL_PCA_COMPONENTS}
ORIGINAL_PCA_VARIANCE: Dict[str, float] = {
    "O": 0.9904, "H": 0.9911, "C": 0.9901, "N": 0.9918,
}

# Per-radius SOAP params share everything except r_cut with the model setup.
_BASE_SOAP = {k: v for k, v in sc.SOAP8_PARAMS.items() if k != "r_cut"}


def _soap_params(r_cut: float) -> Dict:
    return {"r_cut": float(r_cut), **_BASE_SOAP}


def _radius_tag(r_cut: float) -> str:
    # 8.0 -> "8", 5.25 -> "5.25"
    return str(int(r_cut)) if float(r_cut).is_integer() else str(r_cut)


# ---------------------------------------------------------------------------
# 1. SOAP @ r -> per-element disk memmaps
# ---------------------------------------------------------------------------

def stream_soap_radius(
    aimel_df: pd.DataFrame,
    r_cut: float,
    cache_dir: str,
    batch_size: int = 400,
) -> Dict[str, Dict]:
    """SOAP at ``r_cut`` for every aimel atom, streamed to per-element memmaps.

    Returns ``meta[el] = {path, n_atoms, n_features, index}`` with ``index`` a
    list of ``(mol_pos, atom_idx)`` in memmap row order. Cached as
    ``soap{tag}_memmap_meta.pkl``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tag = _radius_tag(r_cut)
    meta_path = os.path.join(cache_dir, f"soap{tag}_memmap_meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)
        print(f"[cache] loaded SOAP@{r_cut} memmap meta from {meta_path}")
        return meta

    work = aimel_df.reset_index(drop=True)
    n_mol = len(work)
    soap_params = _soap_params(r_cut)

    counts = {el: 0 for el in ELEMENTS}
    for _, row in work.iterrows():
        for el in row["atom"]:
            counts[str(el)] += 1
    print(f"[soap{tag}] atom counts:", {k: f"{v:,}" for k, v in counts.items()})

    n_feat: Optional[int] = None
    writers, cursors, index = {}, {el: 0 for el in ELEMENTS}, {el: [] for el in ELEMENTS}

    for start in range(0, n_mol, batch_size):
        stop = min(start + batch_size, n_mol)
        for mol_pos in range(start, stop):
            row = work.iloc[mol_pos]
            symbols = [str(s) for s in row["atom"]]
            pos = np.column_stack([row["position_x"], row["position_y"], row["position_z"]])
            soap = compute_soap_descriptors(Atoms(symbols=symbols, positions=pos), soap_params)
            if n_feat is None:
                n_feat = soap.shape[1]
                for el in ELEMENTS:
                    path = os.path.join(cache_dir, f"soap{tag}_{el}.f32")
                    writers[el] = np.memmap(path, dtype=np.float32, mode="w+",
                                            shape=(counts[el], n_feat))
            for atom_idx, (el, vec) in enumerate(zip(symbols, soap)):
                writers[el][cursors[el]] = vec.astype(np.float32)
                index[el].append((mol_pos, atom_idx))
                cursors[el] += 1
        if start % (batch_size * 10) == 0:
            print(f"[soap{tag}] {stop}/{n_mol} molecules ({100*stop/n_mol:.1f}%)", flush=True)

    for el in ELEMENTS:
        writers[el].flush()
        del writers[el]

    meta = {el: {"path": os.path.join(cache_dir, f"soap{tag}_{el}.f32"),
                 "n_atoms": counts[el], "n_features": n_feat,
                 "index": index[el]} for el in ELEMENTS}
    with open(meta_path, "wb") as fh:
        pickle.dump(meta, fh)
    print(f"[cache] wrote SOAP@{r_cut} memmap meta to {meta_path}")
    return meta


# ---------------------------------------------------------------------------
# 2. PCA (streamed covariance) -> projected coordinates
# ---------------------------------------------------------------------------

def project_pca_radius(
    element: str,
    meta: Dict[str, Dict],
    r_cut: float,
    cache_dir: str,
    pca_variance: float = 0.99,
    n_components: Optional[int] = None,
    chunk: int = 50_000,
) -> Dict:
    """Project one element's SOAP@r memmap to PCA space.

    If ``n_components`` is given, keep exactly that many components (used at 4.5
    Bohr to reproduce the original H20/C35/N30/O20 projection); otherwise keep
    enough for ``pca_variance``. The basis is fit on *all* aimel atoms (the
    coordinate system the holdout was selected in); the train/holdout split and
    whitening happen later. Cached per element/radius/component-spec.
    """
    tag = _radius_tag(r_cut)
    spec = f"k{n_components}" if n_components is not None else f"v{pca_variance}"
    cache = os.path.join(cache_dir, f"pcaproj_r{tag}_{element}_{spec}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            out = pickle.load(fh)
        print(f"[cache] loaded PCA proj {element}@{r_cut}: k={out['k']} "
              f"({out['var_retained']:.1f}% var)")
        return out

    m = meta[element]
    mm = np.memmap(m["path"], dtype=np.float32, mode="r",
                   shape=(m["n_atoms"], m["n_features"]))

    mean, cov = sc._streaming_mean_cov(mm)
    evals, evecs = np.linalg.eigh(cov)          # ascending
    evals, evecs = evals[::-1], evecs[:, ::-1]  # descending
    evals = np.clip(evals, 0, None)
    ratio = np.cumsum(evals) / evals.sum()
    if n_components is not None:
        k = int(min(n_components, len(evals)))
    else:
        k = int(np.searchsorted(ratio, pca_variance) + 1)
    var_retained = float(ratio[k - 1] * 100)
    W = evecs[:, :k]
    evr = evals[:k]

    X_pca = np.empty((m["n_atoms"], k), dtype=np.float32)
    for i in range(0, m["n_atoms"], chunk):
        X = np.asarray(mm[i:i + chunk], dtype=np.float64) - mean
        X_pca[i:i + chunk] = (X @ W).astype(np.float32)

    print(f"[pca:{element}@{r_cut}] {m['n_features']} -> {k} comps "
          f"({var_retained:.2f}% var) on {m['n_atoms']:,} atoms")

    out = {"element": element, "X_pca": X_pca, "evr": evr, "index": m["index"],
           "k": k, "var_retained": var_retained}
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


# ---------------------------------------------------------------------------
# 3. Mahalanobis k-NN novelty of the held-out atoms vs training atoms
# ---------------------------------------------------------------------------

def _query_mask(
    proj: Dict,
    aimel_df: pd.DataFrame,
    holdout_pos: np.ndarray,
    target_labels: Optional[set],
) -> np.ndarray:
    """Boolean over ``proj['index']`` rows: held-out atoms in the OOD query set.
    With ``target_labels`` set, restrict to held-out atoms whose original
    4.5-label is one of them (the atoms OOD-CCC is reported on)."""
    idx = proj["index"]
    mol_pos = np.fromiter((t[0] for t in idx), dtype=np.int64, count=len(idx))
    in_hold = np.isin(mol_pos, holdout_pos)
    if target_labels is None:
        return in_hold
    work = aimel_df.reset_index(drop=True)
    acl = work["atom_cluster_labels"].tolist()
    is_target = np.fromiter(
        (acl[t[0]][t[1]] in target_labels for t in idx),
        dtype=bool, count=len(idx))
    return in_hold & is_target


def leakage_at_radius(
    proj: Dict,
    aimel_df: pd.DataFrame,
    holdout_pos: np.ndarray,
    r_cut: float,
    target_labels: Optional[set] = HOLDOUT_TARGET_LABELS,
    knn: int = 15,
    n_ref_max: int = 150_000,
    n_calib_max: int = 100_000,
    whiten: bool = True,
    per_label: bool = True,
    leak_pct: float = 50.0,
    seed: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrated novelty of the held-out OOD atoms vs training, at one radius.

    Reference and calibration sets are disjoint splits of the *training* atoms
    (atoms in non-holdout molecules): the k-NN index is built on the reference,
    the calibration atoms' d_k to it gives the null distribution, and the query
    (held-out) atoms are scored against the same reference. Distances are in
    training-whitened PCA space (Mahalanobis) when ``whiten``. An atom is
    ``leaked`` if its ``novelty_pct`` is below ``leak_pct`` (default 50, the
    median training atom) -- i.e. it sits inside training density.

    Returns ``(atom_novelty, summary)``: per-atom novelty (file, mol_pos,
    atom_idx, label4, dk, novelty_pct, leaked) and one summary row per element
    (and per target label) with n, median_novelty_pct, leak_frac.
    """
    from sklearn.neighbors import NearestNeighbors

    el = proj["element"]
    rng = np.random.default_rng(seed)

    idx = proj["index"]
    mol_pos = np.fromiter((t[0] for t in idx), dtype=np.int64, count=len(idx))
    atom_idx = np.fromiter((t[1] for t in idx), dtype=np.int64, count=len(idx))
    in_hold = np.isin(mol_pos, holdout_pos)

    X = proj["X_pca"].astype(np.float32)
    if whiten:
        std = X[~in_hold].std(axis=0).astype(np.float32)
        std[std == 0] = 1.0
        X = X / std

    q_mask = _query_mask(proj, aimel_df, holdout_pos, target_labels)
    Xq = X[q_mask]
    if len(Xq) == 0:
        return (pd.DataFrame(), pd.DataFrame([{"element": el, "r_cut": r_cut, "n": 0}]))

    # disjoint training reference / calibration split (disjoint so a calibration
    # atom never self-matches in the k-NN index, which would bias its d_k to 0).
    train_rows = np.where(~in_hold)[0]
    perm = rng.permutation(len(train_rows))
    n_ref = min(n_ref_max, int(0.7 * len(train_rows)))
    ref_rows = train_rows[perm[:n_ref]]
    cal_rows = train_rows[perm[n_ref:n_ref + min(n_calib_max, len(train_rows) - n_ref)]]

    nn = NearestNeighbors(n_neighbors=knn, n_jobs=-1).fit(X[ref_rows])
    d_cal = nn.kneighbors(X[cal_rows])[0][:, -1]
    d_q = nn.kneighbors(Xq)[0][:, -1]

    d_cal_sorted = np.sort(d_cal)
    pct = np.searchsorted(d_cal_sorted, d_q, side="right") / len(d_cal_sorted) * 100.0

    work = aimel_df.reset_index(drop=True)
    acl = work["atom_cluster_labels"].tolist()
    files = work["file"].to_numpy()
    q_pos = np.where(q_mask)[0]
    label4 = [acl[mol_pos[i]][atom_idx[i]] for i in q_pos]

    atom_novelty = pd.DataFrame({
        "element": el, "r_cut": r_cut,
        "file": files[mol_pos[q_mask]],
        "mol_pos": mol_pos[q_mask], "atom_idx": atom_idx[q_mask],
        "label4": label4, "dk": d_q.astype(np.float32),
        "novelty_pct": pct.astype(np.float32),
    })
    atom_novelty["leaked"] = atom_novelty["novelty_pct"] < leak_pct

    def _summ(sub: pd.DataFrame, label: str) -> Dict:
        return {
            "element": el, "r_cut": r_cut, "label": label, "n": len(sub),
            "median_novelty_pct": round(float(sub["novelty_pct"].median()), 1),
            "leak_frac": round(float((sub["novelty_pct"] < leak_pct).mean()), 4),
        }

    rows = [_summ(atom_novelty, f"{el}_ALL")]
    if per_label and target_labels is not None:
        for lab, sub in atom_novelty.groupby("label4"):
            rows.append(_summ(sub, lab))

    s = rows[0]
    print(f"[leak:{el}@{r_cut}] n={s['n']:,} median_pct={s['median_novelty_pct']:.0f} "
          f"leak(<{leak_pct:.0f})={100*s['leak_frac']:.1f}%", flush=True)
    return atom_novelty, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. assemble the novelty-vs-radius sweep
# ---------------------------------------------------------------------------

def leakage_curve(
    aimel_df: pd.DataFrame,
    holdout_pos: np.ndarray,
    cache_dir: str,
    radii: Tuple[float, ...] = DEFAULT_RADII,
    target_labels: Optional[set] = HOLDOUT_TARGET_LABELS,
    knn: int = 15,
    elements: List[str] = ELEMENTS,
    whiten: bool = True,
    leak_pct: float = 50.0,
    fixed_components: Optional[Dict[float, Dict[str, int]]] = None,
    pca_variance: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the whole sweep. Returns ``(summary, atom_novelty)``.

    ``summary`` is one row per (radius, element, label); ``atom_novelty``
    concatenates the per-atom tables across radii. Heavy SOAP/PCA is cached.

    ``fixed_components`` maps ``r_cut -> {element: n_components}`` (defaults to the
    original H20/C35/N30/O20 at 4.5 Bohr, so the 4.5 novelty matches the holdout's
    own projection); ``pca_variance`` maps ``element -> target variance`` for the
    other radii (defaults to the variance the 4.5 projection retained, so every
    radius keeps the same information budget).
    """
    if fixed_components is None:
        fixed_components = DEFAULT_FIXED_COMPONENTS
    if pca_variance is None:
        pca_variance = ORIGINAL_PCA_VARIANCE
    summ_frames, atom_frames = [], []
    for r in radii:
        meta = stream_soap_radius(aimel_df, r, cache_dir)
        n_comp_map = fixed_components.get(r, {})
        for el in elements:
            proj = project_pca_radius(el, meta, r, cache_dir,
                                      pca_variance=pca_variance[el],
                                      n_components=n_comp_map.get(el))
            atoms, summ = leakage_at_radius(
                proj, aimel_df, holdout_pos, r,
                target_labels=target_labels, knn=knn, whiten=whiten,
                leak_pct=leak_pct)
            if len(summ):
                summ_frames.append(summ)
            if len(atoms):
                atom_frames.append(atoms)
    summary = pd.concat(summ_frames, ignore_index=True)
    atom_novelty = (pd.concat(atom_frames, ignore_index=True)
                    if atom_frames else pd.DataFrame())
    return summary, atom_novelty


# ---------------------------------------------------------------------------
# 5. accuracy tie-in: scores on leaked vs novel held-out atoms, per radius
# ---------------------------------------------------------------------------

def stratified_ccc_by_novelty(
    exp_dir: str,
    models: List[str],
    atom_novelty: pd.DataFrame,
    r_cut: float,
    test_df: pd.DataFrame,
    properties: List[str] = PROPERTIES,
    n_folds: int = 25,
    min_atoms: int = 30,
) -> pd.DataFrame:
    """Per-model held-out scores split into ``leaked`` vs ``novel`` at ``r_cut``.

    The held-out OOD atoms are split by their ``leaked`` flag at this radius, then
    scores are recomputed from the cached ``test_preds.pkl`` (no retraining).
    Predictions are matched to the novelty table by the stable ``(file,
    atom_idx)`` key -- ``test_preds.pkl`` has no ``file`` column, but its index
    matches ``test_df``, from which ``file`` is recovered. Leaked vs novel is
    compared WITHIN each label (at 8 Bohr "leaked" is almost all N/O, so a pooled
    comparison would be confounded by element composition). Fold scores are
    averaged within repeats (repeat = fold // 5); the summary reports mean and
    half-width of a t-based 95% CI over the 5 repeat means.
    """
    from scipy import stats as _st

    sub = atom_novelty[atom_novelty["r_cut"] == r_cut].copy()
    sub["stratum"] = np.where(sub["leaked"], "leaked", "novel")
    status = sub[["file", "atom_idx", "stratum", "label4"]]
    file_of = test_df["file"]   # index (original mol id) -> file

    recs = []
    matched_reported = False
    for model in models:
        for fold in range(n_folds):
            path = os.path.join(exp_dir, model, f"fold_{fold}", "test_preds.pkl")
            if not os.path.exists(path):
                continue
            dfp = pd.read_pickle(path)
            atom = df_to_atom_table(dfp)
            # df_to_atom_table concatenates dfp['atom'] in row order; attach the
            # same-ordered (file, atom_idx). file comes from test_df's shared index.
            files_mol = file_of.loc[dfp.index].to_numpy()
            atom["file"] = np.concatenate(
                [np.full(len(a), f) for f, a in zip(files_mol, dfp["atom"])])
            atom["atom_idx"] = np.concatenate(
                [np.arange(len(a)) for a in dfp["atom"]])
            # inner-join -> keeps ONLY the OOD query atoms, tagged leaked/novel.
            merged = atom.merge(status, on=["file", "atom_idx"], how="inner")
            if not matched_reported:
                print(f"[ccc@{r_cut}] matched {len(merged):,}/{len(status):,} query "
                      f"atoms in {model} fold {fold} "
                      f"(of {len(atom):,} test atoms)", flush=True)
                matched_reported = True
            for label4, lg in merged.groupby("label4"):
                # 'all' = the reported OOD set for this label; 'leaked'/'novel'
                # are its two strata. CCC(all) vs CCC(novel) is the inflation check.
                groups = {"all": lg}
                groups.update({s: g for s, g in lg.groupby("stratum")})
                for stratum, gdf in groups.items():
                    for prop in properties:
                        if len(gdf) < min_atoms:
                            continue
                        t = gdf[f"target_{prop}"].to_numpy()
                        p = gdf[f"pred_{prop}"].to_numpy()
                        ss_tot = float(np.sum((t - t.mean()) ** 2))
                        recs.append({
                            "model": model, "fold": fold, "repeat": fold // 5,
                            "label4": label4, "stratum": stratum, "property": prop,
                            "ccc": concordance_correlation_coefficient(t, p),
                            "r2": (1.0 - float(np.sum((t - p) ** 2)) / ss_tot
                                   if ss_tot > 0 else np.nan),
                            "spearman": float(_st.spearmanr(t, p)[0]),
                            "mae": float(np.mean(np.abs(p - t))),
                            "n_atoms": len(gdf)})
        print(f"[ccc@{r_cut}] {model} done", flush=True)

    raw = pd.DataFrame(recs)
    if raw.empty:
        return raw
    # repeat-level mean of each metric, then t-CI over the 5 repeats. CCC & R^2
    # reward target spread; Spearman (rank) and MAE (raw error) do not and are the
    # variance-robust reads. All four are computed on the SAME atom subsets.
    metrics = ["ccc", "r2", "spearman", "mae"]
    keys = ["model", "label4", "stratum", "property"]
    rep = raw.groupby(keys + ["repeat"])[metrics].mean().reset_index()
    t975 = _st.t.ppf(0.975, 4)

    def _summ(g):
        out = {}
        for metric in metrics:
            m = g[metric].mean()
            sem = g[metric].std(ddof=1) / np.sqrt(len(g))
            out[f"{metric}_mean"] = m
            out[f"{metric}_ci95"] = t975 * sem
        out["n_repeats"] = len(g)
        return pd.Series(out)

    return rep.groupby(keys).apply(_summ, include_groups=False).reset_index()


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

# Per-held-out-label line colours for the novelty curve.
HOLDOUT_LABEL_COLORS: Dict[str, str] = {
    "H_10": "#37B221",
    "C_11": "#4A5568",
    "N_13": "#047DD9",
    "O_10": "#FF6363",
}


def plot_leakage_curve(
    summary: pd.DataFrame,
    target_labels: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    label_fontsize: float = 14.0,
    tick_fontsize: float = 14.0,
    legend_fontsize: float = 11.0,
):
    """Median novelty percentile vs radius, one line per held-out label
    (C_11, H_10, N_13, O_10). The percentile drops toward 50 (a typical training
    atom) as the holdout leaks. Font sizes mirror ``plot_min_degree_cutoff_pdf``.
    """
    import matplotlib.pyplot as plt

    if target_labels is None:
        target_labels = sorted(HOLDOUT_TARGET_LABELS)
    if colors is None:
        colors = HOLDOUT_LABEL_COLORS

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for lab in target_labels:
        g = summary[summary["label"] == lab].sort_values("r_cut")
        if len(g):
            ax.plot(g["r_cut"], g["median_novelty_pct"], marker="o", lw=2,
                    color=colors.get(lab), label=lab)

    ax.set_xlabel("SOAP cutoff / perception radius (Bohr)", fontsize=label_fontsize)
    ax.set_ylabel("Median novelty percentile", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(title="held-out label", fontsize=legend_fontsize,
              title_fontsize=legend_fontsize, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        base = os.path.splitext(save_path)[0]
        for ext in (".svg", ".pdf"):  # vector copies alongside the raster
            if not save_path.endswith(ext):
                fig.savefig(base + ext, bbox_inches="tight")
    return fig, ax
