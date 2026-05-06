"""
result_analysis.py
==================
Analysis utilities for the 5×5 cross-validation results.

Public API
----------
Metric functions (moved from notebook cell 2):
  concordance_correlation_coefficient  – Lin's CCC
  cvar_mae                             – CVaR of absolute errors
  stratified_metric                    – bulk / tail split (fixed-q or IQR-adaptive)
  _compute_metrics_dict                – all 12 metrics for one (target, pred) pair

Vectorised data loading:
  df_to_atom_table    – molecule-level DataFrame → atom-level table (fast)
  precompute_metrics  – full orchestration returning elem_metrics & cluster_metrics

ETB figure-of-merit:
  compute_etb_table   – Equivalence-to-Best scores per (model, element, property)
  build_score_table   – aggregate into (element × model) score table

Visualisation:
  plot_etb_heatmap    – 4×6 heatmap of ETB scores
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import scipy.sparse as sp
from scipy.stats import spearmanr, t as t_dist, pearsonr, levene, shapiro
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import NamedTuple


class TukeyResult(NamedTuple):
    """Output of one (panel, property) Tukey HSD computation.

    pc        : k × k DataFrame of pairwise adjusted p-values.
    means     : Series of per-method means (over the repeat-level units fed
                to the RM-ANOVA, after any avg_over_* collapsing).
    msd_half  : Tukey minimum significant difference on the MEAN scale.
                |mean_i − mean_j| > msd_half  ⇔  pc[i, j] < α (used for α).
    df_err    : residual degrees of freedom from the RM-ANOVA.
    """
    pc: pd.DataFrame
    means: pd.Series
    msd_half: float
    df_err: int

# ── Physical constants ────────────────────────────────────────────────────────

# Correct Frobenius weights for a traceless symmetric rank-2 tensor stored as
# [Q_XY, Q_XZ, Q_YZ, (Q_XX-Q_YY)/2, Q_ZZ].
# Derivation: ||T||²_F = Q_xx²+Q_yy²+Q_zz²+2(Q_xy²+Q_xz²+Q_yz²) with
# tracelessness Q_xx+Q_yy+Q_zz=0 and Q_aniso=(Q_xx-Q_yy)/2.
# Result: 2·xy² + 2·xz² + 2·yz² + 2·aniso² + 1.5·zz²
FROBENIUS_WEIGHTS = np.array([2.0, 2.0, 2.0, 2.0, 1.5], dtype=np.float64)

# Base property names (stored in prediction files)
ATOM_TARGETS = ['N', 'LI', 'Mu_X', 'Mu_Y', 'Mu_Z',
                'Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ']
# Derived norm properties computed from components
DERIVED_PROPS = ['|Mu|', '|Q|']
ALL_PROPS = ATOM_TARGETS + DERIVED_PROPS

# ── Defaults ──────────────────────────────────────────────────────────────────

_HIGHER_IS_BETTER = {
    'CCC', 'Spearman', 'R2',
}

# Three complementary metrics for the ETB summary table.
#
# CCC     – overall agreement including mean bias.  For positive-valued
#           properties (N, LI, |Mu|, |Q|), the non-zero mean makes CCC
#           sensitive to systematic over/under-prediction that Pearson r
#           misses.
#
# Spearman – pure rank correlation.  Captures whether the model correctly
#            orders atoms from lowest to highest property value across the
#            full distribution — the "broader range" criterion.

DEFAULT_ETB_METRICS    = ['CCC', 'Spearman']
DEFAULT_SUMMARY_PROPS  = ['N', 'LI', '|Mu|', '|Q|']
DEFAULT_ETB_SPLIT      = 'clusters'


# ══════════════════════════════════════════════════════════════════════════════
# Metric functions
# ══════════════════════════════════════════════════════════════════════════════

def concordance_correlation_coefficient(y_true, y_pred) -> float:
    """Lin's Concordance Correlation Coefficient (CCC).

    Measures agreement on the identity line, combining Pearson r with a
    bias-correction factor.  Returns a value in [-1, 1]; 1 = perfect
    agreement on the 45° line.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom < 1e-12:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    return float(2.0 * cov / denom)



def _compute_metrics_dict(
    t,
    p,
) -> dict:
    """Compute all 12 standard metrics for one (target, pred) pair.

    Parameters
    ----------
    t, p : array-like  target and prediction arrays
    sq : float  stratification quantile (default 0.95)
    adaptive_iqr : bool  pass through to stratified_metric

    Returns
    -------
    dict with keys: MAE, RMSE, R2, CCC, Spearman, CVaR_MAE,
                    R2_bulk, R2_tail, CCC_bulk, CCC_tail, MAE_bulk, MAE_tail
    """
    rho, _ = spearmanr(t, p)
    return {
        'MAE':       mean_absolute_error(t, p),
        'RMSE':      float(np.sqrt(mean_squared_error(t, p))),
        'R2':        r2_score(t, p),
        'CCC':       concordance_correlation_coefficient(t, p),
        'Spearman':  rho
    }


# ══════════════════════════════════════════════════════════════════════════════
# Repeated-measures Tukey HSD (pairwise p-values vs. best)
# ══════════════════════════════════════════════════════════════════════════════

def _rm_anova_mse(
    avg_df: pd.DataFrame, dv: str, within: str, subject: str,
) -> tuple[float, int, int, int, dict, list]:
    """RM-ANOVA residual MSE / df via the classical decomposition.

    SS_err = SS_total − SS_subject − SS_group;   df_err = (n−1)(k−1).
    Returns (mse, df_err, n, k, group_means_dict, group_order).
    """
    pivot = (
        avg_df.pivot_table(index=subject, columns=within, values=dv).dropna()
    )
    n, k = pivot.shape
    cols = list(pivot.columns)
    if n < 2 or k < 2:
        return np.nan, 0, n, k, {c: np.nan for c in cols}, cols
    grand     = pivot.values.mean()
    subj_mean = pivot.mean(axis=1).values
    grp_mean  = pivot.mean(axis=0).values
    ss_subj   = k * np.sum((subj_mean - grand) ** 2)
    ss_grp    = n * np.sum((grp_mean  - grand) ** 2)
    ss_tot    = np.sum((pivot.values - grand) ** 2)
    ss_err    = max(ss_tot - ss_subj - ss_grp, 0.0)
    df_err    = (n - 1) * (k - 1)
    mse       = ss_err / df_err if df_err > 0 else np.nan
    return mse, df_err, n, k, dict(zip(cols, grp_mean)), cols


def rm_tukey_pairwise(
    avg_df: pd.DataFrame,
    metric: str,
    group_col: str = 'method',
    subject: str = 'repeat',
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.Series, float, int]:
    """Pairwise Tukey HSD p-values using the RM-ANOVA error term.

    Mirrors notebooks/model_comparison.py::rm_tukey_hsd: SE from RM-ANOVA
    MSE, studentized-range p-values and critical value.  Unlike a
    between-subjects Tukey, this accounts for the paired structure across
    models within each subject (here, repeat).

    Parameters
    ----------
    avg_df : long-form DataFrame with one row per (subject, group) already
             averaged over any nuisance dimension (typically inner folds
             collapsed into 5 repeat-level means per model).
    metric : numeric column.
    group_col : usually 'method'.
    subject  : usually 'repeat'.
    alpha    : FWER for the returned minimum significant difference.

    Returns
    -------
    pc        : (k × k) DataFrame of adjusted p-values; diagonal = 1.
    means     : per-group means (Series).
    msd_half  : Tukey minimum significant difference on the MEAN scale:
                q(α, k, df_err) · sqrt(MSE / n).  |mean_i − mean_j| > msd_half
                ⇔ p < α.  Useful for drawing an "equivalence region" around
                the best model's mean.
    df_err    : residual degrees of freedom.
    """
    from statsmodels.stats.libqsturng import psturng, qsturng

    mse, df_err, n, k, grp_mean, cols = _rm_anova_mse(
        avg_df, dv=metric, within=group_col, subject=subject,
    )
    means = pd.Series(grp_mean, dtype=float)
    pc = pd.DataFrame(index=cols, columns=cols, data=1.0, dtype=float)
    if not np.isfinite(mse) or mse <= 0 or k < 2 or n < 2:
        return pc, means, np.nan, df_err

    se_diff = np.sqrt(2.0 * mse / n)
    for i, m1 in enumerate(cols):
        for j, m2 in enumerate(cols):
            if i < j:
                q_stat = abs(means[m1] - means[m2]) / se_diff
                p = psturng(q_stat * np.sqrt(2.0), k, df_err)
                p = float(p[0]) if isinstance(p, np.ndarray) else float(p)
                pc.loc[m1, m2] = pc.loc[m2, m1] = p

    q_crit   = qsturng(1.0 - alpha, k, df_err)
    msd_half = float(q_crit * np.sqrt(mse / n))
    return pc, means, msd_half, df_err


# ══════════════════════════════════════════════════════════════════════════════
# Vectorised data loading
# ══════════════════════════════════════════════════════════════════════════════

def df_to_atom_table(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Expand a molecule-level prediction DataFrame to an atom-level table.

    Uses vectorised np.concatenate instead of row-by-row iteration.
    For a fold with 5840 molecules this is ~12× faster than the original
    per-molecule Python loop (one concatenation per column vs 5840 appends).

    Columns returned
    ----------------
    element, cluster,
    pred_{prop} / target_{prop}  for each prop in ATOM_TARGETS,
    pred_|Mu|, target_|Mu|,
    pred_|Q|,  target_|Q|
    """
    data: dict[str, np.ndarray] = {}
    data['element'] = np.concatenate(df_pred['atom'].tolist())
    data['cluster'] = np.concatenate(df_pred['atom_cluster_labels'].tolist())

    for prop in ATOM_TARGETS:
        for prefix in ('pred_', 'target_'):
            col = prefix + prop
            data[col] = np.concatenate(
                df_pred[col].apply(np.asarray).tolist()
            )

    # Dipole norm  ||μ|| = sqrt(Mu_X² + Mu_Y² + Mu_Z²)
    for pfx in ('pred', 'target'):
        x, y, z = (data[f'{pfx}_Mu_{c}'] for c in ('X', 'Y', 'Z'))
        data[f'{pfx}_|Mu|'] = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Quadrupole Frobenius norm (mathematically correct for traceless tensor)
    for pfx in ('pred', 'target'):
        tensor = np.stack(
            [data[f'{pfx}_Q_{c}'] for c in ('XY', 'XZ', 'YZ', 'aniso', 'ZZ')],
            axis=-1,
        )
        data[f'{pfx}_|Q|'] = np.sqrt(
            np.sum(FROBENIUS_WEIGHTS * tensor ** 2, axis=-1) + 1e-8
        )

    return pd.DataFrame(data)


def precompute_metrics(
    experiment_dir: str,
    models: list[str],
    elements: list[str],
    cluster_labels: list[str],
    prop_names: list[str],
    n_inner_folds: int = 5,
    adaptive_iqr: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute elem_metrics and cluster_metrics DataFrames from raw predictions.

    Optimised over the original notebook loop:
      - The molecule-level DataFrame is loaded once per fold (not once per property).
      - df_to_atom_table vectorises all concatenation steps.
      - A single pass over the atom table computes metrics for all properties.

    Parameters
    ----------
    experiment_dir : str
        Root directory containing {model}/fold_{k}/{split}_preds.pkl files.
    models : list[str]
    elements : list[str]   e.g. ['H', 'C', 'N', 'O']
    cluster_labels : list[str]  e.g. ['H_10', 'C_11', 'N_13', 'O_10']
    prop_names : list[str]   e.g. ALL_PROPS
    n_inner_folds : int   number of inner folds (used to compute repeat index)
    stratify_quantile : float   bulk/tail split percentile (default 0.95)
    adaptive_iqr : bool   use IQR-based adaptive percentile (see stratified_metric)

    Returns
    -------
    elem_metrics : DataFrame  shape (n_models × n_folds × 2 splits × n_elems × n_props, 18)
    cluster_metrics : DataFrame  shape (n_models × n_folds × n_clusters × n_props, 17)
    """
    def _get_folds(model: str) -> list[int]:
        model_dir = os.path.join(experiment_dir, model)
        if not os.path.isdir(model_dir):
            return []
        folds = []
        for name in os.listdir(model_dir):
            if name.startswith('fold_'):
                try:
                    folds.append(int(name.split('_', 1)[1]))
                except ValueError:
                    pass
        return sorted(folds)

    def _safe_read(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            warnings.warn(f'Missing: {path}')
            return pd.DataFrame()
        try:
            return pd.read_pickle(path)
        except Exception as exc:
            warnings.warn(f'Cannot read {path}: {exc}')
            return pd.DataFrame()

    elem_records: list[dict] = []
    cluster_records: list[dict] = []

    for model in models:
        folds = _get_folds(model)
        if not folds:
            warnings.warn(f'No folds for {model}; skipping.')
            continue

        for fold in folds:
            repeat = fold // n_inner_folds

            # ── Element metrics (val + test) ──────────────────────────────────
            for split in ('val', 'test'):
                path = os.path.join(
                    experiment_dir, model, f'fold_{fold}', f'{split}_preds.pkl'
                )
                df_pred = _safe_read(path)
                if df_pred.empty:
                    continue

                atom_df = df_to_atom_table(df_pred)
                elem_grps = atom_df.groupby('element', sort=False)

                for elem in elements:
                    if elem not in elem_grps.groups:
                        continue
                    grp = elem_grps.get_group(elem)

                    for prop in prop_names:
                        p_col, t_col = f'pred_{prop}', f'target_{prop}'
                        if p_col not in grp.columns:
                            continue
                        p_arr = grp[p_col].values
                        t_arr = grp[t_col].values

                        rec = {
                            'method': model, 'cv_cycle': fold,
                            'repeat': repeat, 'split': split,
                            'element': elem, 'property': prop,
                        }
                        rec.update(
                            _compute_metrics_dict(
                                t_arr, p_arr,
                            )
                        )
                        elem_records.append(rec)

            # ── Cluster metrics (test only) ───────────────────────────────────
            path = os.path.join(
                experiment_dir, model, f'fold_{fold}', 'test_preds.pkl'
            )
            df_pred = _safe_read(path)
            if df_pred.empty:
                continue

            atom_df = df_to_atom_table(df_pred)
            cl_grps = atom_df.groupby('cluster', sort=False)

            for cl in cluster_labels:
                if cl not in cl_grps.groups:
                    continue
                grp = cl_grps.get_group(cl)

                for prop in prop_names:
                    p_col, t_col = f'pred_{prop}', f'target_{prop}'
                    if p_col not in grp.columns:
                        continue
                    p_arr = grp[p_col].values
                    t_arr = grp[t_col].values

                    rec = {
                        'method': model, 'cv_cycle': fold,
                        'repeat': repeat, 'cluster': cl, 'property': prop,
                    }
                    rec.update(
                        _compute_metrics_dict(
                            t_arr, p_arr,
                        )
                    )
                    cluster_records.append(rec)

    elem_metrics    = pd.DataFrame(elem_records)
    cluster_metrics = pd.DataFrame(cluster_records)
    return elem_metrics, cluster_metrics


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX table
# ══════════════════════════════════════════════════════════════════════════════

_METRIC_LABELS = {
    'CCC':      r'CCC',
    'Spearman': r'Spearman $\rho$',
    'R2':       r'$R^2$',
    'MAE':      r'MAE',
    'RMSE':     r'RMSE',
}


def _repeat_mean_stats(
    df: pd.DataFrame,
    models: list[str],
    metric: str,
    panel_col: str,
    panels: list[str],
    properties: list[str],
    avg_over_panels: bool,
    avg_over_props: bool,
    split: str | None = None,
    ci_level: float = 0.95,
) -> dict[tuple, tuple[float, float]]:
    """Return stats[(pk, prk, model)] = (mean, ci_half) over 5 repeat-means.

    Inner-fold values are first averaged within each repeat, then summarised
    as mean ± t_{(1+ci_level)/2, 4} × SEM.  Whichever of {panels, properties}
    is flagged ``avg_over_*=True`` gets collapsed into the key 'avg' and is
    averaged over before the SEM is computed.
    """
    sub = df[
        df[panel_col].isin(panels) &
        df['property'].isin(properties) &
        df['method'].isin(models)
    ]
    if split is not None:
        sub = sub[sub['split'] == split]

    group_keys = ['method', 'repeat']
    if not avg_over_panels:
        group_keys = [panel_col] + group_keys
    if not avg_over_props:
        group_keys = ['property'] + group_keys
    metric_sub  = sub[list(dict.fromkeys(group_keys + [metric]))].dropna(subset=[metric])
    repeat_avgs = metric_sub.groupby(group_keys)[metric].mean().reset_index()

    panel_keys = ['avg'] if avg_over_panels else panels
    prop_keys  = ['avg'] if avg_over_props  else properties
    stats: dict[tuple, tuple[float, float]] = {}
    for pk in panel_keys:
        for prk in prop_keys:
            rows = repeat_avgs
            if not avg_over_panels:
                rows = rows[rows[panel_col] == pk]
            if not avg_over_props:
                rows = rows[rows['property'] == prk]
            for model in models:
                vals = rows[rows['method'] == model][metric].values
                n = len(vals)
                if n == 0:
                    stats[(pk, prk, model)] = (np.nan, np.nan)
                    continue
                mean_v = float(vals.mean())
                sem_v  = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                t_crit = t_dist.ppf((1 + ci_level) / 2, max(n - 1, 1)) if n > 1 else 0.0
                stats[(pk, prk, model)] = (mean_v, t_crit * sem_v)
    return stats


def metric_summary_table(
    df: pd.DataFrame,
    models: list[str],
    metric: str = 'CCC',
    properties: list[str] | None = None,
    panels: list[str] | None = None,
    panel_col: str = 'cluster',
    split: str | None = None,
    ci_level: float = 0.95,
    n_decimals: int = 3,
    print_table: bool = True,
) -> pd.DataFrame:
    """Per-(panel, property, model) mean ± CI with rank — for hand-built tables.

    Columns of the returned DataFrame:
        panel, property, model, mean, ci_half, rank
    where ``rank`` is 1, 2, 3, ... within each (panel, property), with the
    direction (max vs min) chosen automatically from ``_HIGHER_IS_BETTER``.

    If ``print_table=True`` (default), prints a plain-text grouped table with
    🥇/🥈/🥉 replaced by '*'/'**'/'***' for easy copy-paste.
    """
    properties = properties or DEFAULT_SUMMARY_PROPS
    if panels is None:
        panels = sorted(df[panel_col].dropna().unique().tolist())
    higher = metric in _HIGHER_IS_BETTER

    stats = _repeat_mean_stats(
        df, models=models, metric=metric, panel_col=panel_col,
        panels=panels, properties=properties,
        avg_over_panels=False, avg_over_props=False,
        split=split, ci_level=ci_level,
    )

    rows: list[dict] = []
    rank_marks = ['***', '**', '*']
    for panel in panels:
        for prop in properties:
            cell_means = {
                m: stats.get((panel, prop, m), (np.nan,))[0] for m in models
            }
            valid = {m: v for m, v in cell_means.items() if not np.isnan(v)}
            ordered = sorted(valid, key=lambda m: valid[m], reverse=higher)
            rank_of = {m: i + 1 for i, m in enumerate(ordered)}
            for m in models:
                mean_v, ci_h = stats.get((panel, prop, m), (np.nan, np.nan))
                rows.append({
                    'panel':     panel,
                    'property':  prop,
                    'model':     m,
                    'mean':      mean_v,
                    'ci_half':   ci_h,
                    'rank':      rank_of.get(m, np.nan),
                })

    out = pd.DataFrame(rows)

    if print_table:
        fmt = f'{{:.{n_decimals}f}}'
        header = f'{"panel":<8}{"property":<8}{"model":<10}{"mean":>10}{"±CI":>10}  rank'
        print(header)
        print('-' * len(header))
        prev_panel = prev_prop = None
        for _, r in out.iterrows():
            panel_s = r['panel'] if r['panel'] != prev_panel else ''
            prop_s  = r['property'] if (r['panel'], r['property']) != (prev_panel, prev_prop) else ''
            mark = ''
            if not pd.isna(r['rank']) and 1 <= r['rank'] <= 3:
                mark = rank_marks[int(r['rank']) - 1]
            mean_s = '---' if pd.isna(r['mean']) else fmt.format(r['mean'])
            ci_s   = '---' if pd.isna(r['ci_half']) else fmt.format(r['ci_half'])
            print(f'{panel_s:<8}{prop_s:<8}{r["model"]:<10}{mean_s:>10}{ci_s:>10}  {mark}')
            prev_panel, prev_prop = r['panel'], r['property']
        print()
        print(f'rank marks: *** = best, ** = 2nd, * = 3rd  (higher is better: {higher})')

    return out


def tukey_pvalue_matrix(
    df: pd.DataFrame,
    models: list[str],
    metric: str = 'CCC',
    properties: list[str] | None = None,
    panels: list[str] | None = None,
    panel_col: str = 'cluster',
    split: str | None = None,
    avg_over_panels: bool = False,
    avg_over_props: bool = False,
    alpha: float = 0.05,
    print_tables: bool = True,
    n_decimals: int = 4,
) -> dict[tuple, TukeyResult]:
    """Full Tukey pairwise-p matrices, optionally after averaging over panels/properties.

    The repeat-level unit fed to RM-ANOVA is always a per-(method, repeat) mean.
    The averaging flags collapse dimensions *before* that mean is computed, so
    a higher-dimensional average reduces variance and increases power at the cost
    of hiding per-stratum interactions.

    Parameters
    ----------
    avg_over_panels : if True, average the metric over all ``panels`` within
        each (method, repeat) before running Tukey.  The result dict then has
        keys ``('avg', prop)`` for each property.
    avg_over_props  : if True, average the metric over all ``properties`` within
        each (method, repeat) before running Tukey.  Keys become ``(panel, 'avg')``.
    When both are True a single matrix is returned under key ``('avg', 'avg')``.

    Returns
    -------
    dict whose keys are (panel_key, prop_key) tuples:
        * ``(panel, prop)``   — both flags False (one matrix per stratum)
        * ``('avg', prop)``   — avg_over_panels=True
        * ``(panel, 'avg')``  — avg_over_props=True
        * ``('avg', 'avg')``  — both True
    Each value is a k×k DataFrame of Tukey-adjusted p-values.
    """
    properties = properties or DEFAULT_SUMMARY_PROPS
    if panels is None:
        panels = sorted(df[panel_col].dropna().unique().tolist())

    sub = df[
        df[panel_col].isin(panels) &
        df['property'].isin(properties) &
        df['method'].isin(models)
    ].copy()
    if split is not None:
        sub = sub[sub['split'] == split]

    # Build the (method, repeat) → scalar repeat-level means for each stratum.
    # Averaging a dimension means we group without it, so the metric is
    # automatically averaged over that axis before the per-repeat mean is taken.
    panel_keys = ['avg'] if avg_over_panels else panels
    prop_keys  = ['avg'] if avg_over_props  else properties

    group_keys = ['method', 'repeat']
    if not avg_over_panels:
        group_keys = [panel_col] + group_keys
    if not avg_over_props:
        group_keys = ['property'] + group_keys

    repeat_avgs = (
        sub[list(dict.fromkeys(group_keys + [metric]))]
        .dropna(subset=[metric])
        .groupby(group_keys)[metric]
        .mean()
        .reset_index()
    )

    out: dict[tuple, TukeyResult] = {}
    for pk in panel_keys:
        for prk in prop_keys:
            rows = repeat_avgs
            if not avg_over_panels:
                rows = rows[rows[panel_col] == pk]
            if not avg_over_props:
                rows = rows[rows['property'] == prk]
            rows = rows.dropna(subset=[metric])
            if rows['method'].nunique() < 2:
                continue
            try:
                pc, means, msd, df_err = rm_tukey_pairwise(
                    rows, metric, alpha=alpha,
                )
            except Exception as e:
                warnings.warn(f'Tukey failed for ({pk}, {prk}): {e}')
                continue
            out[(pk, prk)] = TukeyResult(pc=pc, means=means, msd_half=msd, df_err=df_err)
            if print_tables:
                avg_note = ''
                if avg_over_panels and avg_over_props:
                    avg_note = ' [avg over panels + props]'
                elif avg_over_panels:
                    avg_note = f' [avg over panels | prop={prk}]'
                elif avg_over_props:
                    avg_note = f' [avg over props | panel={pk}]'
                else:
                    avg_note = f' [panel={pk} | prop={prk}]'
                print(f'=== Tukey p-values{avg_note} ===')
                print(f'    means:  {dict(means.round(n_decimals))}')
                print(f'    MSD:    {msd:.{n_decimals}f}   df_err={df_err}')
                print(pc.round(n_decimals).to_string())
                print()
    return out


def latex_metric_table(
    df: pd.DataFrame,
    models: list[str],
    metric: str = 'CCC',
    properties: list[str] | None = None,
    panels: list[str] | None = None,
    panel_col: str = 'cluster',
    split: str | None = None,
    avg_over_panels: bool = False,
    avg_over_props: bool = False,
    orient: str = 'tall',
    rank_colors: tuple[str, str, str] = (
        r'\cellcolor{gold!50}',
        r'\cellcolor{gray!25}',
        r'\cellcolor{orange!30}',
    ),
    n_decimals: int = 3,
    ci_level: float = 0.95,
    alpha: float = 0.05,
    caption: str | None = None,
    label: str = 'tab:metrics',
    landscape: bool = False,
    print_preamble: bool = True,
) -> str:
    """LaTeX table of mean ± CI for ONE metric, with best-3 colouring.

    Layout
    ------
    * orient='tall' — models in COLUMNS, (panel × property) in ROWS.
        - Both flags False → 16 rows (4 panels × 4 properties), with the
          panel column rendered as a multirow of 4 over its properties.
        - One flag True   → 4 rows (the dimension that is *not* averaged).
        - Both flags True → 1 row (the global summary).
        Ranking colours are assigned across model columns within each row.

    * orient='wide' — models in ROWS, (panel × property) in COLUMNS.
        - Both flags False → 4 multicolumn panel headers each spanning 4
          property sub-columns (16 data columns).
        - One flag True   → 4 columns.
        - Both flags True → 1 column.
        Ranking colours are assigned across model rows within each column.

    The metric name (and its label) goes in the caption, not in the header.

    Parameters
    ----------
    df : DataFrame.
    models : list of model names (used to define both order and ranking ties).
    metric : single metric column name (default 'CCC').
    properties, panels : dimension lists; default to DEFAULT_SUMMARY_PROPS and
                         the unique values of df[panel_col].
    panel_col : 'cluster' or 'element'.
    split : pre-filter on split if not None.
    avg_over_panels, avg_over_props : collapse the corresponding axis.
    orient : 'tall' or 'wide'.
    rank_colors : LaTeX \\cellcolor commands for 1st, 2nd, 3rd.
    n_decimals, ci_level, caption, label, landscape, print_preamble : as before.

    Returns
    -------
    str — complete LaTeX table (also printed).
    """
    if orient not in ('tall', 'wide'):
        raise ValueError(f"orient must be 'tall' or 'wide', got {orient!r}")

    properties = properties or DEFAULT_SUMMARY_PROPS
    if panels is None:
        panels = sorted(df[panel_col].dropna().unique().tolist())

    stats = _repeat_mean_stats(
        df, models=models, metric=metric, panel_col=panel_col,
        panels=panels, properties=properties,
        avg_over_panels=avg_over_panels, avg_over_props=avg_over_props,
        split=split, ci_level=ci_level,
    )

    # Tukey results per stratum, used to colour 2nd / 3rd by statistical
    # equivalence rather than raw mean rank.  The MSD here is recomputed on
    # whatever averaging level the table is showing, so it shrinks
    # automatically as more dimensions are collapsed (averaging reduces
    # within-model variance → narrower equivalence band).
    tukey_results = tukey_pvalue_matrix(
        df, models=models, metric=metric, properties=properties, panels=panels,
        panel_col=panel_col, split=split,
        avg_over_panels=avg_over_panels, avg_over_props=avg_over_props,
        alpha=alpha, print_tables=False,
    )

    panel_keys = ['avg'] if avg_over_panels else panels
    prop_keys  = ['avg'] if avg_over_props  else properties
    higher = metric in _HIGHER_IS_BETTER
    panel_label = 'Cluster' if panel_col == 'cluster' else 'Element'
    metric_label = _METRIC_LABELS.get(metric, metric)

    fmt = f'{{:.{n_decimals}f}}'
    col_sep = ' & '
    row_end = r' \\' + '\n'

    def _escape(s: str) -> str:
        return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    def _val(pk, prk, model) -> str:
        mean_v, ci_h = stats.get((pk, prk, model), (np.nan, np.nan))
        if np.isnan(mean_v):
            return '---'
        return f'{fmt.format(mean_v)}\\,$\\pm$\\,{fmt.format(ci_h)}'

    def _rank_map(item_keys):
        """Rank model entries by Tukey-equivalence to the best.

        Gold goes to the model with the highest mean (or lowest, if
        ``higher_is_better`` is False).  Silver goes to the next-highest
        mean *only if* it is not significantly worse than gold
        (Tukey p ≥ alpha).  Bronze goes to the next-highest mean *only if*
        it is not significantly worse than gold or silver.  Models that
        are significantly worse than every coloured one are left blank.

        If a Tukey result isn't available for the stratum (e.g. only one
        model has data), the function falls back to mean-only top-3.
        """
        means  = {k: stats.get(k, (np.nan,))[0] for k in item_keys}
        valid  = {k: v for k, v in means.items() if not np.isnan(v)}
        if not valid:
            return {}
        ranked = sorted(valid, key=lambda k: valid[k], reverse=higher)

        # Resolve the stratum key (all keys in item_keys share the same pk, prk).
        pk0, prk0, _ = ranked[0]
        res = tukey_results.get((pk0, prk0))

        out: dict = {ranked[0]: rank_colors[0]}
        if res is None or res.pc is None or res.pc.empty:
            for i, k in enumerate(ranked[1:1 + len(rank_colors) - 1], start=1):
                out[k] = rank_colors[i]
            return out

        pc = res.pc
        for k in ranked[1:]:
            if len(out) >= len(rank_colors):
                break
            m_k = k[2]
            if m_k not in pc.index:
                continue
            tied = False
            for prev_key in out:
                m_prev = prev_key[2]
                if m_prev in pc.index:
                    p_val = float(pc.loc[m_k, m_prev])
                    if np.isfinite(p_val) and p_val >= alpha:
                        tied = True
                        break
            if tied:
                out[k] = rank_colors[len(out)]
            else:
                # Not equivalent to any coloured model → stop here; subsequent
                # ranked models are even further from the top.
                break
        return out

    def _coloured(value: str, colour: str) -> str:
        if value == '---' or not colour:
            return value
        return f'{colour}{value}'

    body_lines: list[str] = []

    if orient == 'tall':
        # Rows = (panel × property); Cols = models.
        prefix_labels: list[str] = []
        if not avg_over_panels:
            prefix_labels.append(panel_label)
        if not avg_over_props:
            prefix_labels.append('Property')
        n_prefix = len(prefix_labels)
        col_spec = 'l' * n_prefix + 'c' * len(models)
        header_str = (col_sep.join(
            prefix_labels + [f'\\textbf{{{_escape(m)}}}' for m in models]
        ) + row_end)

        n_prk = len(prop_keys)
        for p_idx, pk in enumerate(panel_keys):
            for pr_idx, prk in enumerate(prop_keys):
                rank_map = _rank_map([(pk, prk, m) for m in models])
                cells = [
                    _coloured(_val(pk, prk, m), rank_map.get((pk, prk, m), ''))
                    for m in models
                ]
                prefix_cells: list[str] = []
                if not avg_over_panels:
                    prefix_cells.append(
                        f'\\multirow{{{n_prk}}}{{*}}{{{_escape(pk)}}}'
                        if pr_idx == 0 else ''
                    )
                if not avg_over_props:
                    prefix_cells.append(_escape(prk))
                body_lines.append(col_sep.join(prefix_cells + cells) + row_end)
            if not avg_over_panels and p_idx < len(panel_keys) - 1:
                body_lines.append('\\addlinespace[2pt]\n')

    else:  # orient == 'wide'
        # Cols = (panel × property); Rows = models.
        n_panels_g = len(panel_keys)
        n_props_g  = len(prop_keys)
        n_data     = n_panels_g * n_props_g
        col_spec   = 'l' + 'c' * n_data

        header_parts: list[str] = []
        if not avg_over_panels and not avg_over_props:
            # Two-row header: panel multicolumns then property sub-headers.
            top_cells = [
                f'\\multicolumn{{{n_props_g}}}{{c}}{{{_escape(pk)}}}'
                for pk in panel_keys
            ]
            header_parts.append(
                '\\multicolumn{1}{l}{}' + col_sep
                + col_sep.join(top_cells) + row_end
            )
            cmid = ''
            for i in range(n_panels_g):
                start = 2 + i * n_props_g
                end   = 1 + (i + 1) * n_props_g
                cmid += f'\\cmidrule(lr){{{start}-{end}}}'
            header_parts.append(cmid + '\n')
            sub_cells = [_escape(prk) for _ in panel_keys for prk in prop_keys]
            header_parts.append(
                'Model' + col_sep + col_sep.join(sub_cells) + row_end
            )
        else:
            # Single-row header.
            if avg_over_props and not avg_over_panels:
                col_lbls = [_escape(pk) for pk in panel_keys]
            elif avg_over_panels and not avg_over_props:
                col_lbls = [_escape(prk) for prk in prop_keys]
            else:  # both averaged
                col_lbls = [metric_label]
            header_parts.append(
                'Model' + col_sep + col_sep.join(col_lbls) + row_end
            )
        header_str = ''.join(header_parts)

        # Per-column ranking across model rows.
        col_keys = [(pk, prk) for pk in panel_keys for prk in prop_keys]
        col_rank_maps = {
            ck: _rank_map([(*ck, m) for m in models]) for ck in col_keys
        }

        for model in models:
            cells = []
            for pk, prk in col_keys:
                cells.append(_coloured(
                    _val(pk, prk, model),
                    col_rank_maps[(pk, prk)].get((pk, prk, model), ''),
                ))
            body_lines.append(
                f'\\textbf{{{_escape(model)}}}' + col_sep
                + col_sep.join(cells) + row_end
            )

    # ── Caption ───────────────────────────────────────────────────────────────
    if caption is None:
        scope_bits = []
        if avg_over_panels:
            scope_bits.append(f'{panel_label.lower()} labels averaged')
        if avg_over_props:
            scope_bits.append('properties averaged')
        scope_str = (', ' + '; '.join(scope_bits)) if scope_bits else ''
        caption = (
            f'Model comparison on {metric_label} '
            f'(mean\\,$\\pm$\\,{int(ci_level*100)}\\,\\%\\,CI over $n=5$ CV repeats'
            f'{scope_str}). '
            f'Cell colour: \\colorbox{{gold!50}}{{1st}}\\,'
            f'\\colorbox{{gray!25}}{{2nd}}\\,'
            f'\\colorbox{{orange!30}}{{3rd}} '
            f'{"per row" if orient == "tall" else "per column"}.'
        )

    # ── Assemble ──────────────────────────────────────────────────────────────
    parts: list[str] = []
    if print_preamble:
        parts.append(
            '% Required packages:\n'
            '%   \\usepackage{booktabs}       % toprule/midrule/bottomrule/cmidrule\n'
            '%   \\usepackage{multirow}       % \\multirow\n'
            '%   \\usepackage[table]{xcolor}  % \\cellcolor\n'
            '%   \\definecolor{gold}{rgb}{1.0, 0.84, 0.0}\n'
            + ('%   \\usepackage{rotating}         % sidewaystable (landscape)\n'
               if landscape else '')
        )

    def _ind(s: str) -> str:
        return '    ' + s.replace('\n', '\n    ').rstrip('    ')

    tbl_env = 'sidewaystable' if landscape else 'table'
    parts += [
        f'\n\\begin{{{tbl_env}}}\n',
        '  \\caption{' + caption + '}\n',
        '  \\label{' + label + '}\n',
        '  \\centering\n',
        f'  \\begin{{tabular}}{{{col_spec}}}\n',
        '    \\toprule\n',
        _ind(header_str),
        '    \\midrule\n',
        *[_ind(ln) for ln in body_lines],
        '    \\bottomrule\n',
        '  \\end{tabular}\n',
        f'\\end{{{tbl_env}}}\n',
    ]
    table_str = ''.join(parts)
    print(table_str)
    return table_str


# ══════════════════════════════════════════════════════════════════════════════
# ANOVA validity diagnostics for the fixed-test-set 5×5 CV design
# ══════════════════════════════════════════════════════════════════════════════
#
# Context
# -------
# The experiment uses a 5×5 repeated k-fold CV on a train+val pool, evaluated
# on a FIXED held-out test set whose molecules contain at least one atom from
# cluster labels {O_10, N_13, C_11, H_10}.  Every one of the 25 fold-level
# test metrics therefore comes from the SAME atoms — the variation is purely
# due to different training splits, not different test data.
#
# Three validity questions are answered here:
#
# Q1 — Within-repeat fold correlation (ICC)
#   Are the 5 inner folds of the same repeat correlated?  Because they share
#   the same outer test set, positive ICC is expected.  The existing analysis
#   already averages within-repeat before computing CIs (df=4); this confirms
#   that step is necessary and quantifies the information loss.
#
# Q2 — Val ↔ test correlation
#   Does fold-level val performance predict fold-level test performance?
#   If yes (high Pearson r), val-based model selection generalises to the test
#   set even though the test environments were never seen during training or
#   validation.  Low r would mean the val metric is useless as a proxy.
#
# Q3 — ANOVA assumptions on repeat-averaged means
#   With n=5 repeat-means per model, does the data satisfy normality
#   (Shapiro-Wilk) and homoscedasticity (Levene) — prerequisites for the
#   one-way ANOVA / Tukey HSD used in the rest of the notebook?
#
# Q4 — Unseen label fraction in val
#   For each fold, what fraction of the unique cluster labels in the val set
#   were NOT present in the training portion of that fold?  A higher fraction
#   means the val set better mimics the test set's out-of-distribution
#   character, making the val metric a better generalization proxy.
# ══════════════════════════════════════════════════════════════════════════════


def _compute_icc_oneway(values: np.ndarray, k: int) -> float:
    """ICC(1,1) for a (n_subjects × k) array via one-way ANOVA decomposition.

    ICC(1,1) = (MS_B − MS_W) / (MS_B + (k−1)·MS_W)

    n_subjects = repeats, k = inner folds per repeat.
    A value near 1 means all variance is between-repeat (inner folds are
    nearly identical) → the 5 fold metrics within a repeat carry only ~1
    independent observation.
    """
    if values.ndim == 1:
        values = values.reshape(-1, k)
    n = values.shape[0]
    grand_mean = values.mean()
    subject_means = values.mean(axis=1)
    ms_b = k * np.sum((subject_means - grand_mean) ** 2) / max(n - 1, 1)
    ms_w = np.sum((values - subject_means[:, None]) ** 2) / max(n * (k - 1), 1)
    denom = ms_b + (k - 1) * ms_w
    if denom < 1e-12:
        return 1.0 if ms_w < 1e-12 else 0.0
    return float(np.clip((ms_b - ms_w) / denom, -1.0, 1.0))


def assess_fold_correlation(
    df: pd.DataFrame,
    models: list[str],
    panel_col: str,
    panels: list[str],
    properties: list[str],
    metric: str,
    n_inner_folds: int = 5,
    split: str | None = 'test',
) -> pd.DataFrame:
    """ICC(1,1) and effective sample size for each (model, panel, property).

    Reshape the 25 raw metric values into a (n_repeats × n_inner_folds) matrix
    and compute ICC so we know how many effectively independent observations
    remain after the within-repeat averaging.

    Parameters
    ----------
    split : None for cluster_metrics (no 'split' column); 'test' for elem_metrics.

    Returns
    -------
    DataFrame: model, panel, property, ICC, n_eff, n_raw.
    n_eff = n_raw / (1 + (n_inner_folds−1) × max(0, ICC))
    """
    sub = df if split is None else df[df['split'] == split]
    records: list[dict] = []
    for model in models:
        m_df = sub[sub['method'] == model]
        for panel in panels:
            p_df = m_df[m_df[panel_col] == panel]
            for prop in properties:
                pp = (
                    p_df[p_df['property'] == prop][['repeat', 'cv_cycle', metric]]
                    .dropna(subset=[metric])
                    .copy()
                )
                if pp.empty:
                    continue
                pp['fold_in_repeat'] = pp['cv_cycle'] % n_inner_folds
                pivot = pp.pivot_table(
                    index='repeat', columns='fold_in_repeat', values=metric
                )
                vals = pivot.dropna().values  # drop repeats with missing folds
                n_raw = vals.size
                if n_raw < 2:
                    continue
                k_act = vals.shape[1]
                icc = _compute_icc_oneway(vals, k_act)
                n_eff = n_raw / max(1.0, 1.0 + (k_act - 1) * max(0.0, icc))
                records.append({
                    'model': model, 'panel': panel, 'property': prop,
                    'ICC': icc, 'n_eff': round(n_eff, 2), 'n_raw': n_raw,
                })
    return pd.DataFrame(records)


def assess_val_test_correlation(
    elem_metrics: pd.DataFrame,
    models: list[str],
    panels: list[str],
    properties: list[str],
    metric: str,
    panel_col: str = 'element',
) -> pd.DataFrame:
    """Pearson r between val and test metric across all 25 cv_cycles per stratum.

    High r (close to 1) means that folds where the model performs better on
    the within-CV val set also perform better on the fixed held-out test set.
    This is the key check that val-based model selection transfers to the test
    distribution even though the test cluster environments are out-of-domain.

    Returns
    -------
    DataFrame: model, panel, property, r_val_test, p_val_test,
               mean_val, mean_test, n_cycles.
    """
    records: list[dict] = []
    for model in models:
        m_df = elem_metrics[elem_metrics['method'] == model]
        for panel in panels:
            p_df = m_df[m_df[panel_col] == panel]
            for prop in properties:
                pp = p_df[p_df['property'] == prop]
                val_rows  = pp[pp['split'] == 'val' ][['cv_cycle', metric]].dropna()
                test_rows = pp[pp['split'] == 'test'][['cv_cycle', metric]].dropna()
                merged = val_rows.merge(
                    test_rows, on='cv_cycle', suffixes=('_val', '_test')
                )
                if len(merged) < 3:
                    continue
                r, p = pearsonr(merged[f'{metric}_val'], merged[f'{metric}_test'])
                records.append({
                    'model': model, 'panel': panel, 'property': prop,
                    'r_val_test': float(r), 'p_val_test': float(p),
                    'mean_val':   float(merged[f'{metric}_val'].mean()),
                    'mean_test':  float(merged[f'{metric}_test'].mean()),
                    'n_cycles':   len(merged),
                })
    return pd.DataFrame(records)


def check_anova_assumptions(
    df: pd.DataFrame,
    models: list[str],
    panel_col: str,
    panels: list[str],
    properties: list[str],
    metric: str,
    n_inner_folds: int = 5,
    split: str | None = 'test',
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Normality (Shapiro-Wilk) and homoscedasticity (Levene) on repeat means.

    The existing ETB/CI pipeline averages the 5 inner folds within each repeat
    to obtain n=5 repeat-level means per model.  One-way ANOVA and Tukey HSD
    require those means to be (approximately) normal and the per-model variances
    to be homogeneous.  This function tests both assumptions for every
    (panel × property) stratum.

    Returns
    -------
    DataFrame: panel, property, model, p_shapiro, p_levene,
               shapiro_ok, levene_ok, n_repeats.
    p_levene is the same for all models in a stratum (it is a global test).
    """
    sub = df if split is None else df[df['split'] == split]
    records: list[dict] = []
    for panel in panels:
        p_df = sub[sub[panel_col] == panel]
        for prop in properties:
            pp = p_df[p_df['property'] == prop]
            repeat_means: dict[str, np.ndarray] = {}
            for model in models:
                vals = (
                    pp[pp['method'] == model]
                    .groupby('repeat')[metric]
                    .mean()
                    .dropna()
                    .values
                )
                if len(vals) >= 3:
                    repeat_means[model] = vals

            valid_groups = list(repeat_means.values())
            p_lev = np.nan
            if len(valid_groups) >= 2:
                try:
                    _, p_lev = levene(*valid_groups)
                except Exception:
                    pass

            for model in models:
                vals = repeat_means.get(model)
                if vals is None:
                    continue
                p_shap = np.nan
                try:
                    _, p_shap = shapiro(vals)
                except Exception:
                    pass
                records.append({
                    'panel': panel, 'property': prop, 'model': model,
                    'p_shapiro': float(p_shap) if not np.isnan(p_shap) else np.nan,
                    'p_levene':  float(p_lev)  if not np.isnan(p_lev)  else np.nan,
                    'shapiro_ok': bool(p_shap > alpha) if np.isfinite(p_shap) else None,
                    'levene_ok':  bool(p_lev  > alpha) if np.isfinite(p_lev)  else None,
                    'n_repeats':  int(len(vals)),
                })
    return pd.DataFrame(records)


def compute_val_unseen_fraction(
    experiment_dir: str,
    models: list[str],
    n_inner_folds: int = 5,
    first_model_only: bool = True,
) -> pd.DataFrame:
    """Fraction of val cluster labels not present in the training fold.

    For each inner fold k of repeat r, the training data is (approximately)
    the union of the other 4 val sets within the same repeat.  We load the
    val_preds.pkl for all folds in a repeat, reconstruct the approximate
    training label set as the union of the remaining folds, and compute the
    fraction of val-fold-k labels that are absent from that union.

    A higher unseen fraction means more novel atomic environments appear in
    the val set, making it a better proxy for the held-out test environments.

    Parameters
    ----------
    first_model_only : bool
        If True, run only for the first model in the list — the split
        structure is identical across models, so this avoids redundant I/O.

    Returns
    -------
    DataFrame: model, repeat, fold, cv_cycle,
               n_val_unique_labels, n_unseen_labels,
               unseen_label_fraction, unseen_labels (sorted list).
    """
    use_models = [models[0]] if first_model_only and models else models
    records: list[dict] = []
    for model in use_models:
        model_dir = os.path.join(experiment_dir, model)
        if not os.path.isdir(model_dir):
            warnings.warn(f'Missing model dir: {model_dir}')
            continue
        folds_by_repeat: dict[int, list[int]] = {}
        for name in os.listdir(model_dir):
            if name.startswith('fold_'):
                try:
                    fold = int(name.split('_', 1)[1])
                    repeat = fold // n_inner_folds
                    folds_by_repeat.setdefault(repeat, []).append(fold)
                except ValueError:
                    pass
        for repeat, folds in sorted(folds_by_repeat.items()):
            fold_labels: dict[int, set] = {}
            for fold in sorted(folds):
                path = os.path.join(model_dir, f'fold_{fold}', 'val_preds.pkl')
                if not os.path.exists(path):
                    continue
                try:
                    df_pred = pd.read_pickle(path)
                    all_labels = np.concatenate(
                        df_pred['atom_cluster_labels'].tolist()
                    )
                    fold_labels[fold] = set(all_labels.tolist())
                except Exception as exc:
                    warnings.warn(f'Cannot read {path}: {exc}')
            for fold, val_lbs in fold_labels.items():
                train_lbs = set().union(
                    *(lbs for f, lbs in fold_labels.items() if f != fold)
                )
                unseen = val_lbs - train_lbs
                records.append({
                    'model': model, 'repeat': repeat, 'fold': fold,
                    'cv_cycle': fold,
                    'n_val_unique_labels': len(val_lbs),
                    'n_unseen_labels':     len(unseen),
                    'unseen_label_fraction': len(unseen) / max(1, len(val_lbs)),
                    'unseen_labels': sorted(unseen),
                })
    return pd.DataFrame(records)


def cv_anova_diagnostics(
    elem_metrics: pd.DataFrame,
    cluster_metrics: pd.DataFrame,
    models: list[str],
    elements: list[str],
    cluster_labels: list[str],
    properties: list[str],
    metric: str = 'CCC',
    n_inner_folds: int = 5,
    alpha: float = 0.05,
    experiment_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Full ANOVA validity diagnostic suite for the 5×5 CV design.

    Runs all four diagnostic questions (ICC, val-test r, ANOVA assumptions,
    unseen label fraction) and returns results as a dict of DataFrames.

    Parameters
    ----------
    elem_metrics, cluster_metrics : outputs of precompute_metrics.
    models : model names to include.
    elements : element symbols for elem_metrics analysis.
    cluster_labels : cluster label strings for cluster_metrics analysis.
    properties : property names.
    metric : which metric to diagnose (default 'CCC').
    n_inner_folds : inner folds per repeat (default 5).
    alpha : significance level for Shapiro/Levene pass/fail flags.
    experiment_dir : if provided, also compute unseen label fractions from
                     val_preds.pkl files.

    Returns
    -------
    dict with keys:
      'icc_elem'      – ICC + n_eff per (model, element, property), test split.
      'icc_cluster'   – ICC + n_eff per (model, cluster, property).
      'val_test_corr' – Val–test Pearson r per (model, element, property).
      'anova_elem'    – Shapiro + Levene per (element, property, model).
      'anova_cluster' – Shapiro + Levene per (cluster, property, model).
      'unseen_labels' – Val unseen-label fraction per fold (needs experiment_dir).
    """
    out: dict[str, pd.DataFrame] = {}
    out['icc_elem'] = assess_fold_correlation(
        elem_metrics, models=models, panel_col='element', panels=elements,
        properties=properties, metric=metric, n_inner_folds=n_inner_folds,
        split='test',
    )
    out['icc_cluster'] = assess_fold_correlation(
        cluster_metrics, models=models, panel_col='cluster', panels=cluster_labels,
        properties=properties, metric=metric, n_inner_folds=n_inner_folds,
        split=None,
    )
    out['val_test_corr'] = assess_val_test_correlation(
        elem_metrics, models=models, panels=elements,
        properties=properties, metric=metric, panel_col='element',
    )
    out['anova_elem'] = check_anova_assumptions(
        elem_metrics, models=models, panel_col='element', panels=elements,
        properties=properties, metric=metric, n_inner_folds=n_inner_folds,
        split='test', alpha=alpha,
    )
    out['anova_cluster'] = check_anova_assumptions(
        cluster_metrics, models=models, panel_col='cluster', panels=cluster_labels,
        properties=properties, metric=metric, n_inner_folds=n_inner_folds,
        split=None, alpha=alpha,
    )
    if experiment_dir is not None:
        out['unseen_labels'] = compute_val_unseen_fraction(
            experiment_dir, models=models, n_inner_folds=n_inner_folds,
        )
    return out


def print_anova_summary(diag: dict[str, pd.DataFrame], metric: str = 'CCC') -> None:
    """Print a concise plain-text summary of the four diagnostic questions."""
    sep = '─' * 64

    # Q1: ICC
    icc = diag.get('icc_elem', pd.DataFrame())
    if not icc.empty:
        print(sep)
        print(f'Q1 — Within-repeat fold correlation (ICC)  [metric={metric}]')
        print(f'     Positive ICC means inner folds within a repeat are correlated.')
        print(f'     The existing analysis is correct to average within repeat first.')
        summary = icc.groupby('model')[['ICC', 'n_eff']].mean().round(3)
        print(summary.to_string())
        median_neff = icc['n_eff'].median()
        print(f'  → Median n_eff across all strata: {median_neff:.1f}  '
              f'(out of {icc["n_raw"].median():.0f} raw folds)')

    # Q2: Val-test correlation
    vt = diag.get('val_test_corr', pd.DataFrame())
    if not vt.empty:
        print(sep)
        print(f'Q2 — Val ↔ test Pearson r  [metric={metric}]')
        print(f'     High r means val performance predicts test performance.')
        summary = vt.groupby('model')['r_val_test'].agg(['mean', 'median', 'min']).round(3)
        summary.columns = ['mean_r', 'median_r', 'min_r']
        print(summary.to_string())
        frac_pos = (vt['r_val_test'] > 0).mean() * 100
        frac_sig  = (vt['p_val_test'] < 0.05).mean() * 100
        print(f'  → {frac_pos:.0f}% of strata have r > 0;  '
              f'{frac_sig:.0f}% are significant (p < 0.05)')

    # Q3: ANOVA assumptions
    for key, label in [('anova_elem', 'element'), ('anova_cluster', 'cluster')]:
        anova = diag.get(key, pd.DataFrame())
        if anova.empty:
            continue
        print(sep)
        print(f'Q3 — ANOVA assumptions ({label}-level)  [metric={metric}, α=0.05]')
        shap_pass = anova['shapiro_ok'].dropna().mean() * 100
        lev_pass  = (
            anova.drop_duplicates(['panel', 'property'])['levene_ok']
            .dropna().mean() * 100
        )
        print(f'     Shapiro-Wilk (normality)  pass rate: {shap_pass:.0f}%  '
              f'(per model × stratum)')
        print(f'     Levene (homoscedasticity)  pass rate: {lev_pass:.0f}%  '
              f'(per stratum, across models)')
        if shap_pass < 80:
            print('  ⚠ Low normality pass rate — consider non-parametric tests '
                  'or bootstrap CIs.')
        if lev_pass < 80:
            print('  ⚠ Low homoscedasticity pass rate — Welch ANOVA or '
                  'Games-Howell post-hoc may be more appropriate.')

    # Q4: Unseen labels
    ul = diag.get('unseen_labels', pd.DataFrame())
    if not ul.empty:
        print(sep)
        print('Q4 — Unseen cluster-label fraction in val folds')
        print(f'     Mean unseen fraction: {ul["unseen_label_fraction"].mean():.3f}')
        print(f'     Range: [{ul["unseen_label_fraction"].min():.3f}, '
              f'{ul["unseen_label_fraction"].max():.3f}]')
        # Which folds have the most / fewest unseen labels
        top = ul.nlargest(3, 'unseen_label_fraction')[
            ['repeat', 'fold', 'unseen_label_fraction', 'unseen_labels']
        ]
        print('     Folds with most unseen labels:')
        for _, row in top.iterrows():
            print(f'       repeat={row["repeat"]} fold={row["fold"]}  '
                  f'frac={row["unseen_label_fraction"]:.3f}  '
                  f'labels={row["unseen_labels"]}')
    print(sep)


def plot_cv_diagnostics(
    diag: dict[str, pd.DataFrame],
    metric: str = 'CCC',
    save_path: str | None = 'cv_diagnostics.pdf',
) -> plt.Figure:
    """Four-panel diagnostic plot for the 5×5 CV ANOVA validity assessment.

    Panel A (top-left)  : ICC distribution per model — element-level test split.
    Panel B (top-right) : Effective sample size n_eff per model.
    Panel C (bot-left)  : Val–test Pearson r distribution per model.
    Panel D (bot-right) : Shapiro-Wilk normality pass rate per model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f'5×5 CV Design Validity  |  metric: {metric}\n'
        f'Fixed held-out test set {{O_10, N_13, C_11, H_10}}',
        fontsize=11,
    )

    def _sorted_models(key: str) -> list[str]:
        df = diag.get(key, pd.DataFrame())
        return sorted(df['model'].unique()) if not df.empty else []

    # ── Panel A: ICC ─────────────────────────────────────────────────────────
    ax = axes[0, 0]
    icc_df = diag.get('icc_elem', pd.DataFrame())
    if not icc_df.empty:
        ms = _sorted_models('icc_elem')
        data = [icc_df[icc_df['model'] == m]['ICC'].dropna().values for m in ms]
        bp = ax.boxplot(data, labels=ms, patch_artist=True, notch=False)
        for patch in bp['boxes']:
            patch.set_facecolor('#a8d8ea')
        ax.axhline(0.0, color='k',      ls='--', lw=0.8, label='ICC=0 (independent)')
        ax.axhline(0.5, color='orange', ls='--', lw=0.8, label='ICC=0.5')
    ax.set_title('A — ICC(1,1): within-repeat fold correlation\n'
                 '(element-level, split=test)', fontsize=9)
    ax.set_ylabel('ICC(1,1)')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=7)

    # ── Panel B: n_eff ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    if not icc_df.empty:
        data = [icc_df[icc_df['model'] == m]['n_eff'].dropna().values for m in ms]
        bp = ax.boxplot(data, labels=ms, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#f8d49a')
        ax.axhline(25, color='green', ls='--', lw=0.8, label='n=25 (all independent)')
        ax.axhline(5,  color='red',   ls='--', lw=0.8, label='n=5 (repeat means only)')
    ax.set_title('B — Effective sample size n_eff\n(25 raw → fewer when ICC > 0)',
                 fontsize=9)
    ax.set_ylabel('n_eff')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=7)

    # ── Panel C: Val-test Pearson r ───────────────────────────────────────────
    ax = axes[1, 0]
    vt_df = diag.get('val_test_corr', pd.DataFrame())
    if not vt_df.empty:
        ms_vt = _sorted_models('val_test_corr')
        data = [vt_df[vt_df['model'] == m]['r_val_test'].dropna().values for m in ms_vt]
        bp = ax.boxplot(data, labels=ms_vt, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#b5ead7')
        ax.axhline(0.0, color='k',      ls='--', lw=0.8)
        ax.axhline(0.5, color='orange', ls='--', lw=0.8, label='r=0.5')
    ax.set_title(f'C — Val–test Pearson r ({metric})\nacross 25 cv_cycles per stratum',
                 fontsize=9)
    ax.set_ylabel(f'Pearson r')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=7)

    # ── Panel D: Shapiro-Wilk pass rate ───────────────────────────────────────
    ax = axes[1, 1]
    shap_df = diag.get('anova_elem', pd.DataFrame())
    if not shap_df.empty and 'shapiro_ok' in shap_df.columns:
        pass_rate = (
            shap_df.groupby('model')['shapiro_ok']
            .apply(lambda x: x.dropna().mean() * 100 if x.notna().any() else np.nan)
            .dropna()
        )
        colors = ['#90ee90' if v >= 80 else '#ffb347' for v in pass_rate.values]
        ax.bar(pass_rate.index, pass_rate.values, color=colors, edgecolor='k',
               linewidth=0.6)
        ax.axhline(80, color='green', ls='--', lw=0.8, label='80% threshold')
        ax.set_ylim(0, 110)
    ax.set_title('D — Shapiro-Wilk normality pass rate\n'
                 '(5 repeat means per model×stratum, α=0.05)', fontsize=9)
    ax.set_ylabel('% strata passing')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'Saved → {save_path}')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX ANOVA validity diagnostics table  (single metric, follows latex_metric_table)
# ══════════════════════════════════════════════════════════════════════════════

_ORANGE_BG = r'\cellcolor{orange!25}'
_RED_BG    = r'\cellcolor{red!20}'

# Display order of the four diagnostic quantities and their column headers.
_DIAG_NAMES = ('ICC', 'n_eff', 'SW', 'Lev')
_DIAG_HEADERS = {
    'ICC':   r'ICC$(1,1)$',
    'n_eff': r'$n_\text{eff}$',
    'SW':    r'SW',
    'Lev':   r'Lev.',
}


def _aggregate_diag_cell(
    icc_df: pd.DataFrame,
    anova_df: pd.DataFrame,
    model: str,
    panels_in_cell: list,
    properties_in_cell: list,
    show_pvalues: bool,
) -> dict[str, tuple[float, float]]:
    """Aggregate per-cell ANOVA diagnostics over the (panel × property) subset.

    Returns ``{diag_name: (mean, sd)}`` for diag_name in ('ICC', 'n_eff', 'SW', 'Lev').

    For ICC and n_eff we average the per-(panel, property) values directly.
    For SW / Lev we either:
      * ``show_pvalues=True``  → mean ± sd of raw ``p_shapiro`` / ``p_levene``;
      * ``show_pvalues=False`` → percent of (panel × property) cells whose
        ``shapiro_ok`` / ``levene_ok`` flag is True (sd reported as NaN since
        a fraction does not have a meaningful within-cell sd).

    The Levene test is intrinsically a *global* test across models for a given
    (panel, property), so its p-value is the same across all models filtered
    on the same (panel, property) cell — averaging over the cells in scope
    therefore produces the per-(panels × properties) Levene summary, which is
    correct.
    """
    result: dict[str, tuple[float, float]] = {}

    if not icc_df.empty:
        sub = icc_df[
            icc_df['model'].eq(model) &
            icc_df['panel'].isin(panels_in_cell) &
            icc_df['property'].isin(properties_in_cell)
        ]
        icc_vals  = sub['ICC'].dropna().values    if 'ICC'   in sub.columns else np.array([])
        neff_vals = sub['n_eff'].dropna().values  if 'n_eff' in sub.columns else np.array([])
    else:
        icc_vals = neff_vals = np.array([])

    result['ICC'] = (
        (float(icc_vals.mean()),
         float(icc_vals.std(ddof=0)) if len(icc_vals) > 1 else np.nan)
        if len(icc_vals) else (np.nan, np.nan)
    )
    result['n_eff'] = (
        (float(neff_vals.mean()),
         float(neff_vals.std(ddof=0)) if len(neff_vals) > 1 else np.nan)
        if len(neff_vals) else (np.nan, np.nan)
    )

    if not anova_df.empty:
        sub2 = anova_df[
            anova_df['model'].eq(model) &
            anova_df['panel'].isin(panels_in_cell) &
            anova_df['property'].isin(properties_in_cell)
        ]
        sw_p   = sub2['p_shapiro'].dropna().values
        sw_ok  = sub2['shapiro_ok'].dropna().values.astype(float)
        lev_p  = sub2['p_levene'].dropna().values
        lev_ok = sub2['levene_ok'].dropna().values.astype(float)
    else:
        sw_p = sw_ok = lev_p = lev_ok = np.array([])

    if show_pvalues:
        result['SW'] = (
            (float(sw_p.mean()),
             float(sw_p.std(ddof=0)) if len(sw_p) > 1 else np.nan)
            if len(sw_p) else (np.nan, np.nan)
        )
        result['Lev'] = (
            (float(lev_p.mean()),
             float(lev_p.std(ddof=0)) if len(lev_p) > 1 else np.nan)
            if len(lev_p) else (np.nan, np.nan)
        )
    else:
        result['SW']  = ((float(sw_ok.mean())  * 100, np.nan)
                         if len(sw_ok)  else (np.nan, np.nan))
        result['Lev'] = ((float(lev_ok.mean()) * 100, np.nan)
                         if len(lev_ok) else (np.nan, np.nan))
    return result


def latex_anova_diagnostics_table(
    diag: dict,
    models: list[str],
    metric: str = 'CCC',
    panels: list[str] | None = None,
    properties: list[str] | None = None,
    panel_type: str = 'cluster',
    avg_over_panels: bool = False,
    avg_over_props: bool = False,
    orient: str = 'tall',
    show_pvalues: bool = False,
    include_intervals: bool = True,
    icc_color_thresholds: tuple = (0.20, 0.40),
    neff_nominal: float = 25.0,
    neff_color_fracs: tuple = (0.80, 0.60),
    passrate_color_thresholds: tuple = (80.0, 60.0),
    pval_color_thresholds: tuple = (0.10, 0.05),
    n_decimals: int = 3,
    caption: str | None = None,
    label: str = 'tab:anova_diag',
    landscape: bool = False,
    print_preamble: bool = True,
) -> str:
    """LaTeX ANOVA validity diagnostics for ONE metric, with three layout modes.

    Three layout modes
    ------------------
    **Mode A — both flags True (panels and properties averaged):**
        * orient='tall'  → rows = models, cols = {ICC, n_eff, SW, Lev}.
        * orient='wide'  → transpose: rows = diagnostics, cols = models.

    **Mode B — exactly one flag True:**
        * orient='tall'  → rows = the non-averaged dimension (4 entries);
                           cols = ``n_models`` blocks of 4 diagnostic sub-cols
                           (one multicolumn header per model).
        * orient='wide'  → transpose: rows = n_models × 4 diagnostics
                           (multirow on Model); cols = the non-averaged dimension.

    **Mode C — both flags False (full grid):**
        Only ``orient='tall'`` and ``landscape=True`` are allowed (the table
        has 16 data columns and only fits sideways).
        * Rows = ``n_models × n_properties`` with a multirow on the model name,
          one row per property within each model block.
        * Cols = ``n_panels × 4 diagnostics`` with a multicolumn header per
          panel, ICC / n_eff / SW / Lev sub-headers under each panel.

    Aggregation (correctness)
    -------------------------
    * ICC, n_eff               — mean ± sd of the per-(panel, property) values
                                 inside the cell.
    * Shapiro-Wilk / Levene    — when ``show_pvalues=False`` (default): pass
                                 rate, i.e. % of (panel × property) cells in
                                 scope where ``shapiro_ok`` / ``levene_ok``
                                 (which are ``p > alpha`` evaluated per cell
                                 in :func:`check_anova_assumptions` with the
                                 same alpha threshold used for shading).
                                 When ``show_pvalues=True``: mean ± sd of raw
                                 p-values across the cells in scope.
    * Levene is a *global* test across models within each (panel, property),
      so when filtered by model it appears once per cell with the same value
      across all models — averaging over the cells in scope correctly
      reproduces the (panels × properties) Levene summary.

    With both flags False each cell aggregates a single (panel, property),
    so pass rates are 0% or 100%; consider ``show_pvalues=True`` in that mode.

    Colour coding (worse = redder)
    ------------------------------
    * |ICC|     ≥ 0.20 / 0.40           → orange / red.
    * n_eff     < 80% / 60% of nominal  → orange / red.
    * pass-rate < 80% / 60%             → orange / red.
    * p-value   < 0.10 / 0.05           → orange / red (failing test).

    Parameters
    ----------
    diag           : dict from :func:`cv_anova_diagnostics`.
    models         : model names.
    metric         : single metric the diagnostics were computed for; goes in
                     caption only.
    panels, properties : dimension lists; default from the diag dict.
    panel_type     : 'cluster' (Sacred Rule) or 'elem'.
    avg_over_panels, avg_over_props : collapse the corresponding axis.
    orient         : 'tall' or 'wide'.
    show_pvalues   : show mean p-values instead of pass rates for SW/Lev.
    include_intervals : show ±sd for ICC / n_eff.
    *_color_thresholds, neff_nominal : shading thresholds (see Colour coding).
    n_decimals, caption, label, landscape, print_preamble : standard.

    Returns
    -------
    str — LaTeX table (also printed).
    """
    if orient not in ('tall', 'wide'):
        raise ValueError(f"orient must be 'tall' or 'wide', got {orient!r}")

    both_avg = avg_over_panels and avg_over_props
    none_avg = (not avg_over_panels) and (not avg_over_props)
    one_avg  = (not both_avg) and (not none_avg)

    if none_avg and not landscape:
        raise ValueError(
            "avg_over_panels=False and avg_over_props=False produces a wide "
            "16-data-column table; pass landscape=True (sidewaystable)."
        )
    if none_avg and orient != 'tall':
        raise ValueError(
            "avg_over_panels=False and avg_over_props=False is only "
            "supported with orient='tall'."
        )

    icc_df   = diag.get(f'icc_{panel_type}',   pd.DataFrame())
    anova_df = diag.get(f'anova_{panel_type}', pd.DataFrame())

    if panels is None:
        panels = sorted(icc_df['panel'].dropna().unique().tolist()) if not icc_df.empty else []
    if properties is None:
        properties = sorted(icc_df['property'].dropna().unique().tolist()) if not icc_df.empty else []

    panel_keys = ['avg'] if avg_over_panels else panels
    prop_keys  = ['avg'] if avg_over_props  else properties

    # Compute per (pk, prk, model) the four diagnostic (mean, sd) tuples.
    stats: dict[tuple, dict[str, tuple[float, float]]] = {}
    for pk in panel_keys:
        ps = panels if pk == 'avg' else [pk]
        for prk in prop_keys:
            rs = properties if prk == 'avg' else [prk]
            for m in models:
                stats[(pk, prk, m)] = _aggregate_diag_cell(
                    icc_df, anova_df, m, ps, rs, show_pvalues=show_pvalues,
                )

    # ── Formatting & colour helpers ───────────────────────────────────────────
    col_sep = ' & '
    row_end = r' \\' + '\n'

    def _escape(s: str) -> str:
        return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    def _fmt(diag_name: str, value: float, sd: float) -> str:
        if pd.isna(value):
            return '---'
        if (diag_name in ('SW', 'Lev')) and not show_pvalues:
            return f'{value:.1f}'
        nd = 1 if diag_name == 'n_eff' else n_decimals
        s = f'{value:.{nd}f}'
        wants_sd = (
            (include_intervals and diag_name in ('ICC', 'n_eff')) or
            (show_pvalues and diag_name in ('SW', 'Lev'))
        )
        if wants_sd and not pd.isna(sd):
            s += rf'\,$\pm$\,{sd:.{nd}f}'
        return s

    def _color(diag_name: str, value: float, cell_str: str) -> str:
        if pd.isna(value) or cell_str == '---':
            return cell_str
        if diag_name == 'ICC':
            v = abs(value)
            ot, rt = icc_color_thresholds
            if v >= rt: return _RED_BG    + cell_str
            if v >= ot: return _ORANGE_BG + cell_str
        elif diag_name == 'n_eff':
            frac = value / neff_nominal
            ot, rt = neff_color_fracs
            if frac < rt: return _RED_BG    + cell_str
            if frac < ot: return _ORANGE_BG + cell_str
        elif diag_name in ('SW', 'Lev'):
            if show_pvalues:
                ot, rt = pval_color_thresholds
                if value < rt: return _RED_BG    + cell_str
                if value < ot: return _ORANGE_BG + cell_str
            else:
                ot, rt = passrate_color_thresholds
                if value < rt: return _RED_BG    + cell_str
                if value < ot: return _ORANGE_BG + cell_str
        return cell_str

    def _cell(pk, prk, model, diag_name) -> str:
        v, sd = stats.get((pk, prk, model), {}).get(diag_name, (np.nan, np.nan))
        return _color(diag_name, v, _fmt(diag_name, v, sd))

    panel_label = 'Cluster' if panel_type == 'cluster' else 'Element'
    n_diag = 4
    diag_headers = [_DIAG_HEADERS[d] for d in _DIAG_NAMES]

    # ── Build body ────────────────────────────────────────────────────────────
    body_lines: list[str] = []

    # ──────────────────── Mode A: both averaged ──────────────────────────────
    if both_avg:
        if orient == 'tall':
            # Rows = models, cols = 4 diagnostics.
            col_spec = 'l' + 'cccc'
            header_str = (
                'Model' + col_sep + col_sep.join(diag_headers) + row_end
            )
            for model in models:
                cells = [_cell('avg', 'avg', model, d) for d in _DIAG_NAMES]
                body_lines.append(
                    f'\\textbf{{{_escape(model)}}}' + col_sep
                    + col_sep.join(cells) + row_end
                )
        else:  # wide
            # Rows = 4 diagnostics, cols = models.
            col_spec = 'l' + 'c' * len(models)
            mod_hdrs = [f'\\textbf{{{_escape(m)}}}' for m in models]
            header_str = (
                'Diagnostic' + col_sep + col_sep.join(mod_hdrs) + row_end
            )
            for d in _DIAG_NAMES:
                cells = [_cell('avg', 'avg', m, d) for m in models]
                body_lines.append(
                    _DIAG_HEADERS[d] + col_sep + col_sep.join(cells) + row_end
                )

    # ──────────────────── Mode B: exactly one averaged ───────────────────────
    elif one_avg:
        non_avg_keys  = panel_keys if not avg_over_panels else prop_keys
        non_avg_label = panel_label if not avg_over_panels else 'Property'

        if orient == 'tall':
            # Rows = non-avg dim (4); cols = n_models × 4 diagnostics.
            col_spec = 'l' + 'cccc' * len(models)
            empty_pre = '\\multicolumn{1}{l}{} & '
            top_cells = [
                f'\\multicolumn{{{n_diag}}}{{c}}{{\\textbf{{{_escape(m)}}}}}'
                for m in models
            ]
            hdr1 = empty_pre + col_sep.join(top_cells) + row_end
            cmid = ''.join(
                f'\\cmidrule(lr){{{2 + i*n_diag}-{1 + (i+1)*n_diag}}}'
                for i in range(len(models))
            )
            hdr1 += cmid + '\n'
            hdr2 = (non_avg_label + col_sep
                    + col_sep.join(diag_headers * len(models)) + row_end)
            header_str = hdr1 + hdr2

            for k in non_avg_keys:
                pk  = k if not avg_over_panels else 'avg'
                prk = k if not avg_over_props  else 'avg'
                cells = [_cell(pk, prk, m, d)
                         for m in models for d in _DIAG_NAMES]
                body_lines.append(_escape(k) + col_sep
                                  + col_sep.join(cells) + row_end)
        else:  # wide
            # Rows = n_models × 4 diagnostics (multirow on Model);
            # cols = non-avg dim (4).
            n_cols   = len(non_avg_keys)
            col_spec = 'll' + 'c' * n_cols
            col_lbls = [_escape(k) for k in non_avg_keys]
            # Top header puts the non-averaged dimension label as a multicolumn
            # over the data columns; sub-header is "Model & Diagnostic & <vals>".
            empty_pre = '\\multicolumn{2}{l}{} & '
            hdr1 = (empty_pre
                    + f'\\multicolumn{{{n_cols}}}{{c}}{{{non_avg_label}}}'
                    + row_end)
            hdr1 += f'\\cmidrule(lr){{3-{2 + n_cols}}}\n'
            hdr2 = ('Model' + col_sep + 'Diagnostic' + col_sep
                    + col_sep.join(col_lbls) + row_end)
            header_str = hdr1 + hdr2
            for m_idx, model in enumerate(models):
                for d_idx, d in enumerate(_DIAG_NAMES):
                    row_cells = []
                    row_cells.append(
                        f'\\multirow{{{n_diag}}}{{*}}{{\\textbf{{{_escape(model)}}}}}'
                        if d_idx == 0 else ''
                    )
                    row_cells.append(_DIAG_HEADERS[d])
                    for k in non_avg_keys:
                        pk  = k if not avg_over_panels else 'avg'
                        prk = k if not avg_over_props  else 'avg'
                        row_cells.append(_cell(pk, prk, model, d))
                    body_lines.append(col_sep.join(row_cells) + row_end)
                if m_idx < len(models) - 1:
                    body_lines.append('\\addlinespace[4pt]\n')

    # ──────────────────── Mode C: neither averaged (orient='tall') ───────────
    else:
        n_panels = len(panels)
        n_props  = len(properties)
        col_spec = 'll' + 'cccc' * n_panels

        # Top header: model+property prefix slots, then panel multicolumns.
        empty_pre = '\\multicolumn{2}{l}{} & '
        top_cells = [
            f'\\multicolumn{{{n_diag}}}{{c}}{{{_escape(pk)}}}'
            for pk in panels
        ]
        hdr1 = empty_pre + col_sep.join(top_cells) + row_end
        cmid = ''.join(
            f'\\cmidrule(lr){{{3 + i*n_diag}-{2 + (i+1)*n_diag}}}'
            for i in range(n_panels)
        )
        hdr1 += cmid + '\n'
        hdr2 = ('Model' + col_sep + 'Property' + col_sep
                + col_sep.join(diag_headers * n_panels) + row_end)
        header_str = hdr1 + hdr2

        for m_idx, model in enumerate(models):
            for pr_idx, prk in enumerate(properties):
                cells = [_cell(pk, prk, model, d)
                         for pk in panels for d in _DIAG_NAMES]
                row = []
                row.append(
                    f'\\multirow{{{n_props}}}{{*}}{{\\textbf{{{_escape(model)}}}}}'
                    if pr_idx == 0 else ''
                )
                row.append(_escape(prk))
                row.extend(cells)
                body_lines.append(col_sep.join(row) + row_end)
            if m_idx < len(models) - 1:
                body_lines.append('\\addlinespace[4pt]\n')

    # ── Caption ───────────────────────────────────────────────────────────────
    if caption is None:
        scope_bits: list[str] = []
        if avg_over_panels:
            scope_bits.append(f'{panel_label.lower()} labels averaged')
        if avg_over_props:
            scope_bits.append('properties averaged')
        scope_str = (', ' + '; '.join(scope_bits)) if scope_bits else ''
        sw_desc = ('mean $p$-value $\\pm$ sd' if show_pvalues else 'pass rate \\%')
        caption = (
            f'ANOVA validity diagnostics on {_METRIC_LABELS.get(metric, metric)} '
            f'for the fixed held-out {panel_type} design{scope_str}. '
            f'\\textbf{{ICC$(1,1)$}}: intraclass correlation across the 5 inner '
            f'folds within each repeat (near 0 means folds carry near-independent '
            f'information); '
            f'\\textbf{{$n_\\text{{eff}}$}}: effective sample size '
            f'(nominal {int(neff_nominal)}); '
            f'\\textbf{{SW}}: Shapiro--Wilk normality on the 5 repeat means '
            f'({sw_desc}); '
            f'\\textbf{{Lev.}}: Levene homoscedasticity across models ({sw_desc}). '
            f'Cell shading: \\colorbox{{orange!25}}{{warning}} / '
            f'\\colorbox{{red!20}}{{fail}} against per-quantity thresholds '
            f'(see docstring).'
        )

    # ── Assemble ──────────────────────────────────────────────────────────────
    parts: list[str] = []
    if print_preamble:
        parts.append(
            '% Required packages: booktabs, multirow, xcolor, colortbl\n'
            + ('% Also: rotating (sidewaystable)\n' if landscape else '')
        )

    def _ind(s: str) -> str:
        return '    ' + s.replace('\n', '\n    ').rstrip(' ')

    tbl_env = 'sidewaystable' if landscape else 'table'
    parts += [
        f'\n\\begin{{{tbl_env}}}\n',
        '  \\caption{' + caption + '}\n',
        '  \\label{' + label + '}\n',
        '  \\centering\n',
        f'  \\begin{{tabular}}{{{col_spec}}}\n',
        '    \\toprule\n',
        _ind(header_str),
        '    \\midrule\n',
        *[_ind(ln) for ln in body_lines],
        '    \\bottomrule\n',
        '  \\end{tabular}\n',
        f'\\end{{{tbl_env}}}\n',
    ]
    table_str = ''.join(parts)
    print(table_str)
    return table_str
