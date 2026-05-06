"""Analysis utilities for the molecular variant × training-fraction CV study.

Reuses notebooks/atomic/result_analysis.py for all statistical functions.
The mol_metrics DataFrame carries a dummy ``panel='global'`` column so that
ra.tukey_pvalue_matrix / ra.metric_summary_table work unchanged.

Typical notebook usage:
    import importlib
    import notebooks.molecular.analysis_molecular as mol
    importlib.reload(mol)
"""
from __future__ import annotations

import os
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator
from scipy.stats import f as f_dist, t as t_dist, ttest_rel, wilcoxon, shapiro

import notebooks.atomic.result_analysis as ra

# ── Constants ─────────────────────────────────────────────────────────────────

MOLECULAR_PROPS: list[str] = ['alpha', 'gap', 'U0', 'Cv']

_PROP_UNITS: dict[str, str] = {
    'alpha': r'$a_0^3$',
    'gap':   'Ha',
    'U0':    'Ha',
    'Cv':    'cal/(mol·K)',
}

_PROP_LABELS: dict[str, str] = {
    'alpha': r'$\alpha$',
    'gap':   r'$\Delta\varepsilon$',
    'U0':    r'$U_0$',
    'Cv':    r'$C_v$',
}

DEFAULT_VARIANTS:  list[str]   = ['informed', 'blind']
DEFAULT_FRACTIONS: list[float] = [0.1, 0.5, 1.0]

DEFAULT_MODELS: list[str] = [
    'informed_1.0', 'informed_0.5', 'informed_0.1',
    'blind_1.0',    'blind_0.5',    'blind_0.1',
]

_HIGHER_IS_BETTER = ra._HIGHER_IS_BETTER  # {'R2', 'CCC', 'Spearman'}

_FONT_SIZES = {
    'title': 11, 'label': 11, 'tick': 10, 'legend': 10, 'suptitle': 12,
}

SIG_WORSE_COLOR = '#d62728'   # red  – significantly worse than best
NOT_SIG_COLOR   = '#999999'   # grey – not significantly worse


# ── Private helpers ───────────────────────────────────────────────────────────

def _safe_read_pickle(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        warnings.warn(f'Missing pickle: {path}')
        return pd.DataFrame()
    try:
        return pd.read_pickle(path)
    except Exception as exc:
        warnings.warn(f'Cannot read {path}: {exc}')
        return pd.DataFrame()


def _get_folds(experiment_dir: str, variant: str) -> list[int]:
    variant_dir = os.path.join(experiment_dir, variant)
    if not os.path.isdir(variant_dir):
        return []
    folds = []
    for name in os.listdir(variant_dir):
        if name.startswith('fold_'):
            try:
                folds.append(int(name.split('_', 1)[1]))
            except ValueError:
                pass
    return sorted(folds)


def _fold_level_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the ``repeat`` column replaced by ``cv_cycle``.

    All ``ra.*`` functions group on ``subject='repeat'`` to build the RM-ANOVA
    pivot.  In the molecular study the 5 inner folds within a repeat have
    *disjoint* test sets (verified: ``fold_0 ∩ fold_1 = ∅``), so the fold-level
    metrics are approximately independent observations — ICC ≈ 0 in
    ``anova_diagnostics_mol``.  Aliasing ``repeat`` to ``cv_cycle`` makes the
    existing ``ra`` machinery pivot on (n=25) folds instead of (n=5) repeat
    means, giving a paired RM-ANOVA design with much higher power.

    The substitution is a no-op for any pandas operation that only uses
    ``repeat`` as a grouping key (which is exactly how every ``ra`` function
    treats it).
    """
    out = df.copy()
    out['repeat'] = out['cv_cycle']
    return out


def _compute_rm_anova(
    df: pd.DataFrame,
    metric: str,
    subject: str = 'cv_cycle',
    within: str = 'method',
) -> dict:
    """Classic RM-ANOVA decomposition.

    Returns a dict with SS, df, MS for the within-subject factor, the subject
    factor, the residual error, and the total — plus F-statistic and p-value
    for the within-subject factor (the model comparison).
    """
    pivot = df.pivot_table(index=subject, columns=within, values=metric).dropna()
    n, k = pivot.shape
    if n < 2 or k < 2:
        return {}

    grand     = float(pivot.values.mean())
    subj_mean = pivot.mean(axis=1).values
    grp_mean  = pivot.mean(axis=0).values

    SS_subj = float(k * np.sum((subj_mean - grand) ** 2))
    SS_grp  = float(n * np.sum((grp_mean  - grand) ** 2))
    SS_tot  = float(np.sum((pivot.values - grand) ** 2))
    SS_err  = max(SS_tot - SS_subj - SS_grp, 0.0)

    df_grp  = k - 1
    df_subj = n - 1
    df_err  = (n - 1) * (k - 1)
    df_tot  = n * k - 1

    MS_grp  = SS_grp  / df_grp  if df_grp  > 0 else np.nan
    MS_subj = SS_subj / df_subj if df_subj > 0 else np.nan
    MS_err  = SS_err  / df_err  if df_err  > 0 else np.nan

    F_grp = MS_grp / MS_err if (MS_err is not np.nan and MS_err > 0) else np.nan
    p_grp = float(f_dist.sf(F_grp, df_grp, df_err)) if np.isfinite(F_grp) else np.nan

    return {
        'n': n, 'k': k,
        'SS_method':  SS_grp,  'df_method':  df_grp,  'MS_method':  MS_grp,
        'F_method':   F_grp,   'p_method':   p_grp,
        'SS_subject': SS_subj, 'df_subject': df_subj, 'MS_subject': MS_subj,
        'SS_error':   SS_err,  'df_error':   df_err,  'MS_error':   MS_err,
        'SS_total':   SS_tot,  'df_total':   df_tot,
    }


def _save_fig(fig, save_path, save_fmt):
    if save_path is not None:
        fig.savefig(save_path, format=save_fmt, bbox_inches='tight', dpi=300)


def _axis_label(base: str, prop: str) -> str:
    unit = _PROP_UNITS.get(prop, '')
    label = _PROP_LABELS.get(prop, prop)
    if unit:
        return f'{base} {label} [{unit}]'
    return f'{base} {label}'


def _add_marginal_density(
    ax, target_values, pred_series, pred_colors,
    target_color='#505050', pred_alpha=0.5,
    max_bins=60, marginal_log=False,
):
    """Top/right marginal histograms for a parity panel (adapted from tmp_plot_cell_updated)."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    ax_top   = divider.append_axes('top',   size='22%', pad=0.00, sharex=ax)
    ax_right = divider.append_axes('right', size='22%', pad=0.00, sharey=ax)

    try:
        edges = np.histogram_bin_edges(target_values, bins='fd')
        if len(edges) - 1 > max_bins:
            edges = np.histogram_bin_edges(target_values, bins=max_bins)
    except Exception:
        edges = np.histogram_bin_edges(target_values, bins=max_bins)

    ax_top.hist(target_values, bins=edges, density=True, color=target_color,
                alpha=1.0, edgecolor='0.15', linewidth=0.35)
    for pred_vals, color in zip(pred_series, pred_colors):
        ax_right.hist(pred_vals, bins=edges, density=True, orientation='horizontal',
                      color=color, alpha=pred_alpha, edgecolor='0.15', linewidth=0.35)

    if marginal_log:
        ax_top.set_yscale('log')
        ax_right.set_xscale('log')
        ax_top.minorticks_off()
        ax_right.minorticks_off()

    ax_top.grid(False);  ax_right.grid(False)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_top.set_yticks([]);   ax_right.set_xticks([])
    for sp in ['right', 'top', 'left', 'bottom']:
        ax_top.spines[sp].set_visible(False)
        ax_right.spines[sp].set_visible(False)


# ── Data loading ──────────────────────────────────────────────────────────────

def precompute_metrics_mol(
    experiment_dir: str,
    experiment_tag: str = '',
    variants: list[str] = DEFAULT_VARIANTS,
    fractions: list[float] = DEFAULT_FRACTIONS,
    properties: list[str] = MOLECULAR_PROPS,
    n_inner_folds: int = 5,
) -> pd.DataFrame:
    """Load fold predictions and compute metrics for all (variant, fraction, fold, property).

    Parameters
    ----------
    experiment_tag : str, optional
        Short label prepended to method names, e.g. ``'cutoff'`` or ``'cp'``.
        When provided, method names become ``'{tag}_{variant}_{frac}'``.
        Defaults to ``''`` for backward-compatible ``'{variant}_{frac}'`` names.

    Returns
    -------
    mol_metrics : DataFrame with columns:
        method, cv_cycle, repeat, split, property, panel (='global'),
        MAE, RMSE, R2, CCC, Spearman

    Notes
    -----
    ``panel='global'`` is a dummy column added so that ra.tukey_pvalue_matrix
    and ra.metric_summary_table can be called with
    ``panels=['global'], panel_col='panel'`` without modification.

    split='both' semantics: when split='both' is passed to analysis functions,
    they pass split=None to ra, which then includes both val and test rows.
    The repeat-level mean averages over 5_folds × 2_splits = 10 values per
    repeat—a conservative pooled estimate.
    """
    records: list[dict] = []

    for variant in variants:
        folds = _get_folds(experiment_dir, variant)
        if not folds:
            warnings.warn(f'No folds found for variant {variant!r}; skipping.')
            continue

        for frac in fractions:
            parts = [p for p in (experiment_tag, variant, str(frac)) if p]
            method = '_'.join(parts)
            frac_dir_name = f'frac_{frac}'

            for fold in folds:
                repeat = fold // n_inner_folds

                for split in ('val', 'test'):
                    path = os.path.join(
                        experiment_dir, variant,
                        f'fold_{fold}', frac_dir_name, f'{split}_preds.pkl',
                    )
                    df = _safe_read_pickle(path)
                    if df.empty:
                        continue

                    for prop in properties:
                        p_col = f'pred_{prop}'
                        t_col = f'target_{prop}'
                        if p_col not in df.columns or t_col not in df.columns:
                            warnings.warn(f'Missing {p_col}/{t_col} in {path}')
                            continue

                        t_arr = df[t_col].to_numpy(dtype=float)
                        p_arr = df[p_col].to_numpy(dtype=float)

                        if np.isnan(t_arr).any() or np.isnan(p_arr).any():
                            warnings.warn(f'NaNs in {path} for {prop}; skipping fold.')
                            continue

                        rec = {
                            'method':   method,
                            'cv_cycle': fold,
                            'repeat':   repeat,
                            'split':    split,
                            'property': prop,
                            'panel':    'global',
                        }
                        rec.update(ra._compute_metrics_dict(t_arr, p_arr))
                        records.append(rec)

    return pd.DataFrame(records)


# ── Tukey HSD forest plot ─────────────────────────────────────────────────────

def tukey_ci_mol(
    metric: str,
    models: list[str],
    mol_metrics: pd.DataFrame,
    split: str = 'test',
    prop_colors: dict[str, Any] | None = None,
    prop_cmap: str = 'winter',
    alpha: float = 0.05,
    save_path: str | None = None,
    save_fmt: str = 'pdf',
) -> None:
    """Forest plot with Tukey HSD significance colouring, one panel per QM property.

    Uses **fold-level** RM-ANOVA (n=25 folds paired across models via
    ``cv_cycle``).  Within each repeat the inner folds have disjoint test
    sets, so each fold metric is an approximately independent observation —
    see :func:`_fold_level_df` for the full justification.

    Parameters
    ----------
    split : 'val', 'test', or 'both'.
        'both' includes val and test rows in the RM-ANOVA; n becomes 50.
    """
    higher = metric in _HIGHER_IS_BETTER
    n_models = len(models)
    n_props  = len(MOLECULAR_PROPS)

    # Property colours
    if prop_colors is not None:
        vline_colors = {p: prop_colors[p] for p in MOLECULAR_PROPS}
    else:
        _cmap = plt.get_cmap(prop_cmap)
        _pal  = [_cmap(x) for x in np.linspace(0.15, 0.85, n_props)]
        vline_colors = dict(zip(MOLECULAR_PROPS, _pal))

    _markers    = ['o', 's', 'D', '^'][:n_props]
    _linestyles = ['-', '--', '-.', ':'][:n_props]
    prop_markers    = dict(zip(MOLECULAR_PROPS, _markers))
    prop_linestyles = dict(zip(MOLECULAR_PROPS, _linestyles))
    offsets = np.linspace(-0.3, 0.3, n_props) if n_props > 1 else [0.0]

    # Pre-compute Tukey results — FOLD-LEVEL via the repeat=cv_cycle hack
    sub: pd.DataFrame = (mol_metrics if split == 'both'
                         else mol_metrics[mol_metrics['split'] == split].copy())
    tukey_results = ra.tukey_pvalue_matrix(
        _fold_level_df(sub),
        models=models,
        metric=metric,
        properties=MOLECULAR_PROPS,
        panels=['global'],
        panel_col='panel',
        split=None,
        avg_over_panels=False,
        avg_over_props=False,
        alpha=alpha,
        print_tables=False,
    )

    # Figure — 2×2 grid, one panel per QM property
    fig_h = max(2.8, 0.42 * n_models + 0.6) * 2 + 0.5
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.0, fig_h), sharey=True, squeeze=False)
    axes = axes_grid.ravel()

    for ax_idx, prop in enumerate(MOLECULAR_PROPS):
        ax = axes[ax_idx]

        # Alternating row shading
        for m_idx in range(n_models):
            if m_idx % 2 == 0:
                ax.axhspan(m_idx - 0.5, m_idx + 0.5, color='#f0f0f0', zorder=0)

        # Fold-level summary (n=25) per method — mean + SEM over all folds.
        prop_sub = sub[sub['property'] == prop].dropna(subset=[metric])
        cycle_stats = prop_sub.groupby('method')[metric]
        means  = cycle_stats.mean()
        sems   = cycle_stats.sem()
        counts = cycle_stats.count()

        # Per-model 95 % CI (display cue only; significance comes from Tukey)
        stats_dict: dict[str, tuple[float, float, float]] = {}
        for model in models:
            if model not in means.index:
                continue
            mean_val = float(means[model])
            n_obs    = int(counts[model])
            sem_val  = float(sems.get(model, np.nan))
            sem_val  = sem_val if np.isfinite(sem_val) else 0.0
            t_crit   = t_dist.ppf(0.975, max(n_obs - 1, 1)) if n_obs > 1 else 0.0
            ci_half  = t_crit * sem_val
            stats_dict[model] = (mean_val, mean_val - ci_half, mean_val + ci_half)

        prop_means = {m: stats_dict[m][0] for m in models if m in stats_dict}
        if not prop_means:
            ax.set_title(f'{prop} (no data)')
            continue

        best = (max(prop_means, key=prop_means.get) if higher
                else min(prop_means, key=prop_means.get))

        res = tukey_results.get(('global', prop))
        tukey_p: dict[str, float] = {}
        msd_half = np.nan
        if res is not None:
            if np.isfinite(res.msd_half):
                msd_half = res.msd_half
            for model in models:
                if model == best or model not in res.pc.index:
                    continue
                tukey_p[model] = float(res.pc.loc[best, model])

        # Draw error bars + markers with significance colouring
        y_pos_list = range(n_models)
        for m_idx, model in enumerate(models):
            if model not in stats_dict:
                continue
            mean_val, ci_lo, ci_hi = stats_dict[model]
            y_pos = m_idx

            if model == best:
                color = vline_colors[prop]
            else:
                p_val = tukey_p.get(model, np.nan)
                color = NOT_SIG_COLOR if (np.isfinite(p_val) and p_val >= alpha) else SIG_WORSE_COLOR

            ax.errorbar(mean_val, y_pos,
                        xerr=[[mean_val - ci_lo], [ci_hi - mean_val]],
                        fmt='none', ecolor=color,
                        capsize=2.2, capthick=0.8, elinewidth=0.8, zorder=3)
            ax.plot(mean_val, y_pos,
                    marker=prop_markers[prop], linestyle='None',
                    markersize=6.0, markerfacecolor=color,
                    markeredgecolor='black', markeredgewidth=0.35, zorder=5)

        # Vertical reference lines: best.mean ± Tukey MSD
        if np.isfinite(msd_half) and best in stats_dict:
            best_mean = stats_dict[best][0]
            for x in (best_mean - msd_half, best_mean + msd_half):
                ax.axvline(x, color=vline_colors[prop],
                           linestyle=prop_linestyles[prop],
                           linewidth=1.8, alpha=0.6, zorder=1)

        ax.set_yticks(range(n_models))
        ax.set_yticklabels(models, fontsize=_FONT_SIZES['label'])
        ax.set_ylim(-0.5, n_models - 0.5)
        prop_label = _PROP_LABELS.get(prop, prop)
        ax.set_title(prop_label, fontsize=_FONT_SIZES['title'], fontweight='bold')
        ax.tick_params(axis='x', labelsize=max(_FONT_SIZES['tick'] - 1, 1), pad=1)
        ax.tick_params(axis='y', pad=1)
        if ax_idx >= 2:  # bottom row
            ax.set_xlabel(metric, fontsize=_FONT_SIZES['label'])
        ax.grid(axis='x', alpha=0.25, linewidth=0.5)

    # Shared legend
    legend_handles = [
        Line2D([0], [0], marker=prop_markers[p], linestyle=prop_linestyles[p],
               color=vline_colors[p], markersize=7,
               markeredgecolor='black', markeredgewidth=0.35,
               linewidth=0.9, label=_PROP_LABELS.get(p, p))
        for p in MOLECULAR_PROPS
    ]
    legend_handles += [
        Line2D([0], [0], marker='o', linestyle='',
               color=NOT_SIG_COLOR, markersize=5,
               markeredgecolor='black', markeredgewidth=0.35, label='non-sig.'),
        Line2D([0], [0], marker='o', linestyle='',
               color=SIG_WORSE_COLOR, markersize=5,
               markeredgecolor='black', markeredgewidth=0.35, label='sig. worse'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               ncol=len(legend_handles), bbox_to_anchor=(0.5, 1.0),
               fontsize=_FONT_SIZES['legend'], frameon=True,
               edgecolor='#cccccc', fancybox=False,
               handletextpad=0.3, columnspacing=0.8, borderpad=0.25)

    split_label = split.upper()
    fig.suptitle(f'{metric} — {split_label}', fontsize=_FONT_SIZES['suptitle'], y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=0.5, w_pad=0.4)
    _save_fig(fig, save_path, save_fmt)
    plt.show()


# ── Learning curve ────────────────────────────────────────────────────────────

def learning_curve_mol(
    metric: str,
    models: list[str],
    mol_metrics: pd.DataFrame,
    split: str = 'test',
    model_prefixes: list[str] | None = None,
    variants: list[str] = DEFAULT_VARIANTS,
    fractions: list[float] = DEFAULT_FRACTIONS,
    model_colors: dict[str, Any] | None = None,
    avg_over_props: bool = False,
    save_path: str | None = None,
    save_fmt: str = 'pdf',
) -> None:
    """Learning curve: metric vs training fraction, one line per model prefix.

    Parameters
    ----------
    model_prefixes : list of strings, optional
        Each entry is the prefix of model names before the fraction, e.g.
        ``['cutoff_informed', 'cutoff_blind', 'cp_informed', 'cp_blind']``.
        Model name for fraction f is ``f'{prefix}_{f}'``.
        Defaults to ``variants`` for backward compatibility
        (prefix = variant, e.g. ``'informed'``).
    split : 'val', 'test', or 'both'.
        'both' shows val (dotted) and test (solid) as separate lines per prefix.
    model_colors : dict keyed by model_prefix (or variant if prefixes not set).
        Defaults to matplotlib tab10 colours.
    avg_over_props : if True, average the metric over all properties per fold
        before forming the per-fraction CI, and produce a single-panel figure
        instead of a 2×2 grid.

    Notes
    -----
    The CI bands use **fold-level** observations (n=25 per fraction): one
    metric per fold, paired across models by ``cv_cycle``.  When properties
    are averaged the per-fold value is the cross-property mean for that fold.
    """
    prefixes = model_prefixes if model_prefixes is not None else variants
    sub = mol_metrics if split == 'both' else mol_metrics[mol_metrics['split'] == split]

    _default_colors = {p: c for p, c in zip(prefixes, plt.rcParams['axes.prop_cycle'].by_key()['color'])}
    colors = model_colors or _default_colors

    splits_to_plot = ['val', 'test'] if split == 'both' else [split]
    linestyles_split = {'test': '-', 'val': ':'}

    def _repeat_ci(prefix: str, frac: float, spl: str,
                   props: list[str]) -> tuple[float, float] | None:
        """Return (mean, ci_half) of fold-level values averaged over props."""
        model = f'{prefix}_{frac}'
        if model not in models:
            return None
        src = (mol_metrics if split == 'both' else sub)
        mask = (src['method'] == model) & src['property'].isin(props)
        if split == 'both':
            mask &= src['split'] == spl
        prop_sub = src[mask]
        if prop_sub.empty:
            return None
        # Per-fold value (averaged over props if multiple); n = #folds (= 25)
        fold_vals = prop_sub.groupby('cv_cycle')[metric].mean()
        n = len(fold_vals)
        if n == 0:
            return None
        mean_v = float(fold_vals.mean())
        sem_v  = float(fold_vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        t_crit = t_dist.ppf(0.975, max(n - 1, 1)) if n > 1 else 0.0
        return mean_v, t_crit * sem_v

    def _draw_lines(ax, props: list[str]) -> None:
        for prefix in prefixes:
            color = colors.get(prefix, None)
            for spl in splits_to_plot:
                fracs_plot, means_plot, ci_plot = [], [], []
                for frac in fractions:
                    result = _repeat_ci(prefix, frac, spl, props)
                    if result is None:
                        continue
                    fracs_plot.append(frac)
                    means_plot.append(result[0])
                    ci_plot.append(result[1])
                if not fracs_plot:
                    continue
                fracs_arr = np.array(fracs_plot)
                means_arr = np.array(means_plot)
                ci_arr    = np.array(ci_plot)
                ls = linestyles_split[spl]
                variant_name = next((v for v in variants if v in prefix), prefix)
                label = variant_name.capitalize() if split != 'both' else f'{variant_name.capitalize()} ({spl})'
                ax.plot(fracs_arr, means_arr, linestyle=ls, marker='o',
                        color=color, label=label, linewidth=1.5, markersize=5)
                ax.fill_between(fracs_arr, means_arr - ci_arr, means_arr + ci_arr,
                                alpha=0.15, color=color)

    def _style_ax(ax, title: str, show_xlabel: bool, show_ylabel: bool) -> None:
        ax.set_title(title, fontsize=_FONT_SIZES['title'], fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(fractions)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        ax.tick_params(axis='x', labelsize=_FONT_SIZES['tick'])
        ax.tick_params(axis='y', labelsize=_FONT_SIZES['tick'])
        if show_xlabel:
            ax.set_xlabel('Training fraction', fontsize=_FONT_SIZES['label'])
        if show_ylabel:
            if metric == 'R2':
                ax.set_ylabel('$R^2$', fontsize=_FONT_SIZES['label'])
            else:
                ax.set_ylabel(metric, fontsize=_FONT_SIZES['label'])
        ax.grid(alpha=0.3, linewidth=0.5)

    if avg_over_props:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        _draw_lines(ax, MOLECULAR_PROPS)
        prop_str = ', '.join(_PROP_LABELS.get(p, p) for p in MOLECULAR_PROPS)
        _style_ax(ax, f'{metric} averaged over [{prop_str}]',
                  show_xlabel=True, show_ylabel=True)
        handles, labels_leg = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels_leg, fontsize=_FONT_SIZES['legend'],
                      frameon=True, edgecolor='#cccccc', loc = 'lower right')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=False, squeeze=False)
        axes_flat = axes.ravel()
        for ax_idx, prop in enumerate(MOLECULAR_PROPS):
            ax = axes_flat[ax_idx]
            _draw_lines(ax, [prop])
            _style_ax(ax, _PROP_LABELS.get(prop, prop),
                      show_xlabel=(ax_idx in (2, 3)),
                      show_ylabel=(ax_idx in (0, 2)))
        handles, labels_leg = axes_flat[0].get_legend_handles_labels()
        if handles:
            axes_flat[0].legend(handles, labels_leg, fontsize=_FONT_SIZES['legend'],
                                frameon=True, edgecolor='#cccccc', loc = 'lower right')

    fig.tight_layout()
    _save_fig(fig, save_path, save_fmt)
    plt.show()


# ── Parity hexbin ─────────────────────────────────────────────────────────────

def parity_mol(
    model: str,
    experiment_dir: str,
    variant: str,
    frac: float,
    mol_metrics: pd.DataFrame,
    properties: list[str] = MOLECULAR_PROPS,
    split: str = 'test',
    panel_metric: str = 'CCC',
    cmap: str = 'viridis',
    color: Any = None,
    marginal_log: bool = False,
    save_path: str | None = None,
    save_fmt: str = 'pdf',
) -> None:
    """Parity hexbin 2×2 grid — one panel per QM property, for a single model.

    Parameters
    ----------
    model : method name as it appears in mol_metrics, e.g. 'cutoff_informed_1.0'.
    experiment_dir : root of the experiment, e.g. experiments/molecular.
    variant : subdirectory name within experiment_dir, e.g. 'informed'.
    frac : training fraction, e.g. 1.0.

    Every molecule appears in exactly 5 test folds (one per repeat).
    Predictions are averaged over those 5 folds per molecule.

    The panel metric (CCC mean ± std) is read from mol_metrics so it matches
    the Tukey analysis — it is NOT recomputed from the averaged predictions.
    """
    if split not in ('val', 'test'):
        raise ValueError("parity_mol supports split='val' or 'test' only")

    folds = _get_folds(experiment_dir, variant)
    dot_color = color or plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    # Load and average predictions per molecule across all folds, for all properties
    pred_accum:   dict[str, dict[int, list[float]]] = {p: defaultdict(list) for p in properties}
    target_accum: dict[str, dict[int, float]]       = {p: {} for p in properties}

    for fold in folds:
        path = os.path.join(experiment_dir, variant,
                            f'fold_{fold}', f'frac_{frac}', f'{split}_preds.pkl')
        df = _safe_read_pickle(path)
        if df.empty:
            continue
        for prop in properties:
            p_col, t_col = f'pred_{prop}', f'target_{prop}'
            if p_col not in df.columns:
                continue
            for mol_id, p_val, t_val in zip(df.index, df[p_col], df[t_col]):
                pred_accum[prop][mol_id].append(float(p_val))
                if mol_id not in target_accum[prop]:
                    target_accum[prop][mol_id] = float(t_val)

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), squeeze=False)
    axes_flat = axes.ravel()

    for ax_idx, prop in enumerate(properties):
        ax = axes_flat[ax_idx]

        if not pred_accum[prop]:
            ax.set_title(f'{_PROP_LABELS.get(prop, prop)} (no data)')
            continue

        mol_ids = sorted(pred_accum[prop].keys())
        arr_t   = np.array([target_accum[prop][m] for m in mol_ids])
        arr_p   = np.array([np.mean(pred_accum[prop][m]) for m in mol_ids])

        data_min = min(arr_t.min(), arr_p.min())
        data_max = max(arr_t.max(), arr_p.max())
        margin   = 0.05 * (data_max - data_min) if data_max > data_min else 0.1
        lim_lo, lim_hi = data_min - margin, data_max + margin

        hb = ax.hexbin(arr_t, arr_p, gridsize=70, mincnt=1,
                       cmap=cmap, norm=LogNorm(), linewidths=0.2)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', lw=0.8)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)

        # Marginals first — make_axes_locatable shrinks the main axes,
        # so the inset colorbar must be placed afterwards.
        _add_marginal_density(ax, arr_t, [arr_p], [dot_color],
                              marginal_log=marginal_log)

        # Inset colorbar (placed after marginals)
        bg = ax.inset_axes([0.44, 0.06, 0.55, 0.18])
        bg.set_facecolor('white'); bg.patch.set_alpha(0.92)
        bg.set_xticks([]); bg.set_yticks([])
        for sp in bg.spines.values():
            sp.set_visible(False)
        cax = ax.inset_axes([0.50, 0.10, 0.46, 0.045])
        cb = plt.colorbar(hb, cax=cax, orientation='horizontal')
        if isinstance(getattr(hb, 'norm', None), LogNorm):
            cb.ax.xaxis.set_major_locator(LogLocator(base=10.0))
            cb.ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))  # type: ignore[arg-type]
        cb.ax.tick_params(labelsize=8, length=2, pad=3)
        cb.outline.set_linewidth(0.4)
        cax.set_title('Count', fontsize=9, pad=4)

        # In-panel metric annotation from mol_metrics
        metric_vals = mol_metrics[
            (mol_metrics['method'] == model) &
            (mol_metrics['split']  == split) &
            (mol_metrics['property'] == prop)
        ][panel_metric].dropna()
        if not metric_vals.empty:
            metric_label = '$R^2$' if panel_metric == 'R2' else panel_metric
            ann = f'{metric_label}={metric_vals.mean():.3f}±{metric_vals.std():.3f}'
        else:
            ann = None

        _bbox = dict(boxstyle='round,pad=0.25', facecolor='white',
                     alpha=0.90, edgecolor='#cccccc')
        prop_label = _PROP_LABELS.get(prop, prop)
        label_text = f'{prop_label}\n{ann}' if ann else prop_label
        ax.text(0.03, 0.97, label_text, transform=ax.transAxes,
                va='top', ha='left', fontsize=10, linespacing=1.35, bbox=_bbox)

        ax.tick_params(axis='both', labelsize=_FONT_SIZES['tick'])
        if ax_idx in (2, 3):
            ax.set_xlabel(_axis_label('target', prop), fontsize=_FONT_SIZES['label'])
        if ax_idx in (0, 2):
            ax.set_ylabel(_axis_label('pred', prop), fontsize=_FONT_SIZES['label'])

    fig.suptitle(f'Parity — {model} ({split})', fontsize=_FONT_SIZES['suptitle'])
    fig.tight_layout()
    _save_fig(fig, save_path, save_fmt)
    plt.show()


# ── Thin wrappers ─────────────────────────────────────────────────────────────

def metric_summary_mol(
    mol_metrics: pd.DataFrame,
    models: list[str],
    metric: str = 'CCC',
    split: str = 'test',
    ci_level: float = 0.95,
    n_decimals: int = 3,
    print_table: bool = True,
) -> pd.DataFrame:
    """Mean ± CI with rank per (property, model), wrapping ra.metric_summary_table.

    Fold-level (n=25): each fold is one observation; CI uses df=24.
    """
    sub: pd.DataFrame = (mol_metrics if split == 'both'
                         else mol_metrics[mol_metrics['split'] == split].copy())
    return ra.metric_summary_table(
        _fold_level_df(sub),
        models=models,
        metric=metric,
        properties=MOLECULAR_PROPS,
        panels=['global'],
        panel_col='panel',
        split=None,
        ci_level=ci_level,
        n_decimals=n_decimals,
        print_table=print_table,
    )


def informed_vs_blind_table(
    mol_metrics: pd.DataFrame,
    fractions: list[float],
    experiment_tag: str = '',
    properties: list[str] = MOLECULAR_PROPS,
    metric: str = 'CCC',
    split: str = 'test',
    alpha: float = 0.05,
    print_table: bool = True,
) -> pd.DataFrame:
    """For each (fraction, property): test if informed beats blind significantly.

    Runs a 2-model **fold-level** RM-ANOVA + Tukey comparison (n=25 paired
    folds via ``cv_cycle``) between ``informed`` and ``blind`` at each training
    fraction independently.

    Parameters
    ----------
    experiment_tag : prepended to model names, e.g. ``'cutoff'`` → models
        ``'cutoff_informed_{frac}'`` and ``'cutoff_blind_{frac}'``.
        Pass ``''`` for untagged names (``'informed_{frac}'``).

    Returns
    -------
    DataFrame indexed by fraction, columns = properties.
    Cell values: ``'Informed'``, ``'Blind'``, or ``'Non-sig.'``.
    """
    higher = metric in _HIGHER_IS_BETTER
    sub: pd.DataFrame = (mol_metrics if split == 'both'
                         else mol_metrics[mol_metrics['split'] == split].copy())
    sub_fold = _fold_level_df(sub)
    prefix = f'{experiment_tag}_' if experiment_tag else ''

    rows = []
    for frac in fractions:
        m_inf   = f'{prefix}informed_{frac}'
        m_blind = f'{prefix}blind_{frac}'
        row: dict = {'fraction': frac}

        for prop in properties:
            tukey = ra.tukey_pvalue_matrix(
                sub_fold,
                models=[m_inf, m_blind],
                metric=metric,
                properties=[prop],
                panels=['global'],
                panel_col='panel',
                split=None,
                avg_over_panels=False,
                avg_over_props=False,
                alpha=alpha,
                print_tables=False,
            )
            res = tukey.get(('global', prop))
            if res is None or res.pc.empty:
                row[prop] = '—'
                continue

            p_val = float(res.pc.loc[m_inf, m_blind])
            if p_val >= alpha:
                row[prop] = 'Non-sig.'
            else:
                if higher:
                    winner = 'Informed' if res.means[m_inf] > res.means[m_blind] else 'Blind'
                else:
                    winner = 'Informed' if res.means[m_inf] < res.means[m_blind] else 'Blind'
                row[prop] = winner

        rows.append(row)

    out = pd.DataFrame(rows).set_index('fraction')

    if print_table:
        print(f'\nInformed vs Blind — {metric} ({split}, α={alpha})')
        print(out.to_string())
        print()

    return out


def anova_diagnostics_mol(
    mol_metrics: pd.DataFrame,
    models: list[str],
    properties: list[str] = MOLECULAR_PROPS,
    metric: str = 'CCC',
    split: str = 'test',
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """Fold-level ANOVA validity diagnostics: Shapiro-Wilk and Levene.

    Intended use: pass all models for one (family, variant) pair, e.g.
    ``['cutoff_informed_0.01', 'cutoff_informed_0.05', ...]``, so that Levene
    homoscedasticity is tested across those fractions — exactly the comparison
    being validated by the learning-curve Tukey test.

    The diagnostics run on the n=25 fold-level metric values per model
    (paired via ``cv_cycle``).  ICC is *not* computed because fold-level
    analysis treats each fold as an independent observation; the ICC value
    is what justifies that step (was checked pre-refactor, ICC ≈ 0).

    Returns
    -------
    dict with key:
      'anova' – DataFrame: panel, property, model, p_shapiro, p_levene,
                           shapiro_ok, levene_ok, n_repeats (= n_folds).
    """
    sub: pd.DataFrame = (mol_metrics if split == 'both'
                         else mol_metrics[mol_metrics['split'] == split].copy())
    anova_df = ra.check_anova_assumptions(
        _fold_level_df(sub),     # repeat=cv_cycle → SW/Levene on n=25 per model
        models=models, panel_col='panel', panels=['global'],
        properties=properties, metric=metric,
        split=None, alpha=alpha,
    )
    return {'anova': anova_df}


# Reuse colour/label constants from result_analysis so the LaTeX output is consistent.
_ORANGE_BG = ra._ORANGE_BG if hasattr(ra, '_ORANGE_BG') else r'\cellcolor{orange!25}'
_RED_BG    = ra._RED_BG    if hasattr(ra, '_RED_BG')    else r'\cellcolor{red!20}'
_DIAG_NAMES   = ('SW', 'Lev')
_DIAG_HEADERS = {
    'SW':    r'SW',
    'Lev':   r'Lev.',
}
_METRIC_LABELS = {
    'CCC': 'CCC', 'Spearman': r'Spearman $\rho$',
    'R2': r'$R^2$', 'MAE': 'MAE', 'RMSE': 'RMSE',
}


def _gather_anova_components(
    mol_metrics: pd.DataFrame,
    models: list[str],
    properties: list[str],
    metric: str,
    split: str,
    avg_over_props: bool,
) -> list[dict]:
    """Run RM-ANOVA per property (or pooled) and return the components for each row."""
    sub = mol_metrics if split == 'both' else mol_metrics[mol_metrics['split'] == split]
    sub = sub[sub['method'].isin(models) & sub['property'].isin(properties)]

    rows: list[dict] = []
    if avg_over_props:
        avg_df = (
            sub.groupby(['method', 'cv_cycle'])[metric]
            .mean().reset_index()
        )
        comp = _compute_rm_anova(avg_df, metric, subject='cv_cycle')
        if comp:
            comp['property'] = 'avg'
            rows.append(comp)
    else:
        for prop in properties:
            prop_sub = sub[sub['property'] == prop][['method', 'cv_cycle', metric]].dropna()
            comp = _compute_rm_anova(prop_sub, metric, subject='cv_cycle')
            if comp:
                comp['property'] = prop
                rows.append(comp)
    return rows


def print_anova_table_mol(
    mol_metrics: pd.DataFrame,
    models: list[str],
    metric: str = 'CCC',
    properties: list[str] = MOLECULAR_PROPS,
    split: str = 'test',
    avg_over_props: bool = False,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Print classic RM-ANOVA tables (Source / SS / df / MS / F / p) per property.

    Uses fold-level paired observations: 25 folds × k models, with ``cv_cycle``
    as the within-subject factor (justification: disjoint test sets within
    repeats; verified ICC ≈ 0).

    Parameters
    ----------
    avg_over_props : average the metric over properties at the fold level
        before the ANOVA → one pooled table.

    Returns
    -------
    DataFrame with one row per property (or one row total if avg_over_props).
    """
    comps = _gather_anova_components(
        mol_metrics, models, properties, metric, split, avg_over_props,
    )
    if not comps:
        warnings.warn('No data for ANOVA table.')
        return pd.DataFrame()

    fmt = '  {:<14} {:>12} {:>5} {:>12} {:>10} {:>14}'
    for comp in comps:
        prop_label = comp['property']
        print(f'\n=== RM-ANOVA — {metric} ({prop_label}) ===')
        print(f'  n={comp["n"]} folds × k={comp["k"]} models '
              f'(paired by cv_cycle)')
        print()
        print(fmt.format('Source', 'SS', 'df', 'MS', 'F', 'p-value'))
        print('  ' + '─' * 70)

        p = comp['p_method']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < alpha else ''
        print(fmt.format(
            'Method',
            f'{comp["SS_method"]:.4e}',  comp['df_method'],
            f'{comp["MS_method"]:.4e}',  f'{comp["F_method"]:.3f}',
            f'{p:.3e} {sig}'.strip(),
        ))
        print(fmt.format(
            'Subject (fold)',
            f'{comp["SS_subject"]:.4e}', comp['df_subject'],
            f'{comp["MS_subject"]:.4e}', '', '',
        ))
        print(fmt.format(
            'Error',
            f'{comp["SS_error"]:.4e}',   comp['df_error'],
            f'{comp["MS_error"]:.4e}',   '', '',
        ))
        print('  ' + '─' * 70)
        print(fmt.format(
            'Total',
            f'{comp["SS_total"]:.4e}',   comp['df_total'], '', '', '',
        ))
        print()
    print(f'Significance codes: *** p<0.001, ** p<0.01, * p<{alpha}')
    return pd.DataFrame(comps)


def latex_anova_table_mol(
    mol_metrics: pd.DataFrame,
    models: list[str],
    metric: str = 'CCC',
    properties: list[str] = MOLECULAR_PROPS,
    split: str = 'test',
    avg_over_props: bool = False,
    family: str = '',
    variant: str = '',
    alpha: float = 0.05,
    n_decimals: int = 4,
    caption: str | None = None,
    label: str = 'tab:anova_mol',
    landscape: bool = False,
    print_preamble: bool = True,
) -> str:
    """LaTeX RM-ANOVA table (one block per property, or one pooled block).

    Each block has Source × (SS, df, MS, F, p-value) rows.  Significance is
    coloured: orange if ``alpha ≤ p < 10·alpha``, red if ``p < alpha``.
    """
    comps = _gather_anova_components(
        mol_metrics, models, properties, metric, split, avg_over_props,
    )
    if not comps:
        warnings.warn('No data for LaTeX ANOVA table.')
        return ''

    col_sep = ' & '
    row_end = r' \\' + '\n'

    def _esc(s: str) -> str:
        return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    def _ss(x: float) -> str:
        return f'{x:.{n_decimals}e}'

    def _ms(x: float) -> str:
        return f'{x:.{n_decimals}e}'

    def _p_cell(p: float) -> str:
        s = f'{p:.{n_decimals}e}'
        if p < alpha:
            return _RED_BG + s
        if p < 10 * alpha:
            return _ORANGE_BG + s
        return s

    def _F_cell(F: float) -> str:
        return f'{F:.3f}' if np.isfinite(F) else '---'

    n_props = len(comps)
    has_property_col = (n_props > 1) or not avg_over_props
    extra_col = 1 if has_property_col else 0
    col_spec = ('l' * (1 + extra_col)) + 'r' * 5

    headers = ['Source']
    if has_property_col:
        headers.insert(0, 'Property')
    headers += ['SS', 'df', 'MS', 'F', 'p-value']
    header_str = col_sep.join(headers) + row_end

    body_lines: list[str] = []
    for c_idx, comp in enumerate(comps):
        prop_label = _PROP_LABELS.get(comp['property'], _esc(comp['property']))
        block_rows = [
            ('Method',         comp['SS_method'],  comp['df_method'],
             comp['MS_method'], _F_cell(comp['F_method']), _p_cell(comp['p_method'])),
            ('Subject (fold)', comp['SS_subject'], comp['df_subject'],
             comp['MS_subject'], '', ''),
            ('Error',          comp['SS_error'],   comp['df_error'],
             comp['MS_error'],   '', ''),
            ('Total',          comp['SS_total'],   comp['df_total'],
             None,               '', ''),
        ]
        for r_idx, (src, ss, df_, ms, fv, pv) in enumerate(block_rows):
            cells: list[str] = []
            if has_property_col:
                cells.append(
                    f'\\multirow{{{len(block_rows)}}}{{*}}{{{prop_label}}}'
                    if r_idx == 0 else ''
                )
            cells.append(_esc(src))
            cells.append(_ss(ss))
            cells.append(str(df_))
            cells.append(_ms(ms) if ms is not None else '')
            cells.append(fv)
            cells.append(pv)
            body_lines.append(col_sep.join(cells) + row_end)
        if c_idx < len(comps) - 1:
            body_lines.append('\\midrule\n')

    if caption is None:
        scope = 'properties averaged' if avg_over_props else 'per property'
        ctx = ''
        if family or variant:
            bits = [b for b in (family, variant) if b]
            ctx = f' for \\textbf{{{_esc(" / ".join(bits))}}}'
        n_obs = comps[0]['n']
        k_obs = comps[0]['k']
        caption = (
            f'Fold-level RM-ANOVA on {_METRIC_LABELS.get(metric, _esc(metric))}'
            f'{ctx} ({scope}). '
            f'$n={n_obs}$ folds paired across $k={k_obs}$ models via '
            r'\texttt{cv\_cycle}; $\mathit{df}_{\text{err}}=(n{-}1)(k{-}1)$. '
            f'p-values shaded \\colorbox{{red!20}}{{red}} when $p<\\alpha={alpha}$ and '
            f'\\colorbox{{orange!25}}{{orange}} when $\\alpha\\le p<10\\alpha$.'
        )

    parts: list[str] = []
    if print_preamble:
        parts.append('% Required packages: booktabs, multirow, xcolor, colortbl\n')

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


def latex_anova_diagnostics_mol(
    diag: dict[str, pd.DataFrame],
    models: list[str],
    fractions: list[float],
    family: str,
    variant: str,
    properties: list[str] = MOLECULAR_PROPS,
    metric: str = 'CCC',
    avg_over_props: bool = False,
    show_pvalues: bool = False,
    include_intervals: bool = True,
    icc_color_thresholds: tuple = (0.20, 0.40),
    neff_nominal: float = 25.0,
    neff_color_fracs: tuple = (0.80, 0.60),
    passrate_color_thresholds: tuple = (80.0, 60.0),
    pval_color_thresholds: tuple = (0.10, 0.05),
    n_decimals: int = 3,
    caption: str | None = None,
    label: str = 'tab:anova_diag_mol',
    landscape: bool = False,
    print_preamble: bool = True,
) -> str:
    """LaTeX ANOVA validity table for one (family, variant) pair.

    Layout
    ------
    * avg_over_props=True  → rows = fractions (one per model), cols = (SW, Lev).
    * avg_over_props=False → rows = fraction × property with \\multirow on fraction,
                             cols = (SW, Lev).

    The diagnostics are fold-level: SW tests normality of the 25 fold-level
    metric values per model; Levene tests homoscedasticity of those 25 values
    across all fractions in ``models``.  ICC is omitted (see
    :func:`anova_diagnostics_mol`).

    Parameters
    ----------
    diag      : output of :func:`anova_diagnostics_mol`.
    models    : model names parallel to fractions,
                e.g. ``['cutoff_informed_0.01', 'cutoff_informed_0.05', ...]``.
    fractions : training fraction values used as row labels.
    family, variant : appear in the auto-generated caption.
    """
    # Empty icc_df keeps `_aggregate_diag_cell` happy without ICC data.
    icc_df   = pd.DataFrame()
    anova_df = diag.get('anova', pd.DataFrame())

    col_sep = ' & '
    row_end = r' \\' + '\n'

    def _escape(s: str) -> str:
        return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    def _fmt(diag_name: str, value: float, sd: float) -> str:
        if pd.isna(value):
            return '---'
        if diag_name in ('SW', 'Lev') and not show_pvalues:
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

    def _cell(model: str, props_in_cell: list[str], diag_name: str) -> str:
        vals = ra._aggregate_diag_cell(
            icc_df, anova_df, model, ['global'], props_in_cell,
            show_pvalues=show_pvalues,
        )
        v, sd = vals.get(diag_name, (np.nan, np.nan))
        return _color(diag_name, v, _fmt(diag_name, v, sd))

    # Column spec and header
    if avg_over_props:
        col_spec = 'r' + 'c' * len(_DIAG_NAMES)
        header = (col_sep.join(
            ['Fraction'] + [_DIAG_HEADERS[d] for d in _DIAG_NAMES]
        ) + row_end)
    else:
        col_spec = 'rr' + 'c' * len(_DIAG_NAMES)
        header = (col_sep.join(
            ['Fraction', 'Property'] + [_DIAG_HEADERS[d] for d in _DIAG_NAMES]
        ) + row_end)

    # Body
    body_lines: list[str] = []
    n_props = len(properties)

    for f_idx, (frac, model) in enumerate(zip(fractions, models)):
        if avg_over_props:
            cells = [_cell(model, properties, d) for d in _DIAG_NAMES]
            body_lines.append(col_sep.join([str(frac)] + cells) + row_end)
        else:
            for p_idx, prop in enumerate(properties):
                frac_cell = (
                    f'\\multirow{{{n_props}}}{{*}}{{{frac}}}' if p_idx == 0 else ''
                )
                prop_label = _PROP_LABELS.get(prop, _escape(prop))
                cells = [_cell(model, [prop], d) for d in _DIAG_NAMES]
                body_lines.append(
                    col_sep.join([frac_cell, prop_label] + cells) + row_end
                )
            if f_idx < len(fractions) - 1:
                body_lines.append('\\addlinespace[2pt]\n')

    # Caption
    if caption is None:
        prop_scope = 'properties averaged' if avg_over_props else 'per property'
        sw_desc = ('mean $p$-value' if show_pvalues else 'pass rate \\%')
        caption = (
            f'Fold-level ANOVA validity diagnostics for \\textbf{{{_escape(family)}}} '
            f'family, \\textbf{{{_escape(variant)}}} variant, metric: '
            f'{_METRIC_LABELS.get(metric, _escape(metric))} ({prop_scope}). '
            f'\\textbf{{SW}}: Shapiro--Wilk normality on the 25 fold-level metric '
            f'values per model ({sw_desc}); '
            f'\\textbf{{Lev.}}: Levene homoscedasticity across fractions on those '
            f'25 values ({sw_desc}).'
        )

    # Assemble
    parts: list[str] = []
    if print_preamble:
        parts.append('% Required packages: booktabs, multirow, xcolor, colortbl\n')

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
        _ind(header),
        '    \\midrule\n',
        *[_ind(ln) for ln in body_lines],
        '    \\bottomrule\n',
        '  \\end{tabular}\n',
        f'\\end{{{tbl_env}}}\n',
    ]
    table_str = ''.join(parts)
    print(table_str)
    return table_str


# ── Paired Informed-vs-Blind tests (per fraction × property) ─────────────────
#
# Rationale
# ---------
# A paired test on fold-aligned (Informed, Blind) values does not require
# equal variance between groups, only that the differences d_i are
# well-behaved (paired t: approximately normal; Wilcoxon: roughly symmetric,
# no normality).  Levene-failure across training fractions — a mathematical
# expectation of learning curves — is therefore irrelevant.

def _paired_arrays(
    mol_metrics: pd.DataFrame,
    m_inf: str,
    m_blind: str,
    prop: str,
    split: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (cv_cycles, vals_inf, vals_blind) aligned by cv_cycle."""
    sub = mol_metrics if split == 'both' else mol_metrics[mol_metrics['split'] == split]
    sub = sub[sub['property'] == prop]
    inf_s   = sub[sub['method'] == m_inf  ].set_index('cv_cycle')[metric].dropna()
    blind_s = sub[sub['method'] == m_blind].set_index('cv_cycle')[metric].dropna()
    common  = inf_s.index.intersection(blind_s.index)
    if len(common) < 3:
        return None
    common = sorted(common)
    cycles = np.array(common)
    return cycles, inf_s.loc[common].to_numpy(), blind_s.loc[common].to_numpy()


def _icc_one_way(values: np.ndarray, groups: np.ndarray) -> float:
    """ICC(1,1) one-way random-effects, single-rater on grouped observations.

    Measures the proportion of variance attributable to between-group
    differences. Near zero ⇒ groups carry no extra information ⇒ observations
    can be treated as independent across groups.
    """
    df = pd.DataFrame({'v': values, 'g': groups})
    g_means = df.groupby('g')['v'].mean()
    g_sizes = df.groupby('g')['v'].size()
    n_groups = len(g_means)
    if n_groups < 2 or g_sizes.min() < 2:
        return np.nan
    k = float(g_sizes.mean())
    grand = float(df['v'].mean())
    SSB = float(((g_means - grand) ** 2 * g_sizes).sum())
    SSW = float(df.groupby('g')['v']
                  .apply(lambda s: float(((s - s.mean()) ** 2).sum())).sum())
    MSB = SSB / (n_groups - 1)
    MSW = SSW / (len(df) - n_groups) if (len(df) - n_groups) > 0 else np.nan
    denom = MSB + (k - 1) * MSW
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float((MSB - MSW) / denom)


def _sw_pvalue(x: np.ndarray) -> float:
    if x is None or len(x) < 3:
        return np.nan
    try:
        return float(shapiro(x).pvalue)
    except Exception:
        return np.nan


def paired_informed_vs_blind_table(
    mol_metrics: pd.DataFrame,
    fractions: list[float],
    experiment_tag: str = 'cutoff',
    properties: list[str] = MOLECULAR_PROPS,
    metric: str = 'R2',
    split: str = 'test',
    alpha: float = 0.05,
    print_table: bool = True,
) -> pd.DataFrame:
    """Paired Informed-vs-Blind comparison per (fraction, property).

    For each (fraction, property) computes on the n=25 fold-paired values:
      - mean and 95 %% CI of the difference d = metric(Informed) − metric(Blind)
      - paired-t two-sided p-value (assumes d approx. normal; CLT-friendly)
      - Wilcoxon signed-rank two-sided p-value (no normality assumption)
      - winner label (uses paired-t; flagged 'Non-sig.' if both tests p ≥ alpha)

    Notes
    -----
    Equal variance between Informed and Blind is NOT required; the only
    relevant assumption is on the *differences*.  Diagnose that with
    :func:`paired_diagnostics_mol`.
    """
    higher = metric in _HIGHER_IS_BETTER
    prefix = f'{experiment_tag}_' if experiment_tag else ''
    rows: list[dict] = []
    for frac in fractions:
        m_inf, m_blind = f'{prefix}informed_{frac}', f'{prefix}blind_{frac}'
        for prop in properties:
            arrs = _paired_arrays(mol_metrics, m_inf, m_blind, prop, split, metric)
            base = {'fraction': frac, 'property': prop}
            if arrs is None:
                rows.append({**base, 'n': 0, 'mean_diff': np.nan,
                             'ci_lo': np.nan, 'ci_hi': np.nan,
                             'p_paired_t': np.nan, 'p_wilcoxon': np.nan,
                             'winner': '—'})
                continue
            _, inf_v, blind_v = arrs
            d = inf_v - blind_v
            n = len(d)
            mean_d = float(d.mean())
            sd     = float(d.std(ddof=1)) if n > 1 else 0.0
            sem    = sd / np.sqrt(n) if n > 1 else 0.0
            t_crit = float(t_dist.ppf(0.975, n - 1)) if n > 1 else 0.0
            ci_lo, ci_hi = mean_d - t_crit * sem, mean_d + t_crit * sem

            _, p_t = ttest_rel(inf_v, blind_v)
            try:
                _, p_w = wilcoxon(inf_v, blind_v, zero_method='wilcox')
            except ValueError:
                p_w = np.nan

            sig = (np.isfinite(p_t) and p_t < alpha) or (np.isfinite(p_w) and p_w < alpha)
            if not sig:
                winner = 'Non-sig.'
            elif higher:
                winner = 'Informed' if mean_d > 0 else 'Blind'
            else:
                winner = 'Informed' if mean_d < 0 else 'Blind'

            rows.append({**base, 'n': n,
                         'mean_diff': mean_d, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                         'p_paired_t': float(p_t), 'p_wilcoxon': float(p_w),
                         'winner': winner})
    out = pd.DataFrame(rows)
    if print_table:
        print(f'\nPaired Informed vs Blind — {metric} ({split}, α={alpha}, '
              f'tag={experiment_tag!r})')
        print(out.to_string(index=False))
        print()
    return out


def paired_diagnostics_mol(
    mol_metrics: pd.DataFrame,
    fractions: list[float],
    experiment_tag: str = 'cutoff',
    properties: list[str] = MOLECULAR_PROPS,
    metric: str = 'R2',
    split: str = 'test',
) -> pd.DataFrame:
    """Per-(fraction, property) paired-test validity diagnostics.

    Returns a DataFrame with one row per (fraction, property) and columns:

        n, mean_inf, std_inf, mean_blind, std_blind,
        p_sw_inf, p_sw_blind, p_sw_diff,   # Shapiro–Wilk normality p-values
        p_paired_t, p_wilcoxon             # paired-test p-values

    The relevant assumption for the paired-t is normality of the *differences*
    (``p_sw_diff``).  Per-group SW (``p_sw_inf``, ``p_sw_blind``) is reported
    as descriptive context only.  Levene/homoscedasticity is intentionally
    omitted: paired tests do not require it.
    """
    prefix = f'{experiment_tag}_' if experiment_tag else ''
    rows: list[dict] = []
    for frac in fractions:
        m_inf, m_blind = f'{prefix}informed_{frac}', f'{prefix}blind_{frac}'
        for prop in properties:
            arrs = _paired_arrays(mol_metrics, m_inf, m_blind, prop, split, metric)
            base = {'fraction': frac, 'property': prop}
            if arrs is None:
                rows.append({**base, 'n': 0})
                continue
            cycles, inf_v, blind_v = arrs
            d = inf_v - blind_v
            repeats = cycles // 5
            icc_diff = _icc_one_way(d, repeats)
            _, p_t = ttest_rel(inf_v, blind_v)
            try:
                _, p_w = wilcoxon(inf_v, blind_v, zero_method='wilcox')
            except ValueError:
                p_w = np.nan
            n = len(d)
            std_d = float(d.std(ddof=1)) if n > 1 else 0.0
            rows.append({
                **base, 'n': n,
                'mean_inf':   float(inf_v.mean()),   'std_inf':   float(inf_v.std(ddof=1)),
                'mean_blind': float(blind_v.mean()), 'std_blind': float(blind_v.std(ddof=1)),
                'mean_diff':  float(d.mean()),       'std_diff':  std_d,
                'sem_diff':   std_d / np.sqrt(n) if n > 0 else np.nan,
                'icc_diff':   icc_diff,
                'p_sw_inf':   _sw_pvalue(inf_v),
                'p_sw_blind': _sw_pvalue(blind_v),
                'p_sw_diff':  _sw_pvalue(d),
                'p_paired_t': float(p_t),
                'p_wilcoxon': float(p_w),
            })
    return pd.DataFrame(rows)


def latex_paired_diagnostics_mol(
    diag_df: pd.DataFrame,
    fractions: list[float],
    properties: list[str] = MOLECULAR_PROPS,
    metric: str = 'R2',
    family: str = 'cutoff',
    alpha: float = 0.05,
    n_decimals: int = 3,
    caption: str | None = None,
    label: str = 'tab:paired_diag_mol',
    landscape: bool = False,
    print_preamble: bool = True,
) -> str:
    """LaTeX paired-test validity table.

    Layout
    ------
    * Header: each property spans a 2-column ``\\multicolumn`` (Informed | Blind).
    * Body : each fraction is a ``\\multirow`` over the test quantities

          mean ± std,  SW (per-model),  SW (Δ),  paired-t p,  Wilcoxon p.

      Per-model quantities (mean ± std, per-model SW) fill both sub-columns;
      the paired quantities (SW(Δ), paired-t, Wilcoxon) use ``\\multicolumn{2}``
      since they have a single value per (fraction, property).

    Coloring
    --------
    All p-value cells: red if ``p < alpha``, orange if ``alpha ≤ p < 10·alpha``.
    For the SW rows this flags **assumption violation**; for the paired-test
    rows it flags **statistical significance**.
    """
    n_props = len(properties)
    col_sep = ' & '
    row_end = r' \\' + '\n'

    def _esc(s: str) -> str:
        return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    def _f(x: float, nd: int = n_decimals) -> str:
        return '---' if not np.isfinite(x) else f'{x:.{nd}f}'

    def _p_cell(p: float) -> str:
        s = _f(p)
        if not np.isfinite(p):
            return s
        if p < alpha:
            return _RED_BG + s
        if p < 10 * alpha:
            return _ORANGE_BG + s
        return s

    # --- Build header (single column per property) ---
    col_spec = 'll' + ('c' * n_props)
    header_cells = ['Fraction', 'Quantity']
    for prop in properties:
        header_cells.append(_PROP_LABELS.get(prop, _esc(prop)))
    header_line = col_sep.join(header_cells) + row_end

    # --- Body ---
    metric_label = _METRIC_LABELS.get(metric, _esc(metric))
    quantities = [
        (f'$\\Delta${metric_label} $\\pm$ SEM',  'diff'),
        ('ICC ($\\Delta$)',                      'icc'),
        ('SW ($\\Delta$) $p$',                   'p'),
        ('paired-$t$ $p$',                       'p'),
        ('Wilcoxon $p$',                         'p'),
    ]

    def _icc_cell(v: float) -> str:
        s = _f(v)
        if not np.isfinite(v):
            return s
        a = abs(v)
        if a >= 0.4:
            return _RED_BG + s
        if a >= 0.2:
            return _ORANGE_BG + s
        return s
    body: list[str] = []
    for f_idx, frac in enumerate(fractions):
        sub = diag_df[diag_df['fraction'] == frac].set_index('property')
        for q_idx, (q_label, q_kind) in enumerate(quantities):
            cells: list[str] = []
            cells.append(
                f'\\multirow{{{len(quantities)}}}{{*}}{{{frac}}}'
                if q_idx == 0 else ''
            )
            cells.append(q_label)
            for prop in properties:
                if prop not in sub.index or sub.loc[prop].get('n', 0) == 0:
                    cells.append('---')
                    continue
                row = sub.loc[prop]
                if q_kind == 'diff':
                    cells.append(f'{_f(row.mean_diff)}$\\pm${_f(row.sem_diff)}')
                elif q_kind == 'icc':
                    cells.append(_icc_cell(row.icc_diff))
                elif q_label == 'SW ($\\Delta$) $p$':
                    cells.append(_p_cell(row.p_sw_diff))
                elif q_label == 'paired-$t$ $p$':
                    cells.append(_p_cell(row.p_paired_t))
                elif q_label == 'Wilcoxon $p$':
                    cells.append(_p_cell(row.p_wilcoxon))
            body.append(col_sep.join(cells) + row_end)
        if f_idx < len(fractions) - 1:
            body.append('\\midrule\n')

    # --- Caption ---
    if caption is None:
        caption = (
            f'Paired Informed-vs-Blind comparison for the '
            f'\\textbf{{{_esc(family)}}} family on '
            f'{_METRIC_LABELS.get(metric, _esc(metric))}, $n=25$ folds paired '
            f'by \\texttt{{cv\\_cycle}}. '
            f'\\textbf{{$\\Delta${_METRIC_LABELS.get(metric, _esc(metric))} $\\pm$ SEM}}: '
            f'fold-wise mean of $\\Delta_i = '
            f'{_METRIC_LABELS.get(metric, _esc(metric))}_{{\\text{{Inf}}}},_i - '
            f'{_METRIC_LABELS.get(metric, _esc(metric))}_{{\\text{{Blind}}}},_i$ '
            f'with standard error of that mean '
            f'(SEM $=$ std($\\Delta$)$/\\sqrt{{n}}$); this is the uncertainty '
            f'on the difference, not a CI half-width and not std($\\Delta$). '
            f'Positive values favour Informed. '
            f'\\textbf{{ICC ($\\Delta$)}}: one-way ICC(1,1) of the 25 fold-level '
            f'differences grouped by the 5 outer repeats (\\texttt{{cv\\_cycle}}'
            f'$//$\\,5); near zero $\\Rightarrow$ folds carry independent '
            f'information across repeats, supporting independence of $\\Delta_i$. '
            f'Shaded \\colorbox{{orange!25}}{{orange}} for $|$ICC$|\\ge 0.20$ '
            f'and \\colorbox{{red!20}}{{red}} for $|$ICC$|\\ge 0.40$. '
            f'\\textbf{{SW ($\\Delta$) $p$}}: Shapiro--Wilk on the 25 paired '
            f'differences --- the only assumption required for the paired $t$. '
            f'\\textbf{{paired-$t$ $p$}}, \\textbf{{Wilcoxon $p$}}: two-sided '
            f'$p$-values testing $\\mu_{{\\Delta}}=0$ (Wilcoxon needs only '
            f'symmetry of $\\Delta$ and serves as a robustness check when '
            f'SW($\\Delta$) is small). '
            f'Cells shaded \\colorbox{{red!20}}{{red}} when $p<\\alpha={alpha}$ '
            f'and \\colorbox{{orange!25}}{{orange}} when $\\alpha\\le p<10\\alpha$.'
        )

    # --- Assemble ---
    parts: list[str] = []
    if print_preamble:
        parts.append('% Required packages: booktabs, multirow, xcolor, colortbl\n')

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
        _ind(header_line),
        '    \\midrule\n',
        *[_ind(ln) for ln in body],
        '    \\bottomrule\n',
        '  \\end{tabular}\n',
        f'\\end{{{tbl_env}}}\n',
    ]
    out = ''.join(parts)
    print(out)
    return out


def latex_table_mol(
    mol_metrics: pd.DataFrame,
    models: list[str],
    metric: str = 'CCC',
    split: str = 'test',
    avg_over_props: bool = False,
    orient: str = 'tall',
    n_decimals: int = 3,
    caption: str | None = None,
    label: str = 'tab:mol_metrics',
) -> str:
    """LaTeX table for molecular results, wrapping ra.latex_metric_table.

    Fold-level (n=25); ranking colours come from the fold-level Tukey p-values.
    """
    sub: pd.DataFrame = (mol_metrics if split == 'both'
                         else mol_metrics[mol_metrics['split'] == split].copy())
    return ra.latex_metric_table(
        _fold_level_df(sub),
        models=models,
        metric=metric,
        properties=MOLECULAR_PROPS,
        panels=['global'],
        panel_col='panel',
        split=None,
        avg_over_panels=True,
        avg_over_props=avg_over_props,
        orient=orient,
        n_decimals=n_decimals,
        caption=caption,
        label=label,
    )
