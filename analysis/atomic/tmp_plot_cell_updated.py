"""Plotting utilities for CV analysis.

This file is intended to be imported from notebooks (e.g. analyze_CV_experiments.ipynb)
so plotting logic can be edited reliably outside the notebook.

Most functions expect certain experiment/data objects (e.g. elem_metrics) to be
provided by the notebook. Call `bind_notebook_globals(...)` after importing.
"""

from __future__ import annotations

import os
import json
import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import result_analysis as ra
from matplotlib.colors import LogNorm
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
)

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None  # noqa: A001

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10
})

# Notebook-provided globals (bound at runtime)
EXPERIMENT_DIR = None
ELEMENTS = None
CLUSTER_LABELS = None
MODEL_COLORS = None
comp_methods = None
elem_metrics = None
cluster_metrics = None

# Additional notebook-provided helpers (optional, but required for some plots)

rm_tukey_hsd = None

def get_folds(exp_dir, model):
    model_dir = os.path.join(exp_dir, model)
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

def safe_read_pickle(path):
    if not os.path.exists(path):
        warnings.warn(f'Missing pickle: {path}')
        return pd.DataFrame()
    try:
        return pd.read_pickle(path)
    except Exception as e:
        warnings.warn(f'Error reading {path}: {e}')
        return pd.DataFrame()


def bind_notebook_globals(mapping: Mapping[str, Any] | None = None, /, **kwargs: Any) -> None:
    """Bind notebook variables/functions into this module's global namespace.

    Typical usage in a notebook:

        import importlib, tmp_plot_cell_updated as plots
        importlib.reload(plots)
        plots.bind_notebook_globals(globals())

    You can also pass explicit keywords to override specific bindings.
    """

    if mapping is not None:
        for key, value in mapping.items():
            globals()[key] = value
    for key, value in kwargs.items():
        globals()[key] = value

# Metrics where higher values are better (used for Tukey "best" selection)
_HIGHER_IS_BETTER = {'R2', 'CCC', 'Spearman'}
_FONT_SIZES = {'title': 11, 'label': 11, 'tick': 10, 'legend': 10, 'suptitle': 12, 'annot': 10}


def _save_fig(fig, save_path, save_fmt):
    """Save figure if save_path is provided."""
    if save_path is not None:
        fig.savefig(save_path, format=save_fmt, bbox_inches='tight', dpi=300)


def boxplot_1x4(prop, metric, split, models=None, save_path=None, save_fmt='pdf'):
    """Box plot for one property, one metric, one split - across 4 elements."""
    models = models or comp_methods
    subdf = elem_metrics[
        (elem_metrics['property'] == prop) &
        (elem_metrics['split'] == split) &
        (elem_metrics['method'].isin(models))
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    for i, elem in enumerate(ELEMENTS):
        ax = axes[i]
        sns.boxplot(
            data=subdf[subdf['element'] == elem],
            x='method', y=metric, order=models,
            palette=[MODEL_COLORS[m] for m in models], ax=ax,
        )
        ax.set_title(elem); ax.set_xlabel('')
        ax.set_ylabel(metric if i == 0 else '')
    plt.suptitle(f"{split.upper()} – {prop} – {metric}")
    plt.tight_layout()
    _save_fig(fig, save_path, save_fmt)
    plt.show()


def tukey_ci_1x4(prop, metric, split, models=None, save_path=None, save_fmt='pdf'):
    """Tukey HSD simultaneous CI for one property – across 4 elements."""
    models = models or comp_methods
    subdf = elem_metrics[
        (elem_metrics['property'] == prop) &
        (elem_metrics['split'] == split) &
        (elem_metrics['method'].isin(models))
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for i, elem in enumerate(ELEMENTS):
        ax = axes[i]
        elem_data = subdf[subdf['element'] == elem]
        if elem_data.empty:
            ax.set_title(f"{elem} (no data)"); continue
        avg = elem_data.groupby(['method', 'repeat'])[metric].mean().reset_index()
        avg = avg.dropna(subset=[metric])
        if avg.empty or avg['method'].nunique() < 2:
            ax.set_title(f"{elem} (insufficient)"); continue
        try:
            tukey = pairwise_tukeyhsd(endog=avg[metric], groups=avg['method'], alpha=0.05)
            best = (avg.groupby('method')[metric].mean().idxmax() if metric in _HIGHER_IS_BETTER
                    else avg.groupby('method')[metric].mean().idxmin())
            tukey.plot_simultaneous(comparison_name=best, ax=ax)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes, ha='center')
        ax.set_title(elem)
    plt.suptitle(f"Tukey CI – {split.upper()} – {prop} – {metric}")
    plt.tight_layout()
    _save_fig(fig, save_path, save_fmt)
    plt.show()


def _pretty_prop_label(prop: str) -> str:
    mapping = {
        'N': r'$N$',
        'LI': r'$\lambda$',
        '|Mu|': r'$||\boldsymbol{\mu}||$',
        '|Q|': r'$||\boldsymbol{Q}||$',
    }
    return mapping.get(prop, str(prop))


_LABEL_XY = (0.03, 0.97)                # top-left anchor for the panel name / R² boxes


def tukey_ci_combined(metric, split, properties=None, models=None,
                      prop_cmap='winter', prop_colors=None,
                      alpha=0.05,
                      save_path=None, save_fmt='pdf'):
    """Forest plot with Tukey HSD equivalence-to-best colouring.

    For each (panel, property) the 5 inner folds are averaged within each
    repeat → 5 repeat-level means per model.  A repeated-measures one-way
    ANOVA gives MSE and residual df; pairwise Tukey p-values are then
    computed via the studentized-range distribution (same construction as
    ``notebooks/model_comparison.py::rm_tukey_hsd``).

    Colour logic (per property):
      * Best model → property colour.
      * Tukey p >= alpha vs best → grey (not significantly worse).
      * Tukey p <  alpha vs best → red (significantly worse).

    The displayed horizontal bar is the per-model 95 % CI of the mean
    (t_{0.975,4} × SEM over 5 repeat-means) — an uncertainty cue only.
    Significance comes from the Tukey pairwise test, not from CI overlap.

    The dashed vertical reference lines bracket the best model's mean by
    ± the Tukey minimum significant difference on the mean scale
    (q(α, k, df_err) · √(MSE / n)) — models whose mean lies inside that
    band are exactly the ones coloured grey.

    Layout: 2×2 grid, ~7 in wide (single journal-page column).
    """
    from scipy.stats import t as t_dist
    from matplotlib.lines import Line2D

    models = models or comp_methods
    properties = properties or ['LI', 'N', '|Mu|', '|Q|']
    use_clusters = (split == 'clusters')
    panels = CLUSTER_LABELS if use_clusters else ELEMENTS
    higher = metric in _HIGHER_IS_BETTER

    n_props = len(properties)
    n_models = len(models)

    # Colours for vertical CI reference lines / best highlight (one per property)
    if prop_colors is not None:
        vline_colors = {p: prop_colors[p] for p in properties}
    else:
        _cmap = plt.get_cmap(prop_cmap) if isinstance(prop_cmap, str) else prop_cmap
        _vline_palette = [_cmap(x) for x in np.linspace(0.15, 0.85, n_props)]
        vline_colors = dict(zip(properties, _vline_palette))

    # Significance colouring
    SIG_WORSE_COLOR = '#d62728'   # red - significantly worse
    NOT_SIG_COLOR   = '#999999'   # grey - overlaps best CI

    # Distinct markers per property (B&W legible)
    _markers = ['o', 's', 'D', '^', 'v', 'P', 'X', 'h'][:n_props]
    prop_markers = dict(zip(properties, _markers))

    # Distinct line styles for best-model CI reference lines
    _linestyles = ['-', '--', '-.', ':',
                   (0, (3, 1, 1, 1)), (0, (5, 2))][:n_props]
    prop_linestyles = dict(zip(properties, _linestyles))

    # Vertical offsets so property dots don't overlap
    offsets = np.linspace(-0.37, 0.37, n_props) if n_props > 1 else [0.0]

    # 2×2 layout fitting a single column
    n_panels = len(panels)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig_h = max(2.8, 0.42 * n_models + 0.6) * nrows + 0.5
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(7.0, fig_h),
                                  sharey=True, squeeze=False)
    axes = axes_grid.ravel()

    # Precompute every (panel, property) Tukey result once.  Same RM-ANOVA
    # logic that drives ra.tukey_pvalue_matrix and ra.latex_metric_table —
    # we just consume pc[best, model] for colours and msd_half for vlines.
    src_df    = cluster_metrics if use_clusters else elem_metrics
    panel_col = 'cluster' if use_clusters else 'element'
    src_split = None if use_clusters else split
    tukey_results = ra.tukey_pvalue_matrix(
        src_df, models=models, metric=metric, properties=properties,
        panels=panels, panel_col=panel_col, split=src_split,
        avg_over_panels=False, avg_over_props=False,
        alpha=alpha, print_tables=False,
    )

    for ax_idx in range(nrows * ncols):
        ax = axes[ax_idx]
        if ax_idx >= n_panels:
            ax.set_visible(False)
            continue
        panel = panels[ax_idx]

        # Alternating row shading
        for m_idx in range(n_models):
            if m_idx % 2 == 0:
                ax.axhspan(m_idx - 0.5, m_idx + 0.5,
                           color='#f0f0f0', zorder=0)

        # First pass: per-model 95% CI (displayed) + look up Tukey result.
        stats = {}       # (prop, model) → (mean, ci_lo, ci_hi)
        tukey_p = {}     # (prop, model) → Tukey p-value vs best (nan for best)
        best_models = {} # prop → best model name
        msd_half = {}    # prop → Tukey minimum significant difference (on mean scale)

        for p_idx, prop in enumerate(properties):
            if use_clusters:
                subdf = cluster_metrics[
                    (cluster_metrics['property'] == prop) &
                    (cluster_metrics['cluster'] == panel) &
                    (cluster_metrics['method'].isin(models))
                ]
            else:
                subdf = elem_metrics[
                    (elem_metrics['property'] == prop) &
                    (elem_metrics['split'] == split) &
                    (elem_metrics['element'] == panel) &
                    (elem_metrics['method'].isin(models))
                ]
            if subdf.empty:
                continue

            avg = (
                subdf.groupby(['method', 'repeat'])[metric]
                .mean().reset_index().dropna(subset=[metric])
            )
            cycle_stats = avg.groupby('method')[metric]
            means  = cycle_stats.mean()
            sems   = cycle_stats.sem()
            counts = cycle_stats.count()

            # Displayed 95% CI of each model's mean (uncertainty cue).
            for model in models:
                if model not in means.index:
                    continue
                mean_val = float(means[model])
                n_obs    = int(counts[model])
                sem_val  = float(sems.get(model, np.nan))
                sem_val  = sem_val if np.isfinite(sem_val) else 0.0
                t_crit   = t_dist.ppf(0.975, max(n_obs - 1, 1)) if n_obs > 1 else 0.0
                ci_half  = t_crit * sem_val
                stats[(prop, model)] = (mean_val, mean_val - ci_half, mean_val + ci_half)

            # Best model from displayed means; Tukey p-values from the precomputed matrix.
            prop_means = {m: stats[(prop, m)][0]
                          for m in models if (prop, m) in stats}
            if not prop_means:
                continue
            best = (max(prop_means, key=prop_means.get) if higher
                    else min(prop_means, key=prop_means.get))
            best_models[prop] = best

            res = tukey_results.get((panel, prop))
            if res is None:
                continue
            pc = res.pc
            if np.isfinite(res.msd_half):
                msd_half[prop] = res.msd_half
            for model in models:
                if model == best or model not in pc.index:
                    continue
                tukey_p[(prop, model)] = float(pc.loc[best, model])

        # Second pass: draw error bars with Tukey-based significance colouring.
        for p_idx, prop in enumerate(properties):
            best = best_models.get(prop)
            for m_idx, model in enumerate(models):
                if (prop, model) not in stats:
                    continue
                mean_val, ci_lo, ci_hi = stats[(prop, model)]
                y_pos = m_idx + offsets[p_idx]

                if model == best:
                    ci_color    = vline_colors[prop]
                    marker_face = vline_colors[prop]
                else:
                    p_val = tukey_p.get((prop, model), np.nan)
                    if np.isfinite(p_val) and p_val >= alpha:
                        ci_color = marker_face = NOT_SIG_COLOR
                    else:
                        ci_color = marker_face = SIG_WORSE_COLOR

                ax.errorbar(mean_val, y_pos,
                            xerr=[[mean_val - ci_lo], [ci_hi - mean_val]],
                            fmt='none',
                            ecolor=ci_color,
                            capsize=2.2, capthick=0.8, elinewidth=0.8,
                            zorder=3)
                ax.plot(mean_val, y_pos,
                        marker=prop_markers[prop], linestyle='None',
                        markersize=6.0,
                        markerfacecolor=marker_face,
                        markeredgecolor='black',
                        markeredgewidth=0.35,
                        zorder=5)

        # Vertical reference lines: best.mean ± Tukey minimum significant diff.
        for p_idx, prop in enumerate(properties):
            best = best_models.get(prop)
            msd  = msd_half.get(prop)
            if best and (prop, best) in stats and msd is not None and np.isfinite(msd):
                best_mean = stats[(prop, best)][0]
                for x in (best_mean - msd, best_mean + msd):
                    ax.axvline(x, color=vline_colors[prop],
                               linestyle=prop_linestyles[prop],
                               linewidth=1.8, alpha=0.6, zorder=1)

        ax.set_yticks(range(n_models))
        ax.set_yticklabels(models, fontsize=_FONT_SIZES['label'])
        ax.set_ylim(-0.5, n_models - 0.5)
        ax.set_title(panel, fontsize=_FONT_SIZES['title'], fontweight='bold')
        ax.tick_params(axis='x', labelsize=max(_FONT_SIZES['tick'] - 1, 1), pad=1)
        ax.tick_params(axis='y', pad=1)
        if ax_idx >= (nrows - 1) * ncols:
            if metric == 'R2':
                ax.set_xlabel('$R^2$', fontsize=_FONT_SIZES['label'])
            else:
                ax.set_xlabel(metric, fontsize=_FONT_SIZES['label'])
        ax.grid(axis='x', alpha=0.25, linewidth=0.5)

    # Legend — property shapes/colors + significance coding
    legend_handles = [
        Line2D([0], [0], marker=prop_markers[p],
               linestyle=prop_linestyles[p],
               color=vline_colors[p], markersize=7,
               markeredgecolor='black', markeredgewidth=0.35,
               linewidth=0.9, label=_pretty_prop_label(p))
        for p in properties
    ]
    legend_handles.append(
        Line2D([0], [0], marker='o', linestyle='',
               color=NOT_SIG_COLOR, markersize=5,
               markeredgecolor='black', markeredgewidth=0.35,
               label='non-sig.')
    )
    legend_handles.append(
        Line2D([0], [0], marker='o', linestyle='',
               color=SIG_WORSE_COLOR, markersize=5,
               markeredgecolor='black', markeredgewidth=0.35,
               label='sig. worse')
    )
    fig.legend(handles=legend_handles, loc='upper center',
               ncol=len(legend_handles),
               bbox_to_anchor=(0.5, 1.0), fontsize=_FONT_SIZES['legend'], frameon=True,
               edgecolor='#cccccc', fancybox=False,
               handletextpad=0.3, columnspacing=0.8, borderpad=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=0.5, w_pad=0.4)
    _save_fig(fig, save_path, save_fmt)
    plt.show()



def tukey_table_1x4(prop, metric, split, models=None):
    """ANOVA + Tukey HSD table for one property – printed per element."""
    models = models or comp_methods
    subdf = elem_metrics[
        (elem_metrics['property'] == prop) &
        (elem_metrics['split'] == split) &
        (elem_metrics['method'].isin(models))
    ]
    for elem in ELEMENTS:
        elem_data = subdf[subdf['element'] == elem]
        if elem_data.empty:
            continue
        avg = elem_data.groupby(['method', 'repeat'])[metric].mean().reset_index()
        avg = avg.dropna(subset=[metric])
        print(f"=== {elem} | {split.upper()} | {prop} | {metric} ===")
        try:
            aov = AnovaRM(avg, depvar=metric, subject='repeat', within=['method']).fit()
            print(f"  ANOVA p = {aov.anova_table['Pr > F'].iloc[0]:.4e}")
        except Exception as e:
            print(f"  ANOVA failed: {e}")
        try:
            tab, means, diff, pc = rm_tukey_hsd(avg, metric, group_col='method', sort=True)
            display(tab)
        except Exception as e:
            print(f"  Tukey failed: {e}")



def _panel_metric_stats(prop, panel, model, use_clusters, splits, metric='R2'):
    """Return mean/std of a metric for one panel/model from precomputed tables.

    Parameters
    ----------
    metric : str or None
        Metric column to summarise.  If None, returns (nan, nan).
    """
    if metric is None:
        return np.nan, np.nan
    if use_clusters:
        stats_df = cluster_metrics[
            (cluster_metrics['cluster'] == panel) &
            (cluster_metrics['method'] == model) &
            (cluster_metrics['property'] == prop)
        ]
    else:
        stats_df = elem_metrics[
            (elem_metrics['element'] == panel) &
            (elem_metrics['method'] == model) &
            (elem_metrics['property'] == prop) &
            (elem_metrics['split'].isin(splits))
        ]
    if stats_df.empty or metric not in stats_df.columns:
        return np.nan, np.nan
    vals = stats_df[metric].dropna()
    if vals.empty:
        return np.nan, np.nan
    return vals.mean(), vals.std()



def _get_property_unit(prop):
    if prop in {'N', 'LI'}:
        return 'e'
    if prop in {'Mu_X', 'Mu_Y', 'Mu_Z', '|Mu|'}:
        return 'e•Bohr'
    if prop in {'Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ', '|Q|'}:
        return 'e•Bohr^2'
    return ''



def _axis_label(base, prop):
    unit = _get_property_unit(prop)
    prop_label = _pretty_prop_label(prop)
    if unit:
        return f"{base} {prop_label} [${unit}$]"
    return f"{base} {prop_label}"


def _add_inpanel_panel_label(ax, panel: str, r2_text: str | None = None, *, xy=None):
    """In-panel label: single unified box, panel name centred above R²."""
    if xy is None:
        xy = _LABEL_XY
    _bbox = dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.90, edgecolor='#cccccc')
    if r2_text:
        ax.text(xy[0], xy[1],
                f"{panel}\n{r2_text}",
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=12,
                multialignment='center',
                linespacing=1.35,
                bbox=_bbox)
    else:
        ax.text(xy[0], xy[1], str(panel),
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=12,
                bbox=_bbox)



def _make_parity_axes():
    """Create a tight 2x2 parity grid."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    fig.subplots_adjust(hspace=0.12, wspace=0.14)
    return fig, np.array(axes).ravel()



def _set_parity_axis_labels(ax, i, prop):
    """Apply compact label strategy for 2x2 layout."""
    ax.set_ylabel(_axis_label('pred', prop) if i in (0, 2) else '')
    ax.set_xlabel(_axis_label('target', prop) if i in (2, 3) else '')



def _target_driven_bin_edges(target_values, max_bins=60):
    """Compute histogram bin edges driven by the target distribution.

    Uses Freedman-Diaconis on the targets, capped at max_bins.
    """
    try:
        edges = np.histogram_bin_edges(target_values, bins='fd')
        if len(edges) - 1 > max_bins:
            edges = np.histogram_bin_edges(target_values, bins=max_bins)
    except Exception:
        edges = np.histogram_bin_edges(target_values, bins=max_bins)
    return edges


def _add_marginal_density(ax, target_values, pred_series, pred_colors,
                          target_color="#505050", pred_alpha=0.5,
                          max_bins=60, marginal_log=False):
    """Add top/right marginal density histograms once per panel.

    target_values: 1D array of targets
    pred_series: list of 1D arrays (one per model)
    pred_colors: list of colors aligned to pred_series
    max_bins: maximum number of bins (edges are target-driven via Freedman-Diaconis)
    marginal_log: if True, use log scale on the density axis so tail bars are visible.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes('top', size='22%', pad=0.00, sharex=ax)
    ax_right = divider.append_axes('right', size='22%', pad=0.00, sharey=ax)

    # Compute target-driven bin edges (shared for both axes so scales match)
    edges = _target_driven_bin_edges(target_values, max_bins=max_bins)

    # Target density in a distinct color (top marginal)
    ax_top.hist(target_values, bins=edges, density=True, color=target_color, alpha=1.0,
                edgecolor='0.15', linewidth=0.35)

    # Predicted densities with reduced opacity (right marginal), use same edges.
    # Use model colors so different models are distinguishable.
    for pred_vals, color in zip(pred_series, pred_colors):
        ax_right.hist(pred_vals, bins=edges, density=True, orientation='horizontal',
                      color=color, alpha=pred_alpha, edgecolor='0.15', linewidth=0.35)

    # Log scale makes tail bars visible when the peak is 1000× taller
    if marginal_log:
        ax_top.set_yscale('log')
        ax_right.set_xscale('log')
        # Suppress log-scale minor ticks
        ax_top.minorticks_off()
        ax_right.minorticks_off()

    ax_top.grid(False)
    ax_right.grid(False)

    ax_top.tick_params(axis='x', labelbottom=False)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_top.set_yticks([])
    ax_right.set_xticks([])
    for sp in ['right', 'top', 'left', 'bottom']:
        ax_top.spines[sp].set_visible(False)
        ax_right.spines[sp].set_visible(False)


def _parity_scatter(ax, arr_t, arr_p, scatter_mode='scatter', color=None,
                    label=None, cmap='viridis', gridsize=80, s=6, alpha=0.5):
    """Draw a parity scatter or hexbin on the given axis.

    scatter_mode: 'scatter' (default, colored dots) or 'hexbin' (log-density).
    """
    if scatter_mode == 'hexbin':
        from matplotlib.colors import LogNorm
        hb = ax.hexbin(arr_t, arr_p, gridsize=gridsize, mincnt=1,
                        cmap=cmap, norm=LogNorm(), linewidths=0.2)
        return hb
    else:
        ax.scatter(arr_t, arr_p, s=s, alpha=alpha, label=label, color=color)
        return None


def _parity_lims(arr_t, arr_p, lim_mode='data', lim_quantile=0.998):
    """Compute axis limits.

    lim_mode:
      'data'     – min/max of targets and preds (original behaviour)
      'quantile' – use lim_quantile of targets; identity line still extends to full range
    """
    data_min = min(float(arr_t.min()), float(arr_p.min()))
    data_max = max(float(arr_t.max()), float(arr_p.max()))
    if lim_mode == 'quantile':
        lo = float(np.quantile(arr_t, 1 - lim_quantile))
        hi = float(np.quantile(arr_t, lim_quantile))
        margin = 0.05 * (hi - lo) if hi > lo else 0.1
        return lo - margin, hi + margin, data_min, data_max
    return data_min, data_max, data_min, data_max

def flatten_predictions_by_element(df, prop):
    atom_df = ra.df_to_atom_table(df)
    return {elem: (g[f'pred_{prop}'].values, g[f'target_{prop}'].values)
            for elem, g in atom_df.groupby('element')}

def flatten_predictions_by_cluster(df, prop, cluster_labels):
    atom_df = ra.df_to_atom_table(df)
    return {cl: (g[f'pred_{prop}'].values, g[f'target_{prop}'].values)
            for cl, g in atom_df.groupby('cluster') if cl in cluster_labels}


def parity_overlap_1x4(prop, split='clusters', models=None, save_path=None, save_fmt='pdf',
                       cmap='viridis', per_atom_mean=True, marginal_log=False,
                       panel_metric='R2'):
    """Parity hexbin plot for one property with all selected models overlaid (2x2).

    Parameters
    ----------
    prop : property name (e.g. 'N', '|Mu|').
    split : 'clusters' (default) or 'test'.  Both use the fixed held-out
            test set; 'clusters' filters to atoms carrying held-out cluster
            labels.  'val' / 'both' are no longer supported.
    per_atom_mean : if True (default), each atom's prediction is averaged
            over the 25 folds before binning.  Because the test set is
            fixed, every atom appears in every fold; per-atom averaging
            makes the hexbin density and the marginal histograms count
            atoms once each.  If False, every (fold \u00d7 atom) prediction is
            treated as a separate event (the old behaviour) \u2014 vertical
            spread at fixed target then reflects fold-to-fold variance.
    marginal_log : if True, the top/right marginal density axes use log
            scale (helpful when one bin dominates).
    panel_metric : 'R2' or 'CCC' \u2014 which scalar to display in the per-panel
            label box (mean \u00b1 std across folds, from elem_metrics /
            cluster_metrics).
    cmap, save_path, save_fmt, models : as before.
    """
    if split not in ('clusters', 'test'):
        raise ValueError(f"split must be 'clusters' or 'test', got {split!r}")
    if panel_metric not in ('R2', 'CCC'):
        raise ValueError(f"panel_metric must be 'R2' or 'CCC', got {panel_metric!r}")

    models = models or comp_methods
    use_clusters = (split == 'clusters')
    panels = CLUSTER_LABELS if use_clusters else ELEMENTS
    splits = ['test']  # always the fixed held-out test set
    panel_metric_label = '$R^2$' if panel_metric == 'R2' else 'CCC'

    fig, axes = _make_parity_axes()
    for i, panel in enumerate(panels):
        ax = axes[i]
        panel_min, panel_max = np.inf, -np.inf
        pred_series, pred_colors, target_series = [], [], []
        metric_lines: list[str] = []

        for model in models:
            # Load every fold's test predictions, then either average per atom
            # across folds or concatenate as fold-events.
            fold_pred_arrays: list[np.ndarray] = []
            target_arr = element_arr = cluster_arr = None
            for fold in get_folds(EXPERIMENT_DIR, model):
                path = os.path.join(EXPERIMENT_DIR, model, f"fold_{fold}", "test_preds.pkl")
                df_pred = safe_read_pickle(path)
                if df_pred.empty:
                    continue
                atom_df = ra.df_to_atom_table(df_pred)
                fold_pred_arrays.append(atom_df[f'pred_{prop}'].values)
                if target_arr is None:
                    target_arr  = atom_df[f'target_{prop}'].values
                    element_arr = atom_df['element'].values
                    cluster_arr = atom_df['cluster'].values

            if not fold_pred_arrays or target_arr is None:
                continue

            if per_atom_mean:
                # Test set is fixed across folds \u2192 every atom appears in every fold.
                preds_full  = np.stack(fold_pred_arrays, axis=0).mean(axis=0)
                target_full = target_arr
                elem_full   = element_arr
                cl_full     = cluster_arr
            else:
                n_f = len(fold_pred_arrays)
                preds_full  = np.concatenate(fold_pred_arrays)
                target_full = np.tile(target_arr,  n_f)
                elem_full   = np.tile(element_arr, n_f)
                cl_full     = np.tile(cluster_arr, n_f)

            mask = (cl_full == panel) if use_clusters else (elem_full == panel)
            arr_p = preds_full[mask]
            arr_t = target_full[mask]

            metric_mean, metric_std = _panel_metric_stats(
                prop, panel, model, use_clusters, splits, metric=panel_metric,
            )
            if np.isfinite(metric_mean):
                prefix = f"{model}: " if len(models) > 1 else ""
                metric_lines.append(
                    f"{prefix}{panel_metric_label}={metric_mean:.3f}\u00b1{metric_std:.3f}"
                )

            if arr_p.size > 0:
                panel_min = min(panel_min, float(arr_t.min()), float(arr_p.min()))
                panel_max = max(panel_max, float(arr_t.max()), float(arr_p.max()))
                target_series.append(arr_t)
                pred_series.append(arr_p)
                pred_colors.append(MODEL_COLORS[model])

        if np.isfinite(panel_min) and np.isfinite(panel_max):
            all_targs = np.concatenate(target_series) if target_series else np.array([panel_min, panel_max])
            all_preds = np.concatenate(pred_series) if pred_series else np.array([panel_min, panel_max])
            lim_lo, lim_hi, line_lo, line_hi = _parity_lims(all_targs, all_preds)
            hb = _parity_scatter(ax, all_targs, all_preds, scatter_mode='hexbin',
                                 cmap=cmap, gridsize=80)
            ax.plot([line_lo, line_hi], [line_lo, line_hi], 'k--', lw=0.8)
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)

            # Per-panel inset colorbar (bottom-right, white background)
            if hb is not None:
                import matplotlib
                # White pad drawn first so it sits behind the colorbar
                bg = ax.inset_axes([0.44, 0.06, 0.55, 0.18])
                bg.set_facecolor('white')
                bg.patch.set_alpha(0.92)
                bg.set_xticks([]); bg.set_yticks([])
                for sp in bg.spines.values():
                    sp.set_visible(False)
                # Colorbar on top of the white pad
                cax = ax.inset_axes([0.50, 0.10, 0.46, 0.045])
                cb = plt.colorbar(hb, cax=cax, orientation='horizontal')
                if isinstance(getattr(hb, 'norm', None), matplotlib.colors.LogNorm):
                    cb.ax.xaxis.set_major_locator(LogLocator(base=10))
                    cb.ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
                cb.ax.tick_params(labelsize=10, length=2, pad=3)
                cb.outline.set_linewidth(0.4)
                # "Count" as a title above the colorbar (inside the white pad)
                cax.set_title('Count', fontsize=10, pad=4)

        if pred_series:
            all_targets = np.concatenate(target_series)
            _add_marginal_density(ax, target_values=all_targets,
                                  pred_series=pred_series, pred_colors=pred_colors,
                                  marginal_log=marginal_log)

        _set_parity_axis_labels(ax, i, prop)
        ax.tick_params(axis='both', labelsize=11)
        ax.set_title('')
        metric_text = "\n".join(metric_lines) if metric_lines else None
        _add_inpanel_panel_label(ax, panel, r2_text=metric_text)

    _save_fig(fig, save_path, save_fmt)
    plt.show()


def cluster_boxplot_1x4(prop, metric='MAE', models=None, save_path=None, save_fmt='pdf'):
    """Cluster box plot for one property - across 4 cluster labels."""
    models = models or comp_methods
    subdf = cluster_metrics[
        (cluster_metrics['property'] == prop) &
        (cluster_metrics['method'].isin(models))
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    for i, cl in enumerate(CLUSTER_LABELS):
        ax = axes[i]
        cl_data = subdf[subdf['cluster'] == cl]
        if cl_data.empty:
            ax.set_title(f"{cl} (no data)")
            continue
        sns.boxplot(
            data=cl_data, x='method', y=metric,
            order=models,
            palette=[MODEL_COLORS[m] for m in models], ax=ax,
        )
        ax.set_title(cl)
        ax.set_xlabel('')
        ax.set_ylabel(metric if i == 0 else '')
    plt.suptitle(f"Clusters - {prop} - {metric}")
    plt.tight_layout()
    _save_fig(fig, save_path, save_fmt)
    plt.show()


def cluster_tukey_table_1x4(prop, metric='MAE', models=None):
    """ANOVA + Tukey HSD table for cluster data – printed per cluster label."""
    models = models or comp_methods
    subdf = cluster_metrics[
        (cluster_metrics['property'] == prop) &
        (cluster_metrics['method'].isin(models))
    ]
    for cl in CLUSTER_LABELS:
        cl_data = subdf[subdf['cluster'] == cl]
        if cl_data.empty:
            continue
        avg = cl_data.groupby(['method', 'repeat'])[metric].mean().reset_index()
        avg = avg.dropna(subset=[metric])
        print(f"=== Cluster {cl} | {prop} | {metric} ===")
        try:
            aov = AnovaRM(avg, depvar=metric, subject='repeat', within=['method']).fit()
            print(f"  ANOVA p = {aov.anova_table['Pr > F'].iloc[0]:.4e}")
        except Exception as e:
            print(f"  ANOVA failed: {e}")
        try:
            tab, means, diff, pc = rm_tukey_hsd(avg, metric, group_col='method', sort=True)
            display(tab)
        except Exception as e:
            print(f"  Tukey failed: {e}")


def cluster_tukey_ci_1x4(prop, metric='MAE', models=None, save_path=None, save_fmt='pdf'):
    """Tukey HSD simultaneous CI for cluster data – across 4 cluster labels."""
    models = models or comp_methods
    subdf = cluster_metrics[
        (cluster_metrics['property'] == prop) &
        (cluster_metrics['method'].isin(models))
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for i, cl in enumerate(CLUSTER_LABELS):
        ax = axes[i]
        cl_data = subdf[subdf['cluster'] == cl]
        if cl_data.empty:
            ax.set_title(f"{cl} (no data)"); continue
        avg = cl_data.groupby(['method', 'repeat'])[metric].mean().reset_index()
        avg = avg.dropna(subset=[metric])
        if avg.empty or avg['method'].nunique() < 2:
            ax.set_title(f"{cl} (insufficient)"); continue
        try:
            tukey = pairwise_tukeyhsd(endog=avg[metric], groups=avg['method'], alpha=0.05)
            best = (avg.groupby('method')[metric].mean().idxmax() if metric in _HIGHER_IS_BETTER
                    else avg.groupby('method')[metric].mean().idxmin())
            tukey.plot_simultaneous(comparison_name=best, ax=ax)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes, ha='center')
        ax.set_title(cl)
    plt.suptitle(f"Tukey CI – Clusters – {prop} – {metric}")
    plt.tight_layout()
    _save_fig(fig, save_path, save_fmt)
    plt.show()


def summary_table_1x4(metric='MAE', split='test', models=None):
    """Mean ± std summary per model × element (or cluster) × property.
    split: 'val', 'test', or 'clusters'."""
    models = models or comp_methods
    if split == 'clusters':
        src = cluster_metrics[cluster_metrics['method'].isin(models)]
        rows = []
        for cl in CLUSTER_LABELS:
            for model in models:
                for prop in PROPERTY_NAMES:
                    vals = src[(src['cluster'] == cl) & (src['method'] == model) & (src['property'] == prop)][metric].dropna()
                    if not vals.empty:
                        rows.append({
                            'cluster': cl, 'model': model, 'property': prop,
                            f'{metric}': f"{vals.mean():.4f} ± {vals.std():.4f}",
                        })
        df = pd.DataFrame(rows)
        return df.pivot_table(index=['cluster', 'model'], columns='property',
                              values=metric, aggfunc='first')[PROPERTY_NAMES]
    subdf = elem_metrics[(elem_metrics['split'] == split) & (elem_metrics['method'].isin(models))]
    rows = []
    for elem in ELEMENTS:
        for model in models:
            for prop in PROPERTY_NAMES:
                vals = subdf[(subdf['element'] == elem) & (subdf['method'] == model) & (subdf['property'] == prop)][metric].dropna()
                if not vals.empty:
                    rows.append({
                        'element': elem, 'model': model, 'property': prop,
                        f'{metric}': f"{vals.mean():.4f} ± {vals.std():.4f}",
                    })
    df = pd.DataFrame(rows)
    return df.pivot_table(index=['element', 'model'], columns='property',
                          values=metric, aggfunc='first')[PROPERTY_NAMES]



def plot_family(family, models=None, metric='MAE', split='test', properties=None,
                save_dir=None, save_fmt='pdf',
                cmap='viridis', prop_cmap='winter', prop_colors=None,
                per_atom_mean=True, marginal_log=False, panel_metric='R2'):
    """Generate one-row-per-property plots from the existing helpers.

    Parameters
    ----------
    family : {'box','tukey','tukey_combined','parity'}
        Which type of plot to draw.
    models : list, optional
        List of model names to include; defaults to comp_methods.
    metric : str, optional
        Error metric used for box/tukey families (ignored for parity).
    split : {'val','test','both','clusters'}, optional
        Dataset split to visualise.
    properties : list, optional
        Properties to loop over. Defaults to ['N'].
    save_dir : str or None, optional
        Directory to save figures into. None disables saving.
    save_fmt : str, optional
        File format for saved figures (default 'pdf').
    cmap : str, optional
        Colormap for parity hexbin mode.
    prop_cmap : str, optional
        Colormap for property colours in tukey_combined.
    prop_colors : dict, optional
        Explicit property colour mapping for tukey_combined.
    """
    models = models or comp_methods
    props = properties or ['N']

    # tukey_combined produces a single figure for all properties at once
    if family == 'tukey_combined':
        sp = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            sp = os.path.join(save_dir, f"tukey_combined_{metric}_{split}.{save_fmt}")
        tukey_ci_combined(metric, split, properties=props, models=models,
                  prop_cmap=prop_cmap, prop_colors=prop_colors,
                  save_path=sp, save_fmt=save_fmt)
        return

    for prop in props:
        # Build save path if saving is requested
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            safe_prop = prop.replace('|', '').replace(' ', '_')
            if family == 'parity':
                tag = f"{family}_{safe_prop}_{split}"
            else:
                tag = f"{family}_{safe_prop}_{metric}_{split}"
            sp = os.path.join(save_dir, f"{tag}.{save_fmt}")
        else:
            sp = None

        if family == 'box':
            if split == 'clusters':
                cluster_boxplot_1x4(prop, metric, models, save_path=sp, save_fmt=save_fmt)
            else:
                boxplot_1x4(prop, metric, split, models, save_path=sp, save_fmt=save_fmt)
        elif family == 'tukey':
            if split == 'clusters':
                cluster_tukey_ci_1x4(prop, metric, models, save_path=sp, save_fmt=save_fmt)
            else:
                tukey_ci_1x4(prop, metric, split, models, save_path=sp, save_fmt=save_fmt)
        elif family == 'parity':
            parity_overlap_1x4(prop, split=split, models=models,
                               save_path=sp, save_fmt=save_fmt, cmap=cmap,
                               per_atom_mean=per_atom_mean,
                               marginal_log=marginal_log,
                               panel_metric=panel_metric)
        else:
            raise ValueError(f"Unknown plot family {family}")


if __name__ == "__main__":
    print("plot functions defined")
