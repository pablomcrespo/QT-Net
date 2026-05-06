"""Analysis utilities for QM9 inference results.

Two analyses are supported:

1. **Molecular property prediction** — compares blind vs informed model
   variants across training fractions on the four QM9 targets (alpha, gap,
   U0, Cv).  Results live at::

       experiments/inference/molecular/{variant}/frac_{f}/qm9_molecular_preds.pkl

   Each pkl has ONE ensemble-averaged prediction per molecule (5 best-per-repeat
   folds already collapsed inside ``predict_from_inferred.py``).  Uncertainty
   is quantified with bootstrap CIs over molecules.

2. **Molecular dipole reconstruction** — compares ``mu_inferred`` (sum of
   per-atom QTAIM Mu vectors) against QM9's reference ``mu`` column.  The
   dipole formula is ``mu_mol = ||sum_i Mu_i||`` (AIMAll convention; Mu already
   includes the bond/charge-transfer term — see ``project_aimel_mu_convention.md``).

Typical notebook usage::

    import importlib
    import notebooks.inference.analyse_inference as ai
    importlib.reload(ai)

    df = ai.load_from_experiment_dir(EXP_DIR)
    metric_df = ai.metric_table_inference(df, n_boot=500)
    ai.learning_curve_inference(metric_df, metric='CCC')
    ai.parity_inference(df, fraction=1.0)

    df_mu = ai.load_dipole_preds(DIPOLE_PKL)
    ai.dipole_parity(df_mu)
"""
from __future__ import annotations

import glob
import os
import sys
import warnings
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Match the visual style of notebooks/atomic/tmp_plot_cell_updated.py
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'font.size':       12,
    'axes.titlesize':  12,
    'axes.labelsize':  12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# Fixed tab10 palette (avoids Pylance stub issues with matplotlib's cmap registry)
_TAB10_HEX = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import notebooks.atomic.result_analysis as ra  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROPERTIES:   list[str]   = ['alpha', 'gap', 'U0', 'Cv']
VARIANTS:     list[str]   = ['blind', 'informed']
FRACTIONS:    list[float] = [0.01, 0.05, 0.1, 1.0]
METRIC_NAMES: list[str]   = ['MAE', 'RMSE', 'R2', 'CCC', 'Spearman']

DIPOLE_COL:  str = 'mu'           # QM9 ground-truth dipole column (Debye)
DIPOLE_PRED: str = 'mu_inferred'  # column added by molecule_dipoles_from_inferred.py

_PROP_LABELS: dict[str, str] = {
    'alpha': r'$\alpha$',
    'gap':   r'$\Delta\varepsilon$',
    'U0':    r'$U_0$',
    'Cv':    r'$C_v$',
}
_PROP_UNITS: dict[str, str] = {
    'alpha': r'$a_0^3$',
    'gap':   'Ha',
    'U0':    'Ha',
    'Cv':    'cal/(mol·K)',
}

_VARIANT_LABELS: dict[str, str] = {
    'blind':   'Blind',
    'informed': 'Informed',
}

VARIANT_COLORS: dict[str, str] = {
    'blind':    '#1f77b4',
    'informed': '#d62728',
}
VARIANT_LINESTYLES: dict[str, str] = {
    'blind':    '--',
    'informed': '-',
}

_PROP_MARKERS: list[str]      = ['o', 's', 'D', '^']
_PROP_LINESTYLES: list[str]   = ['-', '--', '-.', ':']

_FONT = {'title': 18, 'label': 16, 'tick': 12, 'legend': 12, 'suptitle': 12, 'annot': 12}

_METRIC_LOWER_IS_BETTER: dict[str, bool] = {
    'MAE': True, 'RMSE': True, 'R2': False, 'CCC': False, 'Spearman': False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_from_experiment_dir(
    exp_dir: str,
    variants: Iterable[str] = VARIANTS,
    fractions: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """Discover and merge all ``qm9_molecular_preds.pkl`` files under *exp_dir*.

    Expected layout::
        {exp_dir}/{variant}/frac_{f}/qm9_molecular_preds.pkl

    Returns a single DataFrame indexed by molecule (first pkl's index), with
    all ``{variant}_{fraction}_pred_{prop}``, ``{variant}_{fraction}_std_{prop}``,
    and ``target_{prop}`` columns merged together.  Missing (variant, fraction)
    combinations are skipped with a warning.
    """
    frames = []
    for variant in variants:
        pattern = os.path.join(exp_dir, variant, 'frac_*', 'qm9_molecular_preds.pkl')
        found = sorted(glob.glob(pattern))
        if not found:
            warnings.warn(f"No pkls found for variant '{variant}' under {exp_dir}")
            continue
        for pkl_path in found:
            frac_tag = os.path.basename(os.path.dirname(pkl_path))  # 'frac_X'
            try:
                frac = float(frac_tag.split('_', 1)[1])
            except (IndexError, ValueError):
                warnings.warn(f"Cannot parse fraction from {frac_tag}, skipping.")
                continue
            if fractions is not None and frac not in list(fractions):
                continue
            try:
                df = pd.read_pickle(pkl_path)
            except Exception as exc:
                warnings.warn(f"Cannot read {pkl_path}: {exc}")
                continue
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No prediction pkls found under {exp_dir}")

    # Merge on index: start with the first frame, join the new pred/std columns from each.
    base = frames[0]
    target_cols = [c for c in base.columns if c.startswith('target_')]
    metadata_cols = [c for c in base.columns
                     if not (c.startswith('target_') or '_pred_' in c or '_std_' in c)]
    out = base.copy()
    for df in frames[1:]:
        new_pred = [c for c in df.columns if '_pred_' in c or '_std_' in c]
        new_tgt  = [c for c in df.columns if c.startswith('target_') and c not in out.columns]
        out = out.join(df[new_pred + new_tgt], how='outer', rsuffix='_dup')
    # Drop duplicated target columns if they got renamed
    out = out.loc[:, ~out.columns.str.endswith('_dup')]
    return out


def discover_fractions(exp_dir: str, variant: str = 'blind') -> list[float]:
    """Return sorted list of fractions found on disk for one variant."""
    dirs = glob.glob(os.path.join(exp_dir, variant, 'frac_*'))
    fracs = []
    for d in dirs:
        tag = os.path.basename(d)
        try:
            fracs.append(float(tag.split('_', 1)[1]))
        except (IndexError, ValueError):
            pass
    return sorted(set(fracs))


def load_dipole_preds(path: str) -> pd.DataFrame:
    """Load a pkl with ``mu`` (reference) and ``mu_inferred`` (inferred) columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    full = pd.read_pickle(path)
    if DIPOLE_COL not in full.columns:
        raise KeyError(f"Column '{DIPOLE_COL}' not in {path}")
    pred_cols = [c for c in full.columns
                 if c == DIPOLE_PRED or c.startswith(f'{DIPOLE_PRED}_')]
    if not pred_cols:
        raise KeyError(
            f"No inferred-dipole column found. "
            "Run scripts/inference/molecule_dipoles_from_inferred.py first."
        )
    keep = [DIPOLE_COL] + pred_cols
    if 'n_atoms' in full.columns:
        keep.append('n_atoms')
    if 'smiles' in full.columns:
        keep.append('smiles')
    out = full[keep].copy()
    if 'n_atoms' not in out.columns:
        for cand in ('elements', 'atom', 'element'):
            if cand in full.columns:
                out['n_atoms'] = full[cand].apply(len)
                break
    return out.dropna(subset=[DIPOLE_COL] + pred_cols)


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE, R2, CCC, Spearman for one (target, pred) pair."""
    return ra._compute_metrics_dict(np.asarray(y_true, dtype=float),
                                    np.asarray(y_pred, dtype=float))


_METRIC_FUNCS: dict[str, Callable] = {
    name: (lambda t, p, _n=name: compute_metrics(t, p)[_n])
    for name in METRIC_NAMES
}


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Union[str, Callable] = 'MAE',
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = 0,
) -> tuple[float, float, float]:
    """Bootstrap (point, lo, hi) for a metric by resampling molecule indices."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    fn = metric if callable(metric) else _METRIC_FUNCS[metric]
    point = float(fn(y_true, y_pred))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot[b] = fn(y_true[idx], y_pred[idx])
        except Exception:
            boot[b] = np.nan
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return point, np.nan, np.nan
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return point, float(lo), float(hi)


def metric_table_inference(
    df: pd.DataFrame,
    metrics: Iterable[str] = ('CCC', 'R2', 'Spearman', 'MAE'),
    properties: Iterable[str] = PROPERTIES,
    variants: Iterable[str] = VARIANTS,
    fractions: Optional[Iterable[float]] = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Long-format bootstrap-CI metric table.

    Returns columns: [variant, fraction, property, metric, point, lo, hi, n].
    Auto-detects available (variant, fraction) from the df columns when
    *fractions* is None.
    """
    # discover fractions from columns
    if fractions is None:
        fracs_set: set[float] = set()
        for col in df.columns:
            if '_pred_' in col:
                parts = col.split('_pred_')[0].split('_')
                try:
                    fracs_set.add(float(parts[-1]))
                except ValueError:
                    pass
        fracs = sorted(fracs_set)
    else:
        fracs = sorted(fractions)

    rows = []
    for variant in variants:
        for frac in fracs:
            for prop in properties:
                pred_col   = f'{variant}_{frac}_pred_{prop}'
                target_col = f'target_{prop}'
                if pred_col not in df.columns or target_col not in df.columns:
                    continue
                pair = df[[target_col, pred_col]].dropna()
                if len(pair) < 2:
                    continue
                t = pair[target_col].to_numpy(dtype=float)
                p = pair[pred_col].to_numpy(dtype=float)
                for metric in metrics:
                    pt, lo, hi = bootstrap_metric_ci(t, p, metric=metric,
                                                     n_boot=n_boot, ci=ci, seed=seed)
                    rows.append({'variant': variant, 'fraction': frac,
                                 'property': prop, 'metric': metric,
                                 'point': pt, 'lo': lo, 'hi': hi, 'n': len(pair)})
    return pd.DataFrame(rows)


def format_metric_table(
    metric_df: pd.DataFrame,
    metric: str = 'CCC',
    n_decimals: int = 3,
) -> pd.DataFrame:
    """Pivot long metric_df to (variant × fraction) × property grid."""
    sub = metric_df[metric_df['metric'] == metric].copy()
    if sub.empty:
        return pd.DataFrame()
    sub['half'] = (sub['hi'] - sub['lo']) / 2.0
    sub['cell'] = sub.apply(
        lambda r: f"{r['point']:.{n_decimals}f} ±{r['half']:.{n_decimals}f}", axis=1
    )
    pivot = sub.pivot_table(
        index=['variant', 'fraction'], columns='property',
        values='cell', aggfunc='first',
    )
    return pivot.reindex(columns=PROPERTIES)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — molecular property analysis
# ─────────────────────────────────────────────────────────────────────────────

def learning_curve_inference(
    metric_df: pd.DataFrame,
    metric: str = 'CCC',
    properties: Iterable[str] = PROPERTIES,
    variants: Iterable[str] = VARIANTS,
    avg_over_props: bool = False,   # deprecated — has no effect
    layout: str = '2x2',            # '2x2' or '1x4'
    err: str = 'shade',             # 'shade' (fill_between) or 'bars' (errorbar)
    save: Optional[str] = None,
    ax_size: tuple[float, float] = (3.4, 2.6),
):
    """Learning curves: metric vs training fraction.

    Parameters
    ----------
    layout : {'2x2', '1x4'}
        '2x2' (default): 2×2 grid, one panel per property; y-label left col only,
        x-label bottom row only.  '1x4': single row of four panels.
    err : {'shade', 'bars'}
        CI display style.  'shade' fills between lo/hi; 'bars' draws error bars
        with caps (capsize=3).
    avg_over_props : bool
        Deprecated — ignored.  Use ``layout`` instead.
    """
    if avg_over_props:
        warnings.warn(
            "avg_over_props is deprecated and has no effect; "
            "use layout='1x4' for a single-row layout.",
            DeprecationWarning, stacklevel=2,
        )
    properties = list(properties)
    variants   = list(variants)

    def _draw_variant(ax, sub, variant, err_mode):
        x     = sub['fraction'].to_numpy(float)
        y     = sub['point'].to_numpy(float)
        lo    = sub['lo'].to_numpy(float)
        hi    = sub['hi'].to_numpy(float)
        color = VARIANT_COLORS.get(variant, 'k')
        ls    = VARIANT_LINESTYLES.get(variant, '-')
        if err_mode == 'bars':
            ax.errorbar(x, y, yerr=[y - lo, hi - y],
                        fmt=ls, color=color, label=_VARIANT_LABELS.get(variant,variant),
                        marker='o', markersize=4, linewidth=1.5,
                        capsize=3, elinewidth=0.8)
        else:
            ax.fill_between(x, lo, hi, color=color, alpha=0.20, linewidth=0)
            ax.plot(x, y, ls, color=color, label=_VARIANT_LABELS.get(variant,variant),
                    marker='o', markersize=4, linewidth=1.5)

    def _style_ax(ax, prop, metric_name, is_bottom, is_left):
        unit = _PROP_UNITS.get(prop, '')
        y_label = metric_name + (f' ({unit})' if unit and metric_name in ('MAE', 'RMSE') else '')
        if metric_name == 'R2':
            y_label = 'QM9 $R^2$'
        ax.set_title(_PROP_LABELS.get(prop, prop), fontsize=_FONT['title'])
        ax.set_xscale('log')
        _set_fraction_ticks(ax, metric_df)
        ax.tick_params(labelsize=_FONT['tick'])
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if is_bottom:
            ax.set_xlabel('AIMEl fraction', fontsize=_FONT['label'])
        if is_left:
            ax.set_ylabel(y_label, fontsize=_FONT['label'])

    if layout == '1x4':
        n_props = len(properties)
        fig, axes = plt.subplots(1, n_props,
                                 figsize=(n_props * ax_size[0], ax_size[1]),
                                 squeeze=False)
        for k, prop in enumerate(properties):
            ax = axes[0][k]
            for variant in variants:
                sub = metric_df[
                    (metric_df['property'] == prop)
                    & (metric_df['variant']  == variant)
                    & (metric_df['metric']   == metric)
                ].sort_values('fraction')
                if not sub.empty:
                    _draw_variant(ax, sub, variant, err)
            _style_ax(ax, prop, metric, is_bottom=True, is_left=(k == 0))
            if k == 0:
                ax.legend(fontsize=_FONT['legend'], frameon=False, loc='lower right')
        fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)

    else:  # '2x2'
        fig, axes = plt.subplots(2, 2,
                                 figsize=(2 * ax_size[0], 2 * ax_size[1]),
                                 sharex=False, sharey=False, squeeze=False)
        for k, prop in enumerate(properties):
            ax = axes[k // 2][k % 2]
            for variant in variants:
                sub = metric_df[
                    (metric_df['property'] == prop)
                    & (metric_df['variant']  == variant)
                    & (metric_df['metric']   == metric)
                ].sort_values('fraction')
                if not sub.empty:
                    _draw_variant(ax, sub, variant, err)
            _style_ax(ax, prop, metric,
                      is_bottom=(k // 2 == 1), is_left=(k % 2 == 0))
            if k == 0:
                ax.legend(fontsize=_FONT['legend'], frameon=False, loc='lower right')
        fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)

    if save:
        fig.savefig(save, dpi=200, bbox_inches='tight')
    return fig


def _set_fraction_ticks(ax, metric_df: pd.DataFrame):
    """Set x-axis ticks from the fractions actually present in metric_df."""
    fracs = sorted(metric_df['fraction'].dropna().unique())
    ax.set_xticks(fracs)
    ax.set_xticklabels([f'{f:g}' for f in fracs])
    ax.xaxis.set_minor_formatter(NullFormatter())


def parity_inference(
    df: pd.DataFrame,
    fraction: float = 1.0,
    properties: Iterable[str] = PROPERTIES,
    variants: Iterable[str] = VARIANTS,
    metric_df: Optional[pd.DataFrame] = None,
    layout: str = '2x4',           # '2x4' (hexbin rows×cols) or '1x4' (scatter overlay)
    alpha_scatter: float = 0.6,    # alpha of overlapping right-marginal histograms (1x4)
    show_panel: bool = False,       # show in-panel annotation box with metric values
    save: Optional[str] = None,
    ax_size: tuple[float, float] = (3.0, 3.0),
    bins: int = 70,
    marginal_log: bool = True,
    cmap: str = 'magma',
    panel_metric: str = 'CCC',
):
    """Parity plots for molecular property predictions.

    Parameters
    ----------
    layout : {'2x4', '1x4'}
        '2x4': rows = variants, cols = properties, log-density hexbin with
        inset colorbar.  '1x4': one row per property with both variants
        overlaid as scatter; right-marginal distributions overlap with
        ``alpha_scatter``; annotation shows CIs for both variants; legend
        replaces colorbar.
    alpha_scatter : float
        Transparency of overlapping right-marginal histograms in '1x4' mode.
    """
    properties = list(properties)
    variants   = list(variants)
    metric_label = '$R^2$' if panel_metric == 'R2' else panel_metric

    if layout == '1x4':
        n_cols = len(properties)
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(n_cols * ax_size[0], ax_size[1]),
                                 squeeze=False)
        for j, prop in enumerate(properties):
            ax = axes[0][j]
            target_col = f'target_{prop}'
            if target_col not in df.columns:
                ax.set_axis_off(); continue

            pred_list, color_list, valid_variants, t_arr = [], [], [], None
            for variant in variants:
                pred_col = f'{variant}_{fraction}_pred_{prop}'
                if pred_col not in df.columns:
                    continue
                pair = df[[target_col, pred_col]].dropna()
                if len(pair) < 2:
                    continue
                t_v = pair[target_col].to_numpy(float)
                p_v = pair[pred_col].to_numpy(float)
                if t_arr is None:
                    t_arr = t_v
                pred_list.append(p_v)
                color_list.append(VARIANT_COLORS.get(variant, '#888'))
                valid_variants.append(variant)

            if t_arr is None:
                ax.set_axis_off(); continue

            all_vals = np.concatenate([t_arr] + pred_list)
            dmin, dmax = all_vals.min(), all_vals.max()
            pad = 0.04 * (dmax - dmin)
            lim = (dmin - pad, dmax + pad)

            scatter_handles = []
            for p_v, color, variant in zip(pred_list, color_list, valid_variants):
                ax.scatter(t_arr, p_v, s=4, alpha=0.15, color=color, rasterized=True)
                scatter_handles.append(
                    Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=color, markersize=7, label=_VARIANT_LABELS.get(variant,variant))
                )

            ax.plot(lim, lim, 'k--', linewidth=0.8, alpha=0.5)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.25, linewidth=0.4, zorder=0)
            ax.set_axisbelow(True)

            _add_marginal_multi(ax, t_arr, pred_list, color_list,
                                alpha=alpha_scatter, marginal_log=marginal_log)
            ax.tick_params(axis='both', which='both', labelsize=_FONT['tick'],
                           direction='out', length=3, pad=2,
                           labelbottom=True, labelleft=True)

            if show_panel:
                ann_parts = [_PROP_LABELS.get(prop, prop)]
                for variant, p_v in zip(valid_variants, pred_list):
                    ann = _format_panel_metric(metric_df, variant, fraction, prop,
                                               panel_metric, t_arr, p_v)
                    ann_parts.append(f"{variant}: {metric_label}={ann}")
                ax.text(0.04, 0.96, '\n'.join(ann_parts),
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=_FONT['annot'],
                        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', alpha=0.92))

            if j == 0:
                ax.legend(handles=scatter_handles, fontsize=_FONT['legend'],
                          frameon=True, edgecolor='#ccc', loc='upper left',
                          handlelength=1, handletextpad=0.5)

            ax.set_xlabel(f"Target {_PROP_LABELS.get(prop, prop)} [{_PROP_UNITS.get(prop, prop)}]", fontsize=_FONT['label'])
            if j == 0:
                ax.set_ylabel('Predictions', fontsize=_FONT['label'])

        fig.tight_layout(pad=0., h_pad=0., w_pad=0.)

    else:  # '2x4' — hexbin, rows = variants, cols = properties
        n_rows, n_cols = len(variants), len(properties)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * ax_size[0], n_rows * ax_size[1]),
            squeeze=False,
        )

        for i, variant in enumerate(variants):
            for j, prop in enumerate(properties):
                ax = axes[i][j]
                pred_col   = f'{variant}_{fraction}_pred_{prop}'
                target_col = f'target_{prop}'
                if pred_col not in df.columns or target_col not in df.columns:
                    ax.set_axis_off(); continue

                pair = df[[target_col, pred_col]].dropna()
                if len(pair) < 2:
                    ax.set_axis_off(); continue

                t = pair[target_col].to_numpy(float)
                p = pair[pred_col].to_numpy(float)

                dmin = min(t.min(), p.min())
                dmax = max(t.max(), p.max())
                pad  = 0.04 * (dmax - dmin)
                lim  = (dmin - pad, dmax + pad)

                hb = ax.hexbin(t, p, gridsize=bins, cmap=cmap,
                               norm=LogNorm(), mincnt=1,
                               extent=(*lim, *lim), linewidths=0.2)
                ax.plot(lim, lim, 'k--', linewidth=0.8, alpha=0.5)
                ax.set_xlim(lim); ax.set_ylim(lim)
                ax.set_aspect('equal', adjustable='box')

                ax.grid(True, alpha=0.25, linewidth=0.4, zorder=0)
                ax.set_axisbelow(True)

                _add_marginal(ax, t, p, marginal_log=marginal_log,
                              color=VARIANT_COLORS.get(variant, '#888'))
                ax.tick_params(axis='both', which='both', labelsize=_FONT['tick'],
                               direction='out', length=3, pad=2,
                               labelbottom=True, labelleft=True)

                # Inset colorbar (bottom-right)
                bg  = ax.inset_axes([0.44, 0.06, 0.55, 0.18])
                bg.set_facecolor('white'); bg.patch.set_alpha(0.92)
                bg.set_xticks([]); bg.set_yticks([])
                for sp in bg.spines.values():
                    sp.set_visible(False)
                cax = ax.inset_axes([0.50, 0.10, 0.46, 0.045])
                cb  = plt.colorbar(hb, cax=cax, orientation='horizontal')
                cb.ax.xaxis.set_major_locator(LogLocator(base=10))
                cb.ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
                cb.ax.tick_params(labelsize=8, length=2, pad=3)
                cb.outline.set_linewidth(0.4)
                cax.set_title('Count', fontsize=9, pad=4)

                if show_panel:
                    ann = _format_panel_metric(metric_df, variant, fraction, prop,
                                               panel_metric, t, p)
                    ax.text(
                        0.04, 0.96,
                        f"{_PROP_LABELS.get(prop, prop)}\n{metric_label}={ann}",
                        transform=ax.transAxes,
                        ha='left', va='top', fontsize=_FONT['annot'],
                        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', alpha=0.92),
                    )

                if i == n_rows - 1:
                    ax.set_xlabel(f"target {_PROP_LABELS.get(prop, prop)}",
                                  fontsize=_FONT['label'])
                if j == 0:
                    ax.set_ylabel(f'{variant}\npredicted', fontsize=_FONT['label'])

        fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.4)

    if save:
        fig.savefig(save, dpi=200, bbox_inches='tight')
    return fig


def _format_panel_metric(
    metric_df: Optional[pd.DataFrame],
    variant: str,
    fraction: float,
    prop: str,
    panel_metric: str,
    t: np.ndarray,
    p: np.ndarray,
) -> str:
    """Return ``'value ±half'`` from metric_df row, or fallback to point estimate."""
    if metric_df is not None:
        row = metric_df[
            (metric_df['variant']  == variant)
            & (metric_df['fraction'] == fraction)
            & (metric_df['property'] == prop)
            & (metric_df['metric']   == panel_metric)
        ]
        if not row.empty:
            r = row.iloc[0]
            half = (r['hi'] - r['lo']) / 2.0
            return f"{r['point']:.3f}±{half:.3f}"
    return f"{compute_metrics(t, p)[panel_metric]:.3f}"


def _add_marginal(ax, t: np.ndarray, p: np.ndarray,
                  color: str = '#888', marginal_log: bool = False,
                  max_bins: int = 60):
    """Top/right marginal density histograms (style from tmp_plot_cell_updated.py)."""
    divider = make_axes_locatable(ax)
    ax_top   = divider.append_axes('top',   size='22%', pad=0.00, sharex=ax)
    ax_right = divider.append_axes('right', size='22%', pad=0.00, sharey=ax)

    try:
        edges = np.histogram_bin_edges(t, bins='fd')
        if len(edges) - 1 > max_bins:
            edges = np.histogram_bin_edges(t, bins=max_bins)
    except Exception:
        edges = np.histogram_bin_edges(t, bins=max_bins)

    ax_top.hist(t, bins=edges, density=True, color='#505050', alpha=1.0,
                edgecolor='0.15', linewidth=0.35)
    ax_right.hist(p, bins=edges, density=True, orientation='horizontal',
                  color=color, alpha=0.55, edgecolor='0.15', linewidth=0.35)

    if marginal_log:
        ax_top.set_yscale('log')
        ax_right.set_xscale('log')
        ax_top.minorticks_off()
        ax_right.minorticks_off()

    for a in (ax_top, ax_right):
        a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(False)
    ax_top.set_yticks([])    # y is not shared with main ax — safe to clear
    ax_right.set_xticks([])  # x is not shared with main ax — safe to clear
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_right.tick_params(axis='y', labelleft=False)


def _add_marginal_multi(
    ax,
    t: np.ndarray,
    pred_list: list,
    colors: list,
    alpha: float = 0.6,
    marginal_log: bool = False,
    max_bins: int = 60,
):
    """Top/right marginals for overlaid variants (1x4 parity layout).

    Top shows the shared target distribution; right overlaps one histogram per
    variant with per-variant color and the given alpha.
    """
    divider = make_axes_locatable(ax)
    ax_top   = divider.append_axes('top',   size='22%', pad=0.00, sharex=ax)
    ax_right = divider.append_axes('right', size='22%', pad=0.00, sharey=ax)

    try:
        edges = np.histogram_bin_edges(t, bins='fd')
        if len(edges) - 1 > max_bins:
            edges = np.histogram_bin_edges(t, bins=max_bins)
    except Exception:
        edges = np.histogram_bin_edges(t, bins=max_bins)

    ax_top.hist(t, bins=edges, density=True, color='#505050', alpha=1.0,
                edgecolor='0.15', linewidth=0.35)
    for p_arr, color in zip(pred_list, colors):
        ax_right.hist(p_arr, bins=edges, density=True, orientation='horizontal',
                      color=color, alpha=alpha, edgecolor='none', linewidth=0)

    if marginal_log:
        ax_top.set_yscale('log')
        ax_right.set_xscale('log')
        ax_top.minorticks_off()
        ax_right.minorticks_off()

    for a in (ax_top, ax_right):
        a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(False)
    ax_top.set_yticks([])    # y is not shared with main ax — safe to clear
    ax_right.set_xticks([])  # x is not shared with main ax — safe to clear
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_right.tick_params(axis='y', labelleft=False)


def latex_metric_table_inference(
    metric_df: pd.DataFrame,
    metric: str = 'CCC',
    properties: Iterable[str] = PROPERTIES,
    variants: Iterable[str] = VARIANTS,
    n_decimals: int = 3,
    highlight_best: bool = True,
    caption: Optional[str] = None,
    label: str = 'tab:inference',
) -> str:
    """LaTeX table: (variant × fraction) rows × property cols, cells = point ± half-CI."""
    sub = metric_df[metric_df['metric'] == metric].copy()
    if sub.empty:
        return ''
    sub['half'] = (sub['hi'] - sub['lo']) / 2.0
    fracs = sorted(sub['fraction'].dropna().unique())

    pt = sub.pivot_table(index=['variant', 'fraction'], columns='property',
                         values='point', aggfunc='first').reindex(columns=list(properties))
    hf = sub.pivot_table(index=['variant', 'fraction'], columns='property',
                         values='half',  aggfunc='first').reindex(columns=list(properties))

    idx = [(v, f) for v in variants for f in fracs if (v, f) in pt.index]
    pt = pt.reindex(idx); hf = hf.reindex(idx)

    lower = _METRIC_LOWER_IS_BETTER.get(metric, True)
    best: dict[str, tuple] = {}
    if highlight_best:
        for prop in properties:
            col = pt[prop].dropna()
            if not col.empty:
                best[prop] = col.idxmin() if lower else col.idxmax()

    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             rf'\begin{{tabular}}{{ll{"c" * len(list(properties))}}}',
             r'\toprule',
             ' & '.join(['variant', 'fraction'] +
                        [_PROP_LABELS.get(p, p) for p in properties]) + r' \\',
             r'\midrule']
    for (v, f) in idx:
        cells = [v, f'{f:g}']
        for prop in properties:
            pv = pt.at[(v, f), prop]
            hv = hf.at[(v, f), prop]
            if pd.isna(pv):
                cells.append('--')
            else:
                cell = rf'${pv:.{n_decimals}f}\!\pm\!{hv:.{n_decimals}f}$'
                if highlight_best and best.get(prop) == (v, f):
                    cell = r'\textbf{' + cell + '}'
                cells.append(cell)
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    if caption:
        lines.append(rf'\caption{{{caption}}}')
    lines.append(rf'\label{{{label}}}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Dipole analysis
# ─────────────────────────────────────────────────────────────────────────────

def _dipole_pred_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    if DIPOLE_PRED in df.columns:
        cols.append(DIPOLE_PRED)
    cols += sorted(c for c in df.columns
                   if c.startswith(f'{DIPOLE_PRED}_') and c != DIPOLE_PRED)
    return cols


def dipole_metric_table(
    df: pd.DataFrame,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    metrics: Iterable[str] = ('MAE', 'RMSE', 'R2', 'CCC', 'Spearman'),
) -> pd.DataFrame:
    """Bootstrap-CI metric table for each inferred-dipole column vs reference ``mu``."""
    pred_cols = _dipole_pred_cols(df)
    if not pred_cols:
        return pd.DataFrame()
    t = df[DIPOLE_COL].to_numpy(float)
    rows = []
    for col in pred_cols:
        p = df[col].to_numpy(float)
        mask = ~(np.isnan(t) | np.isnan(p))
        for metric in metrics:
            pt, lo, hi = bootstrap_metric_ci(t[mask], p[mask], metric=metric,
                                             n_boot=n_boot, ci=ci, seed=seed)
            rows.append({'architecture': col, 'metric': metric,
                         'point': pt, 'lo': lo, 'hi': hi, 'n': int(mask.sum())})
    return pd.DataFrame(rows)


def dipole_parity(
    df: pd.DataFrame,
    metric: str = 'MAE',
    metric_df: Optional[pd.DataFrame] = None,
    save: Optional[str] = None,
    ax_size: tuple[float, float] = (3.4, 3.4),
    bins: int = 80,
    marginal_log: bool = True,
    cmap: str = 'winter',
):
    """Hexbin parity of inferred dipole vs reference ``mu`` (Debye).

    Parameters
    ----------
    metric : str
        Metric to display in the in-panel annotation.  Any of MAE/RMSE/R2/CCC/Spearman.
    metric_df : optional
        Output of :func:`dipole_metric_table`.  When provided, the panel shows
        ``point ±half-CI``.  Otherwise the point estimate alone.
    """
    pred_cols = _dipole_pred_cols(df)
    if not pred_cols:
        raise ValueError("No inferred-dipole columns found.")
    n = len(pred_cols)
    fig, axes = plt.subplots(1, n, figsize=(n * ax_size[0], ax_size[1]),
                              squeeze=False)
    t = df[DIPOLE_COL].to_numpy(float)

    for j, (ax, col) in enumerate(zip(axes[0], pred_cols)):
        p = df[col].to_numpy(float)
        mask = ~(np.isnan(t) | np.isnan(p))
        tt, pp = t[mask], p[mask]

        dmin = min(tt.min(), pp.min())
        dmax = max(tt.max(), pp.max())
        pad  = 0.03 * (dmax - dmin)
        lim  = (dmin - pad, dmax + pad)

        hb = ax.hexbin(tt, pp, gridsize=bins, cmap=cmap,
                       norm=LogNorm(), mincnt=1,
                       extent=(*lim, *lim), linewidths=0.2)
        ax.plot(lim, lim, 'k--', linewidth=0.8, alpha=0.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect('equal', adjustable='box')

        ax.grid(True, alpha=0.25, linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)

        _add_marginal(ax, tt, pp, color="#27d687", marginal_log=marginal_log)
        ax.tick_params(axis='both', which='both', labelsize=_FONT['tick'],
                       direction='out', length=3, pad=2,
                       labelbottom=True, labelleft=True)

        # Inset colorbar
        bg  = ax.inset_axes([0.44, 0.06, 0.55, 0.18])
        bg.set_facecolor('white'); bg.patch.set_alpha(0.92)
        bg.set_xticks([]); bg.set_yticks([])
        for sp in bg.spines.values():
            sp.set_visible(False)
        cax = ax.inset_axes([0.50, 0.10, 0.46, 0.045])
        cb  = plt.colorbar(hb, cax=cax, orientation='horizontal')
        cb.ax.xaxis.set_major_locator(LogLocator(base=10))
        cb.ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        cb.ax.tick_params(labelsize=8, length=2, pad=3)
        cb.outline.set_linewidth(0.4)
        cax.set_title('Count', fontsize=9, pad=4)

        # In-panel metric annotation (with CI if metric_df given)
        ann = _format_dipole_metric(metric_df, col, metric, tt, pp)
        metric_label = '$R^2$' if metric == 'R2' else metric
        unit = ' D' if metric in ('MAE', 'RMSE') else ''
        ax.text(0.04, 0.96,
                f"{metric_label}={ann}{unit}",
                transform=ax.transAxes, ha='left', va='top', fontsize=_FONT['annot'],
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', alpha=0.92))

        ax.set_xlabel(r'Target molecular $|\mathbf{\mu}|$ [$D$]', fontsize=14)
        if j == 0:
            ax.set_ylabel(r'Molecular $|\mathbf{\mu}|$ from QT-Net', fontsize=14)

    fig.tight_layout(pad=0., w_pad=0.0)
    if save:
        fig.savefig(save, dpi=200, bbox_inches='tight')
    return fig


def _format_dipole_metric(
    metric_df: Optional[pd.DataFrame],
    arch_col: str,
    metric: str,
    t: np.ndarray,
    p: np.ndarray,
) -> str:
    """Return ``'value ±half'`` from a dipole metric_df row, or point estimate."""
    if metric_df is not None:
        row = metric_df[
            (metric_df['architecture'] == arch_col)
            & (metric_df['metric'] == metric)
        ]
        if not row.empty:
            r = row.iloc[0]
            half = (r['hi'] - r['lo']) / 2.0
            return f"{r['point']:.3f}±{half:.3f}"
    return f"{compute_metrics(t, p)[metric]:.3f}"


def dipole_residual_breakdown(
    df: pd.DataFrame,
    by: str = 'n_atoms',
    save: Optional[str] = None,
    ax_size: tuple[float, float] = (9.0, 4.0),
):
    """Box plot of |mu_inferred − mu| grouped by *by* (e.g. n_atoms)."""
    if by not in df.columns:
        raise KeyError(f"Column '{by}' not in DataFrame.")
    pred_cols = _dipole_pred_cols(df)
    t = df[DIPOLE_COL].astype(float)
    work = pd.DataFrame({'group': df[by].values})
    for col in pred_cols:
        work[col] = (df[col].astype(float) - t).abs()
    work = work.dropna()
    groups = sorted(work['group'].unique())

    fig, ax = plt.subplots(figsize=ax_size)
    width = 0.8 / max(len(pred_cols), 1)
    for k, col in enumerate(pred_cols):
        positions = np.arange(len(groups)) + (k - (len(pred_cols) - 1) / 2) * width
        data = [work.loc[work['group'] == g, col].values for g in groups]
        bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                        patch_artist=True, showfliers=False)
        color = _TAB10_HEX[k % len(_TAB10_HEX)]
        for patch in bp['boxes']:
            patch.set_facecolor(color); patch.set_alpha(0.6)
        for elt in ('whiskers', 'caps', 'medians'):
            for line in bp[elt]:
                line.set_color('black')

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_xlabel(by, fontsize=_FONT['label'])
    ax.set_ylabel(r'$|\mu_\mathrm{inferred}-\mu_\mathrm{ref}|$ (D)', fontsize=_FONT['label'])
    ax.set_title(f'Dipole residual by {by}', fontsize=_FONT['title'])
    ax.tick_params(labelsize=_FONT['tick'])
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    if len(pred_cols) > 1:
        handles = [Line2D([0], [0], marker='s', color='w',
                          markerfacecolor=_TAB10_HEX[k % len(_TAB10_HEX)], markersize=9, alpha=0.7,
                          label=col)
                   for k, col in enumerate(pred_cols)]
        ax.legend(handles=handles, fontsize=_FONT['legend'], frameon=False)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200, bbox_inches='tight')
    return fig


def compare_variants_bootstrap(
    df: pd.DataFrame,
    fraction: float,
    prop: str,
    variant_a: str = 'informed',
    variant_b: str = 'blind',
    metric: str = 'CCC',
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Paired bootstrap CI on ``metric(variant_a) − metric(variant_b)``.

    Each bootstrap resample picks the same molecule indices for both variants,
    so the CI accounts for the paired structure (the variants are evaluated on
    the same molecules).  CI containing zero ⇒ the variants are not
    significantly different at this fraction/property.

    This is the rigorous answer to "is informed significantly better than
    blind at fraction X for property Y?" given that we have a single
    ensemble-averaged prediction per molecule.
    """
    pred_a   = f'{variant_a}_{fraction}_pred_{prop}'
    pred_b   = f'{variant_b}_{fraction}_pred_{prop}'
    target_c = f'target_{prop}'
    for c in (pred_a, pred_b, target_c):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not in DataFrame.")
    fn = _METRIC_FUNCS[metric] if isinstance(metric, str) else metric

    sub = df[[target_c, pred_a, pred_b]].dropna()
    t = sub[target_c].to_numpy(float)
    a = sub[pred_a].to_numpy(float)
    b = sub[pred_b].to_numpy(float)
    n = len(t)
    if n < 2:
        raise ValueError("Not enough valid molecules.")

    point = float(fn(t, a) - fn(t, b))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[k] = fn(t[idx], a[idx]) - fn(t[idx], b[idx])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(diffs, [alpha, 1.0 - alpha])
    return {
        'point':     point,
        'lo':        float(lo),
        'hi':        float(hi),
        'metric':    metric,
        'fraction':  fraction,
        'property':  prop,
        'variant_a': variant_a,
        'variant_b': variant_b,
        'n':         n,
        'n_boot':    n_boot,
        'significant': not (lo <= 0 <= hi),
    }


def compare_variants_table(
    df: pd.DataFrame,
    properties: Iterable[str] = PROPERTIES,
    fractions: Optional[Iterable[float]] = None,
    metric: str = 'CCC',
    variant_a: str = 'informed',
    variant_b: str = 'blind',
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Run :func:`compare_variants_bootstrap` over every (fraction, property).

    Returns a long-format DataFrame: each row is a (fraction, property) cell
    with the metric difference, its 95% CI, and a ``significant`` flag.
    The sign convention is ``metric(variant_a) − metric(variant_b)``: positive
    ⇒ variant_a wins on a higher-is-better metric.
    """
    if fractions is None:
        fracs_set: set[float] = set()
        for col in df.columns:
            if '_pred_' in col:
                try:
                    fracs_set.add(float(col.split('_pred_')[0].split('_')[-1]))
                except ValueError:
                    pass
        fractions = sorted(fracs_set)

    rows = []
    for frac in fractions:
        for prop in properties:
            try:
                res = compare_variants_bootstrap(
                    df, fraction=frac, prop=prop,
                    variant_a=variant_a, variant_b=variant_b,
                    metric=metric, n_boot=n_boot, ci=ci, seed=seed,
                )
            except (KeyError, ValueError):
                continue
            rows.append(res)
    return pd.DataFrame(rows)


def compare_architectures_bootstrap(
    df: pd.DataFrame,
    arch_a: str,
    arch_b: str,
    metric: str = 'MAE',
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Paired bootstrap CI on metric(arch_a) − metric(arch_b)."""
    for c in (arch_a, arch_b):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not in DataFrame.")
    fn = _METRIC_FUNCS[metric] if isinstance(metric, str) else metric
    t = df[DIPOLE_COL].to_numpy(float)
    a = df[arch_a].to_numpy(float)
    b = df[arch_b].to_numpy(float)
    mask = ~(np.isnan(t) | np.isnan(a) | np.isnan(b))
    t, a, b = t[mask], a[mask], b[mask]
    point = float(fn(t, a) - fn(t, b))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, len(t), len(t))
        diffs[k] = fn(t[idx], a[idx]) - fn(t[idx], b[idx])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(diffs, [alpha, 1.0 - alpha])
    return {'point': point, 'lo': float(lo), 'hi': float(hi),
            'metric': metric, 'arch_a': arch_a, 'arch_b': arch_b,
            'n_boot': n_boot, 'n': int(mask.sum())}
