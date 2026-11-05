"""
Figures, tables and checks of the environment-holdout study, on top of
``env_holdout_analysis.py`` (``eha``, which writes the result tables).

Used by ``analysis/molecular/analyze_env_holdout.ipynb``.  Nothing here fits a model; it
reads the run directories ``experiments/molecular/env_holdout/<variant>/fold_<n>/frac_<f>/``
(``config.json``, ``test_preds.pkl``, ``val_preds.pkl``, ``remainder_preds.pkl``,
``train_ids.json``) and the shipped label files, and joins everything on the QM9 id.

  * checks: grid completeness, equal training sizes, code commits, and that every panel-B/C
    molecule carries an environment absent from its run's training molecules (``env_support``);
  * informed vs blind: R² per property and fraction on the AIMEl test set and on the QM9
    remainder, with fold-paired t / Wilcoxon (Holm) and Tukey's HSD over the eight models;
  * panels: mean-over-properties R² of each panel minus that of the pooled test set;
  * novelty with respect to the model's own training set: each atom's distance to its 15th-nearest
    training atom in the 6-Bohr classification space (unscaled PCA), as a percentile of the model's validation
    atoms, and the error of molecules grouped by their number of rare (> p99) atoms.

The unit of inference is the fold (n = 25), paired across models by fold.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import shapiro, spearmanr, t as t_dist, ttest_1samp, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (os.path.join(REPO_ROOT, 'analysis', 'atomic'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_holdout_analysis as eha  # noqa: E402
import result_analysis as ra  # noqa: E402  (QT-Net: analysis/atomic/result_analysis.py)

PROPS = list(eha.PROPS)
VARIANTS = list(eha.VARIANTS)
PANELS = list(eha.PANELS)
FRACTIONS = [0.01, 0.05, 0.1, 1.0]
MODELS = [f'{v}_{f}' for v in VARIANTS for f in FRACTIONS]
EXPECTED_N_TRAIN = {0.01: 160, 0.05: 803, 0.1: 1607, 1.0: 16070}
RUNS_DIR = eha.RUNS_DIR
OUT_DIR = eha.OUT_DIR
DIST_CACHE = os.path.join(OUT_DIR, 'distance_cache')

VARIANT_COLORS = {'informed': '#d62728', 'blind': '#1f77b4'}
PANEL_DESC = {'A': 'chemotype held out, environments seen',
              'B': 'chemotype held out, environments unseen',
              'C': 'chemotype seen, environments unseen',
              'D': 'chemotype seen, environments seen'}
PROP_LABELS = {'alpha': r'$\alpha$', 'gap': r'$\Delta\varepsilon$', 'U0': r'$U_0$', 'Cv': r'$C_v$',
               'mean': 'mean over properties'}
# novelty groups, less -> more novel, coloured blue -> red like the rest of the molecular figures
NOVELTY_GROUPS = ['0', '1–2', '≥3']
NOVELTY_COLORS = {'0': '#3b4cc0', '1–2': '#f39c7d', '≥3': '#b40426'}


# ── imports and small statistics ─────────────────────────────────────────────

def holm(p: Iterable[float]) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (NaNs left out of the family)."""
    p = np.asarray(list(p), dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (m - i) * p[j])
        out[j] = min(running, 1.0)
    return out


def fold_summary(values) -> dict:
    """Mean over folds with its SE, 95 % t-interval, and one-sample t / Wilcoxon p (H0: 0)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return {'n_folds': n, 'mean': float(v.mean()) if n else np.nan, 'se': np.nan,
                'ci_lo': np.nan, 'ci_hi': np.nan, 'p_t': np.nan, 'p_wilcoxon': np.nan}
    se = float(v.std(ddof=1) / np.sqrt(n))
    h = float(t_dist.ppf(0.975, n - 1)) * se
    try:
        p_w = float(wilcoxon(v).pvalue)
    except ValueError:
        p_w = np.nan
    return {'n_folds': n, 'mean': float(v.mean()), 'se': se, 'ci_lo': float(v.mean()) - h,
            'ci_hi': float(v.mean()) + h, 'p_t': float(ttest_1samp(v, 0.0).pvalue), 'p_wilcoxon': p_w}


def fmt_ci(mean, lo, hi, d: int = 3) -> str:
    return f'{mean:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'


def fmt_p(p) -> str:
    return '—' if not np.isfinite(p) else ('<0.001' if p < 1e-3 else f'{p:.3f}')


# ── inventory and coherence ──────────────────────────────────────────────────

def _run_dirs(runs_dir=RUNS_DIR):
    for p in sorted(glob.glob(os.path.join(runs_dir, '*', 'fold_*', 'frac_*', 'config.json'))):
        variant, fold, frac = p.split(os.sep)[-4:-1]
        yield variant, int(fold.split('_')[1]), float(frac.split('_')[1]), os.path.dirname(p)


def run_inventory(runs_dir=RUNS_DIR) -> pd.DataFrame:
    """One row per run: identity, sizes, training outcome and which outputs exist."""
    rows = []
    for variant, fold, frac, d in _run_dirs(runs_dir):
        cfg = json.load(open(os.path.join(d, 'config.json')))
        res = cfg.get('results') or {}
        rec = {'variant': variant, 'fold': fold, 'fraction': frac, 'n_train': cfg['n_train'],
               'epochs': cfg['epochs'], 'code_commit': cfg.get('code_commit'),
               'best_epoch': res.get('best_epoch'), 'predicted_from': res.get('predicted_from')}
        for f in ('test_preds.pkl', 'val_preds.pkl', 'remainder_preds.pkl', 'train_ids.json'):
            rec[f'has_{f.split(".")[0]}'] = os.path.exists(os.path.join(d, f))
        rows.append(rec)
    return pd.DataFrame(rows)


def _code_differs(c1: str, c2: str, paths=('src', 'scripts')) -> Optional[bool]:
    """True/False if the training + deployment code differs between two commits; None if unknown."""
    try:
        r = subprocess.run(['git', '-C', REPO_ROOT, 'diff', '--quiet', c1, c2, '--', *paths],
                           capture_output=True, timeout=30)
    except Exception:
        return None
    return {0: False, 1: True}.get(r.returncode)


def check_inventory(inv: pd.DataFrame, n_folds: int = 25) -> pd.DataFrame:
    """Grid completeness per variant × fraction.  Raises if a training size differs from the
    design; prints the code-commit check."""
    g = inv.groupby(['variant', 'fraction'])
    out = pd.DataFrame({
        'runs': g.size(), 'n_train': g['n_train'].agg(lambda s: '/'.join(map(str, sorted(set(s))))),
        'test_preds': g['has_test_preds'].sum(), 'remainder_preds': g['has_remainder_preds'].sum(),
        'train_ids': g['has_train_ids'].sum(),
        'best_checkpoint': g['predicted_from'].agg(lambda s: int((s == 'best_checkpoint').sum())),
        'best_epoch_median': g['best_epoch'].median(),
    }).reset_index()
    bad = inv[inv['n_train'] != inv['fraction'].map(EXPECTED_N_TRAIN)]
    if len(bad):
        raise AssertionError(f"{len(bad)} runs with a training size off the design:\n{bad}")
    commits = sorted(inv['code_commit'].dropna().unique())
    print(f"runs: {len(inv)} | code commits: " +
          ', '.join(f"{c[:7]} ({int((inv['code_commit'] == c).sum())})" for c in commits))
    for i in range(1, len(commits)):
        d = _code_differs(commits[0], commits[i])
        verdict = {False: 'identical', True: 'DIFFERENT', None: 'unknown (commit not in this repository)'}[d]
        print(f"  src/ + scripts/ between {commits[0][:7]} and {commits[i][:7]}: {verdict}")
    return out


def fold_design(runs_dir=RUNS_DIR) -> pd.DataFrame:
    """One row per fold, from the committed ``config.json`` of the blind frac-1.0 runs (the splits are
    the same for both variants and every fraction): repeat r, chemotype group c, the reserved
    environment fold (r + c) mod 5 with its seed label, the number of held-out chemotypes, the four
    panel sizes and the training pool."""
    rows = []
    for variant, fold, frac, d in _run_dirs(runs_dir):
        if variant != 'blind' or frac != 1.0:
            continue
        c = json.load(open(os.path.join(d, 'config.json')))
        rows.append({'fold': fold, 'repeat': c['repeat'], 'chemotype_group': c['cv_fold'], 'env_fold': c['env_fold'],
                     'env_seed': c['env_fold_seed'], 'n_heldout_chemotypes': len(c['heldout_chemotypes']),
                     **{p: c['panel_sizes'][p] for p in PANELS}, 'n_test': c['n_test'], 'pool': c['pool_size'],
                     'n_val': c['n_val'], 'n_train_frac1': c['n_train']})
    return pd.DataFrame(rows).set_index('fold').sort_index()


def panel_visits(labels: pd.DataFrame, env_folds: list, runs_dir=RUNS_DIR):
    """How the 25 test sets visit each AIMEl molecule.  Per molecule (blind frac-1.0 test sets): the
    number of folds in which it is scored in each panel, and ``k`` = the number of environment folds
    it carries (an atom with one of the fold's labels).  Returns (per-molecule table, crosstab of k
    against the number of B visits).  By design every molecule is in A or B five times (once per
    repeat, when its chemotype group is held out), in B k times and in C 4k times; the exceptions are
    the chemotypes that change group between repeats."""
    parts = []
    for variant, fold, frac, d in _run_dirs(runs_dir):
        if variant == 'blind' and frac == 1.0:
            parts.append(pd.read_pickle(os.path.join(d, 'test_preds.pkl'))[['index', 'panel']].assign(fold=fold))
    t = pd.concat(parts, ignore_index=True)
    visits = t.pivot_table(index='index', columns='panel', values='fold', aggfunc='size', fill_value=0)[PANELS]
    sets = [set(f['labels']) for f in env_folds]
    lab = labels.set_index('index')['atom_env_labels_6b']
    visits['k'] = [sum(bool(s & set(lab.loc[i])) for s in sets) for i in visits.index]
    return visits, pd.crosstab(visits['k'], visits['B'], margins=True)


def load_scales(out_dir=OUT_DIR) -> dict:
    """eha's error scales (std of each target over AIMEl); checked against the committed JSON."""
    path = os.path.join(out_dir, 'property_scales.json')
    committed = json.load(open(path)) if os.path.exists(path) else None
    if os.path.exists(eha.AIMEL_PKL):
        scales = eha.property_scales()
        if committed is not None:
            assert all(np.isclose(scales[p], committed[p], rtol=1e-12) for p in PROPS), (scales, committed)
        return scales
    if committed is None:
        raise FileNotFoundError(f"neither {eha.AIMEL_PKL} nor {path}")
    return committed


def compare_tables(new: pd.DataFrame, committed: pd.DataFrame, keys: Sequence[str], rtol=1e-9) -> pd.DataFrame:
    """Max relative difference of every numeric column after joining on ``keys``; raises when rows
    are missing on either side, a value is NaN on one side only, or a value differs beyond ``rtol``."""
    m = new.merge(committed, on=list(keys), how='outer', suffixes=('_new', '_csv'), indicator=True)
    if (m['_merge'] != 'both').any():
        raise AssertionError(f"{int((m['_merge'] != 'both').sum())} rows present on one side only")
    rows = []
    for c in new.columns:
        if c in keys or f'{c}_new' not in m or not pd.api.types.is_numeric_dtype(m[f'{c}_new']) \
                or pd.api.types.is_bool_dtype(m[f'{c}_new']):
            continue
        a, b = m[f'{c}_new'].to_numpy(float), m[f'{c}_csv'].to_numpy(float)
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
        ok = ~np.isnan(a) & ~np.isnan(b)
        rel = float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-12))) if ok.any() else 0.0
        rows.append({'column': c, 'max_rel_diff': rel if nan_mismatch == 0 else np.inf,
                     'nan_mismatches': nan_mismatch})
    out = pd.DataFrame(rows)
    if (out['max_rel_diff'] > rtol).any():
        raise AssertionError(f"tables differ:\n{out[out['max_rel_diff'] > rtol]}")
    return out


def _labelled_atoms(labels: pd.DataFrame) -> pd.DataFrame:
    """(index, label) for every non-noise atom."""
    a = labels[['index', 'atom_env_labels_6b']].explode('atom_env_labels_6b').rename(
        columns={'atom_env_labels_6b': 'label'})
    a = a[~a['label'].str.endswith('_-1')]
    return a.assign(index=a['index'].astype(np.int64)).reset_index(drop=True)


def _training_sets(runs_dir=RUNS_DIR):
    """(fold, fraction) -> (train ids, val ids, test frame with index/panel), asserting that the
    two variants were trained, validated and tested on the same molecules."""
    sets = {}
    for variant, fold, frac, d in _run_dirs(runs_dir):
        ids = json.load(open(os.path.join(d, 'train_ids.json')))
        test = pd.read_pickle(os.path.join(d, 'test_preds.pkl'))[['index', 'panel']]
        key = (fold, frac)
        if key in sets:
            tr, va, te = sets[key]
            if set(tr) != set(ids['train']) or set(va) != set(ids['val']):
                raise AssertionError(f"fold {fold} frac {frac}: variants trained on different molecules")
            if not te.sort_values('index').reset_index(drop=True).equals(
                    test.sort_values('index').reset_index(drop=True)):
                raise AssertionError(f"fold {fold} frac {frac}: variants tested on different molecules")
        else:
            sets[key] = (list(ids['train']), list(ids['val']), test)
    return sets


def env_support(labels: pd.DataFrame, runs_dir=RUNS_DIR) -> pd.DataFrame:
    """Per (fold, fraction) and test molecule: ``n_labelled`` atoms, ``n_unseen_env_atoms`` (labelled
    atoms whose environment has no atom in THIS run's training molecules) and ``min_env_train_atoms``
    (-1 when every atom is HDBSCAN noise)."""
    lab = _labelled_atoms(labels)
    ids = lab['index'].to_numpy()
    parts = []
    for (fold, frac), (train, _, test) in sorted(_training_sets(runs_dir).items()):
        counts = lab.loc[np.isin(ids, train), 'label'].value_counts()
        t = lab[np.isin(ids, test['index'].to_numpy())]
        c = t['label'].map(counts).fillna(0).to_numpy()
        per = pd.DataFrame({'index': t['index'].to_numpy(), 'c': c}).groupby('index')['c'].agg(
            n_labelled='size', n_unseen_env_atoms=lambda s: int((s == 0).sum()), min_env_train_atoms='min')
        out = test.merge(per, left_on='index', right_index=True, how='left')
        out['n_labelled'] = out['n_labelled'].fillna(0).astype(int)
        out['n_unseen_env_atoms'] = out['n_unseen_env_atoms'].fillna(0).astype(int)
        out['min_env_train_atoms'] = out['min_env_train_atoms'].fillna(-1).astype(int)
        parts.append(out.assign(fold=fold, fraction=frac))
    return pd.concat(parts, ignore_index=True)


def coherence_table(support: pd.DataFrame) -> pd.DataFrame:
    """Per fraction × panel: share of molecules with >= 1 environment absent from the run's training
    molecules.  Asserts the design: every B and C molecule carries a reserved environment."""
    bc = support[support['panel'].isin(['B', 'C'])]
    if (bc['n_unseen_env_atoms'] < 1).any():
        raise AssertionError(f"{int((bc['n_unseen_env_atoms'] < 1).sum())} panel-B/C molecules have "
                             "every environment in training: a reserved environment leaked")
    return (100 * support.groupby(['fraction', 'panel'])['n_unseen_env_atoms'].agg(lambda s: (s > 0).mean())
            ).round(1).unstack('panel')


# ── R²: informed vs blind ─────────────────────────────────────────────────────

def aimel_metrics(fm: pd.DataFrame, panel: str = 'all') -> pd.DataFrame:
    """``eha.aimel_fold_metrics`` rows of one panel ('all' = the pooled test set), in the long schema
    variant, fraction, fold, property, R2, MAE, ..."""
    s = fm[fm['panel'] == panel]
    return s.rename(columns={'cv_cycle': 'fold'})[['variant', 'fraction', 'fold', 'property', 'n_mols',
                                                   'R2', 'MAE', 'RMSE', 'CCC', 'Spearman']].reset_index(drop=True)


def remainder_metrics_all(runs_dir=RUNS_DIR) -> pd.DataFrame:
    """R², MAE, ... of every run on the whole QM9 remainder (98,096 molecules, all strata pooled)."""
    rows = []
    for variant, fold, frac, d in _run_dirs(runs_dir):
        r = pd.read_pickle(os.path.join(d, 'remainder_preds.pkl'))
        for p in PROPS:
            rec = {'variant': variant, 'fraction': frac, 'fold': fold, 'property': p, 'n_mols': len(r)}
            rec.update(ra._compute_metrics_dict(r[f'target_{p}'].to_numpy(float), r[f'pred_{p}'].to_numpy(float)))
            rows.append(rec)
    return pd.DataFrame(rows)


def _tukey_p(m: pd.DataFrame, metric: str, props: Sequence[str]) -> dict:
    """Tukey HSD p-value matrices over the eight models (variant × fraction) per property, from the
    fold-level RM-ANOVA (``ra.tukey_pvalue_matrix`` with the fold as the subject)."""
    sub = m.assign(method=m['variant'] + '_' + m['fraction'].astype(str),   # 'informed_0.01' ... 'blind_1.0'
                   cv_cycle=m['fold'], repeat=m['fold'], split='test', panel='global')
    res = ra.tukey_pvalue_matrix(sub, models=MODELS, metric=metric, properties=list(props), panels=['global'],
                                 panel_col='panel', split=None, print_tables=False)
    return {prop: r for (_, prop), r in res.items()}


def informed_vs_blind(m: pd.DataFrame, metric: str = 'R2', props: Sequence[str] = PROPS, alpha: float = 0.05,
                      verdict_by: str = 'p_t_holm') -> pd.DataFrame:
    """Per fraction × property: fold-level mean ± 95 % t-CI of each variant, the paired difference
    informed − blind with its CI, paired t and Wilcoxon p (Holm over the fraction × property cells
    of the table), Shapiro–Wilk p of the differences, and the Tukey HSD p of informed_f vs blind_f
    among the eight models.  ``verdict``: which variant is better (``verdict_by`` < alpha: the Holm
    paired-t p by default, 'p_tukey' for the Tukey HSD p as in the atomic comparison)."""
    higher = metric in {'R2', 'CCC', 'Spearman'}
    tuk = _tukey_p(m, metric, props)
    rows = []
    for frac in FRACTIONS:
        for prop in props:
            w = m[(m['fraction'] == frac) & (m['property'] == prop)].pivot_table(
                index='fold', columns='variant', values=metric).dropna()
            if w.empty:
                continue
            d = (w['informed'] - w['blind']).to_numpy()
            s, si, sb = fold_summary(d), fold_summary(w['informed']), fold_summary(w['blind'])
            t = tuk.get(prop)
            rows.append({'fraction': frac, 'property': prop, 'n_folds': s['n_folds'],
                         'informed': si['mean'], 'informed_lo': si['ci_lo'], 'informed_hi': si['ci_hi'],
                         'blind': sb['mean'], 'blind_lo': sb['ci_lo'], 'blind_hi': sb['ci_hi'],
                         'diff': s['mean'], 'diff_lo': s['ci_lo'], 'diff_hi': s['ci_hi'],
                         'p_t': s['p_t'], 'p_wilcoxon': s['p_wilcoxon'],
                         'p_shapiro_diff': float(shapiro(d).pvalue) if len(d) >= 3 else np.nan,
                         'p_tukey': float(t.pc.loc[f'informed_{frac}', f'blind_{frac}']) if t is not None else np.nan,
                         'msd_tukey': float(t.msd_half) if t is not None else np.nan})
    out = pd.DataFrame(rows)
    out['p_t_holm'] = holm(out['p_t'])
    out['p_wilcoxon_holm'] = holm(out['p_wilcoxon'])
    better = np.where(out['diff'] > 0, 'informed', 'blind') if higher else np.where(out['diff'] < 0, 'informed', 'blind')
    out['verdict'] = np.where(out[verdict_by] < alpha, better, 'n.s.')
    return out


def readable(tab: pd.DataFrame, value_cols=('informed', 'blind', 'diff'), d: int = 3) -> pd.DataFrame:
    """Formatted view of :func:`informed_vs_blind` / :func:`panel_delta` / :func:`panel_contrasts`
    output for the notebook: ``mean [95 % CI]`` strings and formatted p-values."""
    out = tab[[c for c in ('variant', 'fraction', 'property', 'panel', 'contrast') if c in tab]].copy()
    for c in value_cols:
        out[c] = [fmt_ci(a, b, e, d) for a, b, e in zip(tab[c], tab[f'{c}_lo'], tab[f'{c}_hi'])]
    for c in ('p_informed_vs_pooled_holm', 'p_blind_vs_pooled_holm', 'p_t_holm', 'p_wilcoxon_holm', 'p_tukey',
              'p_shapiro_diff'):
        if c in tab:
            out[c] = tab[c].map(fmt_p)
    for c in ('informed_vs_pooled', 'blind_vs_pooled', 'verdict'):
        if c in tab:
            out[c] = tab[c]
    return out



def _fmt_p_tukey_caps(p) -> str:
    """Tukey p in LaTeX, its tabulated limits as inequalities (the studentized-range lookup returns
    0.001 and 0.900 at its ends)."""
    if not np.isfinite(p):
        return '—'
    return r'$\leq$0.001' if p <= 1e-3 else (r'$\geq$0.9' if p >= 0.9 - 1e-9 else f'{p:.3f}')


def latex_ivb_table(tab: pd.DataFrame, caption: str, label: str, d: int = 3, holm: bool = True) -> str:
    """LaTeX: R² of each variant (fold mean ± 95 % CI half-width), the difference informed − blind
    and its Holm-adjusted paired-t p and Tukey p, per property and fraction.  ``holm=False``: Tukey p
    only, its limits as inequalities (bold follows ``verdict``, so build ``tab`` with
    ``verdict_by='p_tukey'``)."""
    if not holm:
        lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\begin{tabular}{llrrrr}', r'\toprule',
                 r'Property & Fraction & Informed & Blind & Informed $-$ Blind & $p_\mathrm{Tukey}$ \\', r'\midrule']
        for k, prop in enumerate(PROPS):
            s = tab[tab['property'] == prop]
            for i, r in enumerate(s.itertuples()):
                name = PROP_LABELS[prop] if i == 0 else ''
                bold = (lambda x, v: rf'\textbf{{{x}}}' if r.verdict == v else x)
                inf = bold(f'{r.informed:.{d}f} $\\pm$ {r.informed_hi - r.informed:.{d}f}', 'informed')
                bl = bold(f'{r.blind:.{d}f} $\\pm$ {r.blind_hi - r.blind:.{d}f}', 'blind')
                lines.append(f'{name} & {r.fraction:g} & {inf} & {bl} & {r.diff:+.{d}f} & '
                             f'{_fmt_p_tukey_caps(r.p_tukey)} \\\\')
            lines.append(r'\midrule' if k < len(PROPS) - 1 else r'\bottomrule')
        lines += [r'\end{tabular}', rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\end{table}']
        return '\n'.join(lines)
    lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Property & Fraction & Informed & Blind & Informed $-$ Blind & $p_\mathrm{Holm}$ & $p_\mathrm{Tukey}$ \\',
             r'\midrule']
    for k, prop in enumerate(PROPS):
        s = tab[tab['property'] == prop]
        for i, r in enumerate(s.itertuples()):
            name = PROP_LABELS[prop] if i == 0 else ''
            bold = (lambda x, v: rf'\textbf{{{x}}}' if r.verdict == v else x)
            inf = bold(f'{r.informed:.{d}f} $\\pm$ {r.informed_hi - r.informed:.{d}f}', 'informed')
            bl = bold(f'{r.blind:.{d}f} $\\pm$ {r.blind_hi - r.blind:.{d}f}', 'blind')
            lines.append(f'{name} & {r.fraction:g} & {inf} & {bl} & {r.diff:+.{d}f} & '
                         f'{fmt_p(r.p_t_holm).replace("<", "$<$")} & {fmt_p(r.p_tukey).replace("<", "$<$")} \\\\')
        lines.append(r'\midrule' if k < len(PROPS) - 1 else r'\bottomrule')
    lines += [r'\end{tabular}', rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\end{table}']
    return '\n'.join(lines)


# ── panels: mean-over-properties R² relative to the pooled test set ──────────

PANEL_CONTRASTS = {
    'B − A': 'environments unseen vs seen, chemotype held out',
    'C − D': 'environments unseen vs seen, chemotype seen',
    'A − D': 'chemotype held out vs seen, environments seen',
    'B − C': 'chemotype held out vs seen, environments unseen',
}


def _panel_r2(fm: pd.DataFrame, metric: str = 'R2') -> pd.DataFrame:
    """Mean over the four properties of each (variant, fraction, fold, panel) metric; columns =
    panels A–D and 'all' (the pooled test set)."""
    return fm.groupby(['variant', 'fraction', 'cv_cycle', 'panel'])[metric].mean().unstack('panel')


def panel_delta(fm: pd.DataFrame, metric: str = 'R2', alpha: float = 0.05):
    """Per (variant, fraction, fold, panel): ΔR² = R² of the panel averaged over the four properties
    minus the same average on the pooled test set of the same model and fold.

    Returns ``(per_fold, summary)``.  summary, per fraction × panel:
      * each variant's fold-level mean ΔR² with 95 % CI, and whether it differs from 0 — the panel
        is harder ('harder') or easier ('easier') than the pooled test set — by a one-sample t-test
        on the 25 fold values, Holm-adjusted over the 32 variant × panel × fraction cells;
      * the paired difference informed − blind (t / Wilcoxon, Holm over the 16 panel × fraction
        cells): whether the descriptors help more or less on that panel than overall."""
    r = _panel_r2(fm, metric)
    per_fold = r[PANELS].sub(r['all'], axis=0).stack().rename('delta').reset_index().rename(columns={'cv_cycle': 'fold'})
    rows = []
    for (frac, panel), g in per_fold.groupby(['fraction', 'panel']):
        w = g.pivot_table(index='fold', columns='variant', values='delta')
        s = fold_summary((w['informed'] - w['blind']).dropna())
        si, sb = fold_summary(w['informed'].dropna()), fold_summary(w['blind'].dropna())
        rows.append({'fraction': frac, 'panel': panel, 'n_folds': s['n_folds'],
                     'informed': si['mean'], 'informed_lo': si['ci_lo'], 'informed_hi': si['ci_hi'], 'p_informed_vs_pooled': si['p_t'],
                     'blind': sb['mean'], 'blind_lo': sb['ci_lo'], 'blind_hi': sb['ci_hi'], 'p_blind_vs_pooled': sb['p_t'],
                     'diff': s['mean'], 'diff_lo': s['ci_lo'], 'diff_hi': s['ci_hi'],
                     'p_t': s['p_t'], 'p_wilcoxon': s['p_wilcoxon']})
    out = pd.DataFrame(rows)
    adj = holm(np.r_[out['p_informed_vs_pooled'], out['p_blind_vs_pooled']])
    out['p_informed_vs_pooled_holm'], out['p_blind_vs_pooled_holm'] = adj[:len(out)], adj[len(out):]
    for v in VARIANTS:
        out[f'{v}_vs_pooled'] = np.where(out[f'p_{v}_vs_pooled_holm'] < alpha,
                                         np.where(out[v] < 0, 'harder', 'easier'), 'n.s.')
    out['p_t_holm'] = holm(out['p_t'])
    out['p_wilcoxon_holm'] = holm(out['p_wilcoxon'])
    out['verdict'] = np.where(out['p_t_holm'] < alpha, np.where(out['diff'] > 0, 'informed', 'blind'), 'n.s.')
    return per_fold, out


def panel_contrasts(fm: pd.DataFrame, metric: str = 'R2', contrasts=tuple(PANEL_CONTRASTS)) -> pd.DataFrame:
    """Fold-paired differences between panels of the mean-over-properties R², per variant and fraction:
    the 2 × 2 of the design (``PANEL_CONTRASTS``).  B − A and C − D are the cost of unseen
    environments (chemotype held out / seen); A − D and B − C the cost of an unseen chemotype
    (environments seen / unseen).  One-sample t and Wilcoxon on the fold differences, Holm over all
    rows of the table."""
    r = _panel_r2(fm, metric)
    rows = []
    for (v, frac), g in r.groupby(level=['variant', 'fraction']):
        for name in contrasts:
            p, q = [x.strip() for x in name.split('−')]
            s = fold_summary((g[p] - g[q]).dropna())
            rows.append({'variant': v, 'fraction': frac, 'contrast': name, 'meaning': PANEL_CONTRASTS.get(name, ''),
                         'n_folds': s['n_folds'], 'diff': s['mean'], 'diff_lo': s['ci_lo'], 'diff_hi': s['ci_hi'],
                         'p_t': s['p_t'], 'p_wilcoxon': s['p_wilcoxon']})
    out = pd.DataFrame(rows)
    out['p_t_holm'] = holm(out['p_t'])
    out['p_wilcoxon_holm'] = holm(out['p_wilcoxon'])
    return out


# ── novelty with respect to the model's training set (6-Bohr space, k = 15) ───

def remainder_sample(labels: pd.DataFrame, n: int = 20000, seed: int = 0) -> np.ndarray:
    """A fixed, seeded sample of remainder QM9 ids, the same for every run."""
    ids = np.sort(labels.loc[~labels['in_aimel'], 'index'].to_numpy(np.int64))
    return np.sort(np.random.default_rng(seed).choice(ids, min(n, len(ids)), replace=False))


def run_distances(pca: Dict, fold: int, frac: float, train, val, test: pd.DataFrame, rem_ids: np.ndarray,
                  ks: Sequence[int] = (1, 5, 15), far_pct: float = 99.0, space: str = 'pca') -> pd.DataFrame:
    """For one training set: per validation, test and sampled remainder molecule, the distance of
    its atomic environments to the environments of the training molecules.

    ``space='pca'`` (default): the 6-Bohr classification's own coordinates, i.e. the SOAP vectors
    projected on the whole-QM9 principal components, unscaled, the space in which HDBSCAN drew the
    environment labels (Euclidean distance there is the SOAP distance up to the 1 % of variance left
    out).  ``space='whitened'``: each component divided by its whole-QM9 standard deviation; this
    shrinks the high-variance directions along which, e.g., triple bonds differ, and misses them.

    Per element, in that space, each query atom's distance d_k to its k-th
    nearest training atom of the same element, calibrated against the run's validation atoms
    (carved from the same training pool, so the same chemotypes; their environments are in the
    pool but, at small fractions, not always in the training subsample), with the training set
    as reference and the validation set as the held-out calibration:
      pct_k = % of validation atoms with d_k <= this atom's;  rare = pct_k > ``far_pct``
      z_k   = log(d_k / median of the validation d_k).
    Per molecule: ``n_far_k`` = number of rare atoms, ``D_max_k`` / ``D_mean_k`` = max / mean z_k,
    ``n_atoms``.  Validation molecules are flagged rare at 1 % per atom by construction.
    """
    import compare_atomic_envs_6b as cae6

    parts = []
    test_ids = test['index'].to_numpy(np.int64)
    for el in cae6.ELEMENTS:
        d = pca[el]
        Xw = cae6.whiten(pca, el) if space == 'whitened' else d['X'].astype(np.float32)
        ids = d['index']
        tr = np.isin(ids, train)
        sets = {'val': np.isin(ids, val), 'test': np.isin(ids, test_ids), 'remainder': np.isin(ids, rem_ids)}
        q = sets['val'] | sets['test'] | sets['remainder']
        D = cae6.knn_distance(Xw[tr], Xw[q], ks)
        name = np.select([sets['val'][q], sets['test'][q]], ['val', 'test'], 'remainder')
        rec = {'index': ids[q], 'set': name}
        isval = name == 'val'
        for j, k in enumerate(ks):
            dv = np.sort(D[isval, j])
            med = float(np.median(dv))
            rec[f'z{k}'] = np.log(np.maximum(D[:, j], 1e-3 * med) / med)
            rec[f'pct{k}'] = np.searchsorted(dv, D[:, j], side='right') / len(dv) * 100.0
        parts.append(pd.DataFrame(rec))
    a = pd.concat(parts, ignore_index=True)
    agg = {'n_atoms': ('set', 'size')}
    for k in ks:
        agg.update({f'D_max_{k}': (f'z{k}', 'max'), f'D_mean_{k}': (f'z{k}', 'mean')})
        a[f'far{k}'] = a[f'pct{k}'] > far_pct
        agg[f'n_far_{k}'] = (f'far{k}', 'sum')
    out = a.groupby(['set', 'index']).agg(**agg).reset_index()
    out = out.merge(test, on='index', how='left')
    return out.assign(fold=fold, fraction=frac)


def distance_tables(pca: Dict, labels: pd.DataFrame, runs_dir=RUNS_DIR, n_remainder: int = 20000,
                    ks: Sequence[int] = (1, 5, 15), cache_dir: Optional[str] = None, verbose: bool = True,
                    seed: int = 0, space: str = 'pca') -> pd.DataFrame:
    """:func:`run_distances` for every (fold, fraction) training set (shared by both variants),
    cached per training set (default cache: ``distance_cache_<space>``).  The cache directory records
    the parameters it was built with (``params.json``) and refuses to be reused with different ones."""
    import time
    cache_dir = cache_dir or f'{DIST_CACHE}_{space}'
    os.makedirs(cache_dir, exist_ok=True)
    params = {'n_remainder': n_remainder, 'remainder_seed': seed, 'ks': list(ks), 'far_pct': 99.0, 'space': space,
              'pca_components': {el: int(v['X'].shape[1]) for el, v in pca.items()},
              'pca_source': str(next(iter(pca.values())).get('source'))}
    ppath = os.path.join(cache_dir, 'params.json')
    if os.path.exists(ppath):
        old = json.load(open(ppath))
        if {k: v for k, v in old.items() if k != 'pca_source'} != {k: v for k, v in params.items() if k != 'pca_source'}:
            raise ValueError(f"{cache_dir} was built with {old}; now {params}. Use another cache_dir or delete it.")
    else:
        json.dump(params, open(ppath, 'w'), indent=1)
    rem_ids = remainder_sample(labels, n_remainder, seed)
    parts = []
    t0 = time.time()
    for (fold, frac), (train, val, test) in sorted(_training_sets(runs_dir).items(), key=lambda kv: (kv[0][1], kv[0][0])):
        p = os.path.join(cache_dir, f'fold_{fold}_frac_{frac}_rem{n_remainder}_k{"-".join(map(str, ks))}.pkl')
        if os.path.exists(p):
            parts.append(pd.read_pickle(p))
            continue
        t = run_distances(pca, fold, frac, train, val, test, rem_ids, ks, space=space)
        t.to_pickle(p)
        parts.append(t)
        if verbose:
            print(f"[distance] fold {fold:>2} frac {frac:<4}: {len(t):,} molecules ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(parts, ignore_index=True)


def wide_errors(errors: pd.DataFrame) -> pd.DataFrame:
    """Long eha errors -> one row per (variant, fraction, fold, index, panel), nerr per property + mean."""
    w = errors.pivot_table(index=['variant', 'fraction', 'fold', 'index', 'panel'], columns='property',
                           values='nerr', observed=True)
    w.columns = [f'nerr_{c}' for c in w.columns]
    w['nerr_mean'] = w[[f'nerr_{p}' for p in PROPS]].mean(axis=1)
    return w.reset_index()


def remainder_sample_errors(runs_dir=RUNS_DIR, ids: Optional[np.ndarray] = None, scales=None) -> pd.DataFrame:
    """nerr per property (+ mean) of the sampled remainder molecules, one row per
    (variant, fraction, fold, molecule)."""
    scales = scales or load_scales()
    parts = []
    for variant, fold, frac, d in _run_dirs(runs_dir):
        r = pd.read_pickle(os.path.join(d, 'remainder_preds.pkl'))
        if ids is not None:
            r = r[r['index'].isin(ids)]
        out = r[['index']].copy()
        for p in PROPS:
            out[f'nerr_{p}'] = (r[f'pred_{p}'] - r[f'target_{p}']).abs().to_numpy() / scales[p]
        out['nerr_mean'] = out[[f'nerr_{p}' for p in PROPS]].mean(axis=1)
        parts.append(out.assign(variant=variant, fold=fold, fraction=frac))
    return pd.concat(parts, ignore_index=True)


def novelty_group(n_far) -> pd.Categorical:
    return pd.cut(np.asarray(n_far, dtype=float), [-0.5, 0.5, 2.5, np.inf], labels=NOVELTY_GROUPS)


def novelty_groups(tab: pd.DataFrame, n_far_col: str = 'n_far_15', prop: str = 'mean'):
    """Error of molecules grouped by their number of rare atoms (0 / 1–2 / ≥3) with respect to the
    model's training set.  ``tab`` has one row per (set, variant, fraction, fold, molecule) with
    ``n_far_col`` and ``nerr_<prop>`` (absolute error ÷ the target's standard deviation over AIMEl;
    'mean' averages the four targets).

    Returns ``(summary, contrast)``.  summary, per set × variant × fraction × group: the share of
    molecules, the fold-level mean nerr with 95 % CI, and ``rel`` = the group's nerr divided by that
    of the 0-rare group of the same model (fold by fold; 1 for the 0 group) with 95 % CI.
    contrast, per set × variant × fraction: the ratio and the fold-paired difference
    nerr(≥3) − nerr(0) with t / Wilcoxon p (Holm within each set)."""
    t = tab.assign(group=novelty_group(tab[n_far_col]))
    per = (t.groupby(['set', 'variant', 'fraction', 'fold', 'group'], observed=False)
            .agg(nerr=(f'nerr_{prop}', 'mean'), n=(f'nerr_{prop}', 'size')).reset_index())
    per['share'] = per['n'] / per.groupby(['set', 'variant', 'fraction', 'fold'])['n'].transform('sum')
    w = per.pivot_table(index=['set', 'variant', 'fraction', 'fold'], columns='group', values='nerr', observed=False)
    rel = w.div(w['0'], axis=0).stack().rename('rel').reset_index()
    per = per.merge(rel, on=['set', 'variant', 'fraction', 'fold', 'group'], how='left')
    rows = []
    for key, g in per.groupby(['set', 'variant', 'fraction', 'group'], observed=True):
        s, sr = fold_summary(g['nerr']), fold_summary(g['rel'])
        rows.append({**dict(zip(['set', 'variant', 'fraction', 'group'], key)), 'share_%': 100 * g['share'].mean(),
                     'molecules_per_fold': g['n'].mean(), 'nerr': s['mean'], 'nerr_lo': s['ci_lo'], 'nerr_hi': s['ci_hi'],
                     'rel': sr['mean'], 'rel_lo': sr['ci_lo'], 'rel_hi': sr['ci_hi'], 'n_folds': s['n_folds']})
    summary = pd.DataFrame(rows)
    zero = summary['group'] == '0'
    summary.loc[zero, ['rel_lo', 'rel_hi']] = 1.0
    rows = []
    for key, g in w.groupby(level=['set', 'variant', 'fraction']):
        s = fold_summary(g['≥3'] - g['0'])
        rows.append({**dict(zip(['set', 'variant', 'fraction'], key)), 'nerr_0': g['0'].mean(), 'nerr_3plus': g['≥3'].mean(),
                     'ratio': g['≥3'].mean() / g['0'].mean(), 'diff': s['mean'], 'diff_lo': s['ci_lo'], 'diff_hi': s['ci_hi'],
                     'p_t': s['p_t'], 'p_wilcoxon': s['p_wilcoxon']})
    contrast = pd.DataFrame(rows)
    for st in contrast['set'].unique():
        m = contrast['set'] == st
        contrast.loc[m, 'p_t_holm'] = holm(contrast.loc[m, 'p_t'])
    return summary, contrast


def within_molecule(rem: pd.DataFrame, dcol: str = 'D_max_15', props=('mean',)) -> pd.DataFrame:
    """The same remainder molecule under the 25 training sets of a fraction: demean its error and
    its distance over the folds, then, per fold, the Spearman ρ of the demeaned values across
    molecules.  Molecule identity (chemistry, size) cancels; what varies is only how far that
    fold's training environments are from the molecule.  Returns per-fold rows."""
    rows = []
    for (v, f), g in rem.groupby(['variant', 'fraction']):
        g = g.copy()
        g['dd'] = g[dcol] - g.groupby('index')[dcol].transform('mean')
        for p in props:
            g[f'de_{p}'] = g[f'nerr_{p}'] - g.groupby('index')[f'nerr_{p}'].transform('mean')
        for fold, h in g.groupby('fold'):
            for p in props:
                rows.append({'variant': v, 'fraction': f, 'fold': fold, 'property': p,
                             'rho_within': float(spearmanr(h[f'de_{p}'], h['dd'])[0])})
    return pd.DataFrame(rows)


def summarise_folds(per_fold: pd.DataFrame, by: Sequence[str], value: str) -> pd.DataFrame:
    rows = []
    for key, g in per_fold.groupby(list(by)):
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), **fold_summary(g[value].to_numpy())})
    return pd.DataFrame(rows)


# The 2 × 2 of the design, molecule-matched: (name, panel with the factor unseen, panel with it seen).
MATCHED_CONTRASTS = (('environment | chemotype held out', 'B', 'A'), ('environment | chemotype seen', 'C', 'D'),
                     ('chemotype | environments seen', 'A', 'D'), ('chemotype | environments unseen', 'B', 'C'))


def matched_contrasts(errors: pd.DataFrame) -> pd.DataFrame:
    """Each factor's effect on the SAME molecules (AIMEl test sets).  ``errors``: one row per (variant,
    fraction, fold, index, panel, property) with ``nerr`` (``eha.molecule_errors``).  For a contrast
    X − Y and every molecule scored at least once in X and once in Y: its mean nerr in X minus in Y.
    Reported per variant × fraction × property as the relative change (% of the molecules' mean nerr
    in Y), the fold-level mean of nerr(X) − mean nerr(Y) of each fold's X molecules with its t-CI and p,
    and, for comparison, the unmatched relative difference of the two panels' mean nerr."""
    rows = []
    for (variant, frac, prop), g in errors.groupby(['variant', 'fraction', 'property']):
        m = g.groupby(['index', 'panel'])['nerr'].mean().unstack('panel')
        pm = g.groupby('panel')['nerr'].mean()
        for name, x, y in MATCHED_CONTRASTS:
            mm = m[[x, y]].dropna()
            xr = g[g['panel'] == x].merge(mm[y].rename('y_mean'), left_on='index', right_index=True)
            s = fold_summary((xr['nerr'] - xr['y_mean']).groupby(xr['fold']).mean())
            rows.append({'variant': variant, 'fraction': frac, 'property': prop, 'contrast': name, 'n_molecules': len(mm),
                         'matched_%': 100 * float((mm[x] - mm[y]).mean() / mm[y].mean()),
                         'unmatched_%': 100 * float((pm[x] - pm[y]) / pm[y]),
                         'fold_diff': s['mean'], 'fold_lo': s['ci_lo'], 'fold_hi': s['ci_hi'], 'p_t': s['p_t']})
    return pd.DataFrame(rows)


# state of a remainder molecule under one run: 2 · (chemotype held out) + (carries a reserved environment)
_STATE = {'D': 0, 'C': 1, 'A': 2, 'B': 3}
FACTORS = {'environments': ('environment | chemotype held out', 'environment | chemotype seen'),
           'chemotype': ('chemotype | environments seen', 'chemotype | environments unseen')}


def matched_increase_folds(errors: pd.DataFrame, contrasts=MATCHED_CONTRASTS, min_molecules: int = 30,
                           by_property: bool = False) -> pd.DataFrame:
    """Error increase on the same molecules, per fold, on the AIMEl test sets (the quantity of the
    same-molecule figure).  For a variant × fraction × property and a contrast X − Y (X = the panel
    with the factor unseen, Y = the panel with it seen):

      1. each molecule's reference error is its mean ``nerr`` over the folds where it sits in Y;
      2. in fold k, over the molecules of panel X that have a reference,
             increase_k = 100 · Σ_i (nerr_ik − ref_i) / Σ_i ref_i ;
      3. the four properties are averaged per fold (``by_property=True``: one row per property
         instead).

    Folds with fewer than ``min_molecules`` such molecules are dropped (as the panel R² drops panels
    with < 30 molecules).  One row per (set, variant, fraction, fold, contrast[, property])."""
    rows = []
    for (variant, frac), g0 in errors.groupby(['variant', 'fraction']):
        for name, x, y in contrasts:
            inc, n = {}, None
            for prop, g in g0.groupby('property'):
                ref = g[g['panel'] == y].groupby('index')['nerr'].mean().rename('ref')
                xr = g[g['panel'] == x].join(ref, on='index', how='inner')
                s = xr.groupby('fold')[['nerr', 'ref']].sum()
                inc[prop] = 100 * (s['nerr'] - s['ref']) / s['ref']
                n = xr.groupby('fold').size()
            inc = pd.DataFrame(inc) if by_property else pd.DataFrame({'mean': pd.DataFrame(inc).mean(axis=1)})
            for fold, r in inc.iterrows():
                if n[fold] < min_molecules:
                    continue
                for prop, v in r.items():
                    rows.append({'set': 'test', 'variant': variant, 'fraction': frac, 'fold': int(fold), 'contrast': name,
                                 **({'property': prop} if by_property else {}), 'increase_%': float(v),
                                 'n_molecules': int(n[fold])})
    return pd.DataFrame(rows)


def remainder_matched_increase_folds(runs_dir=RUNS_DIR, scales=None, contrasts=MATCHED_CONTRASTS) -> pd.DataFrame:
    """The same quantity on the QM9 remainder, where every run predicts every molecule.  In run k a
    molecule is in state A, B, C or D according to whether its chemotype is held out in k and whether
    it carries one of k's reserved environments; the 25 runs of a variant × fraction play the part of
    the folds.  Molecules whose chemotype has no AIMEl molecule (``chemotype == 'never_in_aimel'``) are
    left out, because their chemotype is never seen.  Two passes over the runs: the per-molecule
    references, then each run's increase."""
    scales = scales or load_scales()
    runs = list(_run_dirs(runs_dir))

    def _load(d):
        r = pd.read_pickle(os.path.join(d, 'remainder_preds.pkl'))
        state = 2 * (r['chemotype'] == 'heldout').to_numpy(int) + r['reserved_env'].to_numpy(int)
        state[(r['chemotype'] == 'never_in_aimel').to_numpy()] = -1
        ne = np.stack([np.abs(r[f'pred_{p}'] - r[f'target_{p}']).to_numpy() / scales[p] for p in PROPS])
        return r['index'].to_numpy(), state, ne

    acc, ids = {}, None
    for variant, fold, frac, d in runs:
        idx, state, ne = _load(d)
        if ids is None:
            ids = idx
        elif not np.array_equal(ids, idx):
            raise AssertionError(f'{d}: remainder rows in a different order')
        tot, cnt = acc.setdefault((variant, frac), (np.zeros((4, len(PROPS), len(ids))), np.zeros((4, len(ids)))))
        for k in range(4):
            m = state == k
            cnt[k, m] += 1
            tot[k][:, m] += ne[:, m]
    rows = []
    for variant, fold, frac, d in runs:
        _, state, ne = _load(d)
        tot, cnt = acc[(variant, frac)]
        for name, x, y in contrasts:
            sel = (state == _STATE[x]) & (cnt[_STATE[y]] > 0)
            ref = tot[_STATE[y]][:, sel] / cnt[_STATE[y]][sel]
            inc = 100 * (ne[:, sel] - ref).sum(axis=1) / ref.sum(axis=1)
            rows.append({'set': 'remainder', 'variant': variant, 'fraction': frac, 'fold': fold, 'contrast': name,
                         'increase_%': float(inc.mean()), 'n_molecules': int(sel.sum())})
    return pd.DataFrame(rows)


def factor_increase_folds(per_fold: pd.DataFrame, factors: Optional[dict] = None) -> pd.DataFrame:
    """One value per factor and fold: the mean of the factor's two contrasts (environments: B − A
    and C − D; chemotype: A − D and B − C), or the one available when the other was dropped."""
    factors = factors or FACTORS
    name_of = {c: f for f, cs in factors.items() for c in cs}
    t = per_fold.assign(factor=per_fold['contrast'].map(name_of)).dropna(subset=['factor'])
    return (t.groupby(['set', 'variant', 'fraction', 'fold', 'factor'], as_index=False)['increase_%'].mean()
             .rename(columns={'factor': 'contrast'}))


def summarise_increase(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Fold-level mean of ``increase_%`` with its 95 % t-interval and one-sample p, per set × variant
    × fraction × contrast (or factor)."""
    return summarise_folds(per_fold, ['set', 'variant', 'fraction', 'contrast'], 'increase_%')


def descriptor_effect(errors: pd.DataFrame) -> pd.DataFrame:
    """Informed vs blind on the same test molecules and folds: relative change of the mean nerr
    (% of blind's), per fraction × property."""
    w = errors.pivot_table(index=['fraction', 'fold', 'index', 'panel', 'property'], columns='variant', values='nerr')
    m = w.groupby(['fraction', 'property'])[['informed', 'blind']].mean()
    return (100 * (m['informed'] - m['blind']) / m['blind']).rename('informed_vs_blind_%').reset_index()


def panel_variance(runs_dir=RUNS_DIR, variant: str = 'blind', fraction: float = 1.0) -> pd.DataFrame:
    """Per panel and property (fold means): the panel's target variance and its MSE, each relative to
    the pooled test set's.  R² = 1 − MSE / variance, so a panel loses R² both from a larger error and
    from a narrower target range."""
    rows = []
    for v, fold, frac, d in _run_dirs(runs_dir):
        if v != variant or frac != fraction:
            continue
        t = pd.read_pickle(os.path.join(d, 'test_preds.pkl'))
        for prop in PROPS:
            y, e2 = t[f'target_{prop}'], (t[f'pred_{prop}'] - t[f'target_{prop}']) ** 2
            for panel, g in t.groupby('panel'):
                rows.append({'panel': panel, 'property': prop, 'fold': fold,
                             'variance_ratio': g[f'target_{prop}'].var() / y.var(),
                             'mse_ratio': e2[g.index].mean() / e2.mean()})
    return pd.DataFrame(rows).groupby(['panel', 'property'])[['variance_ratio', 'mse_ratio']].mean().reset_index()


def within_molecule_curve(tab: pd.DataFrame, dcol: str = 'D_mean_15', n_bins: int = 10,
                          min_models: int = 3) -> pd.DataFrame:
    """Error against distance to the training environments, within molecule.  ``tab``: one row per
    (set, variant, fraction, fold, index) with ``nerr_mean`` and ``dcol`` (from :func:`run_distances`).
    For every molecule predicted by at least ``min_models`` models of a fraction: x = its distance
    under a model minus its mean distance over those models; y = its error under that model as a %
    of its mean error over them.  x is binned in ``n_bins`` quantiles per set × variant × fraction;
    per bin the mean x and the fold-level mean of y with its 95 % t-interval."""
    rows = []
    for (st, v, frac), g in tab.groupby(['set', 'variant', 'fraction']):
        g = g[g.groupby('index')['fold'].transform('size') >= min_models]
        x = g[dcol] - g.groupby('index')[dcol].transform('mean')
        y = 100 * (g['nerr_mean'] / g.groupby('index')['nerr_mean'].transform('mean') - 1)
        b = pd.qcut(x, n_bins, labels=False, duplicates='drop')
        for k, idx in pd.Series(x.index, index=x.index).groupby(b.to_numpy()):
            s = fold_summary(y.loc[idx].groupby(g.loc[idx, 'fold']).mean())
            rows.append({'set': st, 'variant': v, 'fraction': frac, 'bin': int(k), 'x': float(x.loc[idx].mean()),
                         'n': len(idx), 'y': s['mean'], 'lo': s['ci_lo'], 'hi': s['ci_hi']})
    return pd.DataFrame(rows)


def _r2(t: np.ndarray, y: np.ndarray) -> float:
    return float(1.0 - ((y - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def zwitterion_effect(zw_ids, runs_dir=RUNS_DIR) -> pd.DataFrame:
    """How much of each model's error comes from a set of molecules (``zw_ids``, QM9 ids; here the
    charge-separated molecules).  One row per set (AIMEl test / remainder) × run × property: the
    number and share of those molecules, their share of the total absolute error, and R² with and
    without them (``dR2`` = R²(without) − R²(all))."""
    zw = set(int(i) for i in zw_ids)
    rows = []
    for variant, fold, frac, d in _run_dirs(runs_dir):
        for st, fn in (('test', 'test_preds.pkl'), ('remainder', 'remainder_preds.pkl')):
            r = pd.read_pickle(os.path.join(d, fn))
            z = r['index'].astype(np.int64).isin(zw).to_numpy()
            for prop in PROPS:
                t, y = r[f'target_{prop}'].to_numpy(float), r[f'pred_{prop}'].to_numpy(float)
                ae = np.abs(y - t)
                rows.append({'set': st, 'variant': variant, 'fraction': frac, 'fold': fold, 'property': prop,
                             'n': len(r), 'n_zw': int(z.sum()), 'zw_%_of_molecules': 100 * float(z.mean()),
                             'zw_%_of_abs_err': 100 * float(ae[z].sum() / ae.sum()),
                             'R2': _r2(t, y), 'R2_without_zw': _r2(t[~z], y[~z])})
    out = pd.DataFrame(rows)
    out['dR2'] = out['R2_without_zw'] - out['R2']
    return out


def zwitterion_summary(per_run: pd.DataFrame, value: str) -> pd.DataFrame:
    """Fold-level mean of ``value`` per set × variant × fraction (rows) and property (columns), plus
    the mean over properties."""
    t = per_run.pivot_table(index=['set', 'variant', 'fraction'], columns='property', values=value, aggfunc='mean')
    t = t[list(PROPS)]
    t['mean over properties'] = t.mean(axis=1)
    return t.sort_index(level=['set', 'variant', 'fraction'], ascending=[False, True, True])


# ── plots ─────────────────────────────────────────────────────────────────────
# Every plotting function takes the displayed text and the font settings as explicit arguments, so the
# notebook can set them for the paper:  fonts = {'title', 'label', 'tick', 'legend', 'star'} sizes in pt (default:
# those of QT-Net's published figures, analysis/inference/analyse_inference.py), title_weight / label_weight.

DEFAULT_FONTS = {'title': 18, 'label': 16, 'tick': 12, 'legend': 12, 'star': 18}
DEFAULT_PANEL_TITLES = {p: f'{p}\n' + '\n'.join(w[0].upper() + w[1:] for w in PANEL_DESC[p].split(', '))
                        for p in PANELS}
DEFAULT_GROUP_LABELS = {'0': 'No rare atom', '1–2': '1–2 rare atoms', '≥3': '≥3 rare atoms'}


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


def _frac_axis(ax, tick_size, fmt: str = '{x:g}'):
    """Log x axis with a tick at each training fraction; ``fmt`` labels them ({x}: the fraction, {p}: in %;
    '{p:g}' keeps 0.05 and 0.1 apart in narrow panels)."""
    from matplotlib.ticker import FuncFormatter, NullLocator
    ax.set_xscale('log')
    ax.set_xticks(FRACTIONS)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt.format(x=x, p=100 * x)))
    ax.set_xlim(0.0075, 1.35)
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.grid(alpha=0.3, lw=0.5)


def _variant_lines(ax, tab, variant_labels, variant_colors, dodge=0.04):
    """informed / blind fold-level means with 95 % CI error bars, slightly dodged in x."""
    for k, v in enumerate(VARIANTS):
        x = tab['fraction'].to_numpy() * (1 + dodge * (k - 0.5))
        y = tab[v].to_numpy()
        ax.errorbar(x, y, yerr=[y - tab[f'{v}_lo'], tab[f'{v}_hi'] - y], color=variant_colors[v], marker='o',
                    ms=4.5, capsize=2.5, lw=1.6, label=variant_labels[v])


def _star_row(ax, fractions, mask, color, y, size):
    for f, m in zip(fractions, mask):
        if m:
            ax.text(f, y, '*', ha='center', va='center', fontsize=size, color=color, transform=ax.get_xaxis_transform())


def _headroom(ax, frac=0.14):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def plot_r2_curves(tables: Dict[str, pd.DataFrame], *, props: Sequence[str] = PROPS,
                   prop_titles: Optional[dict] = None, row_labels: Optional[dict] = None,
                   ylabel: str = r'$R^2$', xlabel: str = 'Training fraction',
                   variant_labels: Optional[dict] = None, variant_colors: Optional[dict] = None,
                   fonts: Optional[dict] = None, title_weight: str = 'normal', label_weight: str = 'normal',
                   legend_loc: str = 'lower right', xtick_fmt: str = '{x:g}', figsize=None, save_path=None):
    """R² vs training fraction, informed vs blind, one column per property and one row per test set
    (``tables``: key -> :func:`informed_vs_blind` output; ``row_labels`` maps keys to the row
    label).  Fold-level mean ± 95 % t-CI (n = 25); * = the table's ``verdict`` (Holm-adjusted paired-t
    p < 0.05 by default, Tukey HSD with ``verdict_by='p_tukey'``), in the colour of the better variant."""
    import matplotlib.pyplot as plt
    _style()
    f = _fonts(fonts)
    prop_titles = {**PROP_LABELS, **(prop_titles or {})}
    row_labels = {k: k for k in tables} | (row_labels or {})
    variant_labels = {v: v for v in VARIANTS} | (variant_labels or {})
    variant_colors = {**VARIANT_COLORS, **(variant_colors or {})}
    fig, axes = plt.subplots(len(tables), len(props), figsize=figsize or (4.0 * len(props), 3.9 * len(tables)),
                             squeeze=False, sharex=True)
    for i, (key, tab) in enumerate(tables.items()):
        for j, prop in enumerate(props):
            ax = axes[i, j]
            s = tab[tab['property'] == prop].sort_values('fraction')
            _variant_lines(ax, s, variant_labels, variant_colors)
            _frac_axis(ax, f['tick'], xtick_fmt)
            _headroom(ax)
            for v in VARIANTS:
                _star_row(ax, s['fraction'], s['verdict'] == v, variant_colors[v], 0.95, f['star'])
            if i == 0:
                ax.set_title(prop_titles[prop], fontsize=f['title'], fontweight=title_weight)
            if j == 0:
                ax.set_ylabel(f'{row_labels[key]}\n{ylabel}', fontsize=f['label'], fontweight=label_weight)
            if i == len(tables) - 1:
                ax.set_xlabel(xlabel, fontsize=f['label'], fontweight=label_weight)
    axes[0, 0].legend(frameon=False, loc=legend_loc, fontsize=f['legend'])
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_panel_delta(summary: pd.DataFrame, *, panels: Sequence[str] = PANELS, panel_titles: Optional[dict] = None,
                     ylabel: str = r'$R^2_\mathrm{panel} - R^2_\mathrm{test}$ (mean over properties)',
                     xlabel: str = 'Training fraction', variant_labels: Optional[dict] = None,
                     variant_colors: Optional[dict] = None, fonts: Optional[dict] = None, title_weight: str = 'normal',
                     label_weight: str = 'normal', legend_loc: str = 'lower left', xtick_fmt: str = '{x:g}',
                     figsize=None, save_path=None):
    """Per panel: mean-over-properties R² of the panel minus that of the pooled test set, informed vs
    blind vs training fraction; fold-level mean ± 95 % t-CI.  Stars: the panel differs from the
    pooled test set (one-sample t, Holm over the 32 variant × panel × fraction cells), one row of
    stars per variant in its colour (informed on top)."""
    import matplotlib.pyplot as plt
    _style()
    f = _fonts(fonts)
    panel_titles = {**DEFAULT_PANEL_TITLES, **(panel_titles or {})}
    variant_labels = {v: v for v in VARIANTS} | (variant_labels or {})
    variant_colors = {**VARIANT_COLORS, **(variant_colors or {})}
    fig, axes = plt.subplots(1, len(panels), figsize=figsize or (4.0 * len(panels), 5.2), sharey=True, squeeze=False)
    for ax, panel in zip(axes[0], panels):
        s = summary[summary['panel'] == panel].sort_values('fraction')
        ax.axhline(0, color='k', lw=0.8)
        _variant_lines(ax, s, variant_labels, variant_colors)
        _frac_axis(ax, f['tick'], xtick_fmt)
        ax.set_title(panel_titles[panel], fontsize=f['title'], fontweight=title_weight)
        ax.set_xlabel(xlabel, fontsize=f['label'], fontweight=label_weight)
    _headroom(axes[0, 0], 0.22)
    for ax, panel in zip(axes[0], panels):
        s = summary[summary['panel'] == panel].sort_values('fraction')
        for v, y in zip(VARIANTS, (0.95, 0.88)):
            _star_row(ax, s['fraction'], s[f'{v}_vs_pooled'] != 'n.s.', variant_colors[v], y, f['star'])
    axes[0, 0].set_ylabel(ylabel, fontsize=f['label'], fontweight=label_weight)
    axes[0, 0].legend(frameon=False, loc=legend_loc, fontsize=f['legend'])
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_novelty_groups(summary: pd.DataFrame, *, value: str = 'rel', layout: str = '2x2',
                        sets: Sequence = (('test', 'AIMEl test'), ('remainder', 'QM9 remainder')),
                        variant_labels: Optional[dict] = None, group_colors: Optional[dict] = None,
                        group_labels: Optional[dict] = None, title_fmt: str = '{variant} — {set}',
                        ylabel: Optional[str] = None, xlabel: str = 'Training fraction',
                        fonts: Optional[dict] = None, title_weight: str = 'normal', label_weight: str = 'normal',
                        legend_loc: str = 'upper left', sharey: bool = True, xtick_fmt: str = '{x:g}', figsize=None,
                        save_path=None):
    """Error of molecules with 0 / 1–2 / ≥3 atoms rare with respect to the model's training set, vs
    training fraction; fold-level mean ± 95 % t-CI, linear axis.

    value  : 'rel'  = error relative to the no-rare group of the same model (that group sits at 1);
             'nerr' = absolute error ÷ the target's AIMEl standard deviation, mean over properties.
    layout : '2x2' (rows = test sets, columns = variants) or '1x4' (one row).
    """
    import matplotlib.pyplot as plt
    _style()
    f = _fonts(fonts)
    variant_labels = {v: v for v in VARIANTS} | (variant_labels or {})
    group_colors = {**NOVELTY_COLORS, **(group_colors or {})}
    group_labels = {**DEFAULT_GROUP_LABELS, **(group_labels or {})}
    if ylabel is None:
        ylabel = ('Error relative to molecules\nwith no rare atom' if value == 'rel'
                  else 'Error / std of the target\n(mean over properties)')
    cells = [(st, st_name, v) for st, st_name in sets for v in VARIANTS]
    nrow, ncol = (len(sets), len(VARIANTS)) if layout == '2x2' else (1, len(cells))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize or (5.4 * ncol, 4.6 * nrow), sharex=True, sharey=sharey,
                             squeeze=False)
    for n, (st, st_name, v) in enumerate(cells):
        ax = axes.ravel()[n]
        s = summary[(summary['set'] == st) & (summary['variant'] == v)]
        if value == 'rel':
            ax.axhline(1.0, color='k', lw=0.8)
        for k, grp in enumerate(NOVELTY_GROUPS):
            g = s[s['group'] == grp].sort_values('fraction')
            x = g['fraction'].to_numpy() * (1 + 0.04 * (k - 1))
            y = g[value].to_numpy()
            ax.errorbar(x, y, yerr=[y - g[f'{value}_lo'], g[f'{value}_hi'] - y], color=group_colors[grp], marker='o',
                        ms=4.5, capsize=2.5, lw=1.6, label=group_labels[grp])
        _frac_axis(ax, f['tick'], xtick_fmt)
        ax.set_title(title_fmt.format(variant=variant_labels[v], set=st_name), fontsize=f['title'], fontweight=title_weight)
        if n % ncol == 0:
            ax.set_ylabel(ylabel, fontsize=f['label'], fontweight=label_weight)
        if n // ncol == nrow - 1:
            ax.set_xlabel(xlabel, fontsize=f['label'], fontweight=label_weight)
    axes.ravel()[0].legend(frameon=False, loc=legend_loc, fontsize=f['legend'])
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# same-molecule figure: one colour per factor, coolwarm's ends as in the novelty figures (unseen environments red,
# unseen chemotype blue); within a factor, its two contrasts differ by line style and marker
DEFAULT_FACTOR_COLORS = {'environments': '#b40426', 'chemotype': '#3b4cc0'}
DEFAULT_FACTOR_LABELS = {'environments': 'Environments unseen', 'chemotype': 'Chemotype unseen'}
DEFAULT_CONTRAST_LABELS = {'environment | chemotype held out': 'Environments, chemotype held out (B − A)',
                           'environment | chemotype seen': 'Environments, chemotype seen (C − D)',
                           'chemotype | environments seen': 'Chemotype, environments seen (A − D)',
                           'chemotype | environments unseen': 'Chemotype, environments unseen (B − C)'}


def plot_matched_increase(summary: pd.DataFrame, *, level: str = 'factor', variants: Sequence[str] = ('blind', 'informed'),
                          sets: Sequence = (('test', 'AIMEl test'), ('remainder', 'QM9 remainder')),
                          variant_labels: Optional[dict] = None, factor_colors: Optional[dict] = None,
                          factor_labels: Optional[dict] = None, contrast_labels: Optional[dict] = None,
                          set_styles: Optional[dict] = None, legend_fmt: str = '{factor}, {set}', dodge: float = 0.035,
                          title_fmt: Optional[str] = None, ylabel: str = 'Error increase on the\nsame molecules (%)',
                          xlabel: str = 'Training fraction', fonts: Optional[dict] = None, title_weight: str = 'normal',
                          label_weight: str = 'normal', legend_loc: str = 'upper left', sharey: bool = True,
                          xtick_fmt: str = '{x:g}', marker_size: float = 4.5, open_markers: Sequence[str] = (),
                          split_legend: bool = False, set_legend_labels: Optional[dict] = None,
                          set_legend_loc: str = 'upper left', legend_handlelength: float = 2.0,
                          figsize=None, save_path=None):
    """Error increase on the same molecules (:func:`summarise_increase`) against the training
    fraction; fold-level mean ± 95 % t-interval, linear y axis, 0 = no effect.

    level='factor'   : ``summary`` of :func:`factor_increase_folds`; one panel per variant (``variants``,
                       left to right), one line per factor (colour) and test set (``set_styles``:
                       set -> (line style, marker); marker '' = none; sets in ``open_markers`` get white-filled
                       markers); legend entries ``legend_fmt`` ({factor}, {set}).  ``split_legend``: two legends
                       instead, the factors (colour) in the first panel and the test sets (black line style,
                       ``set_legend_labels``) in the last, at ``set_legend_loc``.
    level='contrast' : ``summary`` of the four contrasts; rows = test sets, columns = variants; colour =
                       the factor, solid / dashed = its two contrasts (``contrast_labels``).
    ``title_fmt=''`` leaves the panels without titles (name them in the caption)."""
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    _style()
    f = _fonts(fonts)
    variant_labels = {v: v for v in VARIANTS} | (variant_labels or {})
    factor_colors = {**DEFAULT_FACTOR_COLORS, **(factor_colors or {})}
    factor_labels = {**DEFAULT_FACTOR_LABELS, **(factor_labels or {})}
    contrast_labels = {**DEFAULT_CONTRAST_LABELS, **(contrast_labels or {})}
    set_styles = {'test': ('-', 'o'), 'remainder': ('--', 's'), **(set_styles or {})}
    factor_of = {c: fa for fa, cs in FACTORS.items() for c in cs}

    def _line(ax, s, color, ls, marker, k, n, label, mfc=None):
        s = s.sort_values('fraction')
        x = s['fraction'].to_numpy() * (1 + dodge * (k - (n - 1) / 2))
        y = s['mean'].to_numpy()
        ax.errorbar(x, y, yerr=[y - s['ci_lo'], s['ci_hi'] - y], color=color, ls=ls, marker=marker, ms=marker_size,
                    mfc=mfc, capsize=2.5, lw=1.6, label=label)

    if level == 'factor':
        cells = [(None, None, v) for v in variants]
        nrow, ncol = 1, len(variants)
        title_fmt = '{variant}' if title_fmt is None else title_fmt
    elif level == 'contrast':
        cells = [(st, st_name, v) for st, st_name in sets for v in variants]
        nrow, ncol = len(sets), len(variants)
        title_fmt = '{variant} — {set}' if title_fmt is None else title_fmt
    else:
        raise ValueError(level)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize or (6.2 * ncol, 5.0 * nrow), sharex=True, sharey=sharey,
                             squeeze=False)
    for n, (st, st_name, v) in enumerate(cells):
        ax = axes.ravel()[n]
        ax.axhline(0, color='k', lw=0.8)
        sv = summary[summary['variant'] == v]
        if level == 'factor':
            lines = [(fa, s_, s_name) for fa in FACTORS for s_, s_name in sets]
            for k, (fa, s_, s_name) in enumerate(lines):
                ls, mk = set_styles[s_]
                _line(ax, sv[(sv['set'] == s_) & (sv['contrast'] == fa)], factor_colors[fa], ls, mk, k, len(lines),
                      legend_fmt.format(factor=factor_labels[fa], set=s_name),
                      mfc='white' if s_ in open_markers else None)
        else:
            for k, c in enumerate(MATCHED_CONTRASTS):
                name = c[0]
                ls, mk = ('-', 'o') if name in (FACTORS['environments'][0], FACTORS['chemotype'][0]) else ('--', 's')
                _line(ax, sv[(sv['set'] == st) & (sv['contrast'] == name)], factor_colors[factor_of[name]], ls, mk,
                      k, len(MATCHED_CONTRASTS), contrast_labels[name])
        _frac_axis(ax, f['tick'], xtick_fmt)
        if title_fmt:
            ax.set_title(title_fmt.format(variant=variant_labels[v], set=st_name), fontsize=f['title'],
                         fontweight=title_weight)
        if n % ncol == 0:
            ax.set_ylabel(ylabel, fontsize=f['label'], fontweight=label_weight)
        if n // ncol == nrow - 1:
            ax.set_xlabel(xlabel, fontsize=f['label'], fontweight=label_weight)
    if split_legend and level == 'factor':
        set_names = dict(sets) | (set_legend_labels or {})
        h_fac = [Line2D([0], [0], color=factor_colors[fa], lw=2.2, label=factor_labels[fa]) for fa in FACTORS]
        h_set = [Line2D([0], [0], color='k', lw=1.6, ls=set_styles[s_][0], marker=set_styles[s_][1] or None,
                        ms=marker_size, mfc='white' if s_ in open_markers else None, label=set_names[s_])
                 for s_, _ in sets]
        axes.ravel()[0].legend(handles=h_fac, frameon=False, loc=legend_loc, fontsize=f['legend'],
                               handlelength=legend_handlelength)
        axes.ravel()[-1].legend(handles=h_set, frameon=False, loc=set_legend_loc, fontsize=f['legend'],
                                handlelength=legend_handlelength)
    else:
        axes.ravel()[0].legend(frameon=False, loc=legend_loc, fontsize=f['legend'], handlelength=legend_handlelength)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# within-molecule figure, one colour ramp per test set (smallest -> largest training fraction), the set colours of
# the t-SNE figures: AIMEl in 'Blues', the QM9 remainder in 'Reds' (blue vs red also reads for colour-blind readers)
DEFAULT_SET_COLORS = {'test': ['#7fb9da', '#3b8bc2', '#135fa7', '#08306b'],          # Blues(0.45, 0.65, 0.82, 1)
                      'remainder': ['#fb7a5a', '#e83429', '#b61319', '#67000d']}     # Reds(0.45, 0.65, 0.82, 1)


def plot_within_molecule_by_variant(curve: pd.DataFrame, *, variants: Sequence[str] = ('blind', 'informed'),
                                    sets: Sequence = (('test', 'AIMEl test'), ('remainder', 'QM9 remainder')),
                                    variant_labels: Optional[dict] = None, set_colors: Optional[dict] = None,
                                    set_styles: Optional[dict] = None, ci: str = 'bar', band_alpha: float = 0.10,
                                    dodge: float = 0.004, title_fmt: str = '{variant}',
                                    xlabel: str = "Distance to the model's training environments,\nvs the molecule's average (Δ log)",
                                    ylabel: str = "Error vs the molecule's\naverage over models (%)",
                                    legend_fmt: str = '{set}, training fraction', fonts: Optional[dict] = None,
                                    title_weight: str = 'normal', label_weight: str = 'normal',
                                    legend_loc: str = 'upper left', figsize=(13, 5.8), save_path=None):
    """:func:`within_molecule_curve` with one panel per variant (``variants``, left to right) and, in
    each, one line per test set × training fraction: each set has its own colour ramp over the
    fractions (``set_colors``: set -> four colours, smallest to largest fraction) and line style
    (``set_styles``).  95 % CI as bands (``ci='band'``) or error bars (``ci='bar'``, the two sets
    shifted by ``dodge`` in x so the bars do not sit on top of each other)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _style()
    f = _fonts(fonts)
    variant_labels = {v: v for v in VARIANTS} | (variant_labels or {})
    set_colors = {**DEFAULT_SET_COLORS, **(set_colors or {})}
    set_styles = {'test': '-', 'remainder': '--', **(set_styles or {})}
    fig, axes = plt.subplots(1, len(variants), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for ax, v in zip(axes[0], variants):
        for j, (st, _) in enumerate(sets):
            for k, frac in enumerate(FRACTIONS):
                c = curve[(curve['set'] == st) & (curve['variant'] == v) & (curve['fraction'] == frac)].sort_values('x')
                col, ls = set_colors[st][k], set_styles[st]
                if ci == 'band':
                    ax.fill_between(c['x'], c['lo'], c['hi'], color=col, alpha=band_alpha, lw=0)
                    ax.plot(c['x'], c['y'], color=col, ls=ls, lw=1.8)
                else:
                    x = c['x'] + dodge * (j - (len(sets) - 1) / 2)
                    ax.errorbar(x, c['y'], yerr=[c['y'] - c['lo'], c['hi'] - c['y']], color=col, ls=ls, lw=1.6,
                                marker='o' if j == 0 else 's', ms=3.5, capsize=2, elinewidth=1)
        ax.axhline(0, color='k', lw=0.8)
        ax.axvline(0, color='k', lw=0.5, ls=':')
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=f['tick'])
        ax.set_title(title_fmt.format(variant=variant_labels[v]), fontsize=f['title'], fontweight=title_weight)
        ax.set_xlabel(xlabel, fontsize=f['label'], fontweight=label_weight)
    axes[0, 0].set_ylabel(ylabel, fontsize=f['label'], fontweight=label_weight)
    for ax, (st, st_name) in zip(axes[0], sets):
        h = [Line2D([0], [0], color=set_colors[st][k], ls=set_styles[st], lw=2.2, label=f'{fr:g}')
             for k, fr in enumerate(FRACTIONS)]
        ax.legend(handles=h, title=legend_fmt.format(set=st_name), frameon=False, loc=legend_loc,
                  fontsize=f['legend'], title_fontsize=f['legend'])
    fig.tight_layout()
    _save(fig, save_path)
    return fig

