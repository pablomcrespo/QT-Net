"""
Analysis of the environment-holdout study → small CSV tables for the figures.

Inputs (written by the two scripts of the study):
  experiments/molecular/env_holdout/<variant>/fold_<f>/frac_<x>/test_preds.pkl
      AIMEl test panels A/B/C/D of each of the 25 folds
      (scripts/molecular/jax/train_molecular_env_holdout.py)
  experiments/molecular/env_holdout/<variant>/fold_<f>/frac_<x>/remainder_preds.pkl
      the same models on the 98,096-molecule QM9 remainder (same run directory)
      (scripts/inference/predict_env_holdout_from_inferred.py)

Tables (analysis/molecular/env_holdout_results/):
  aimel_fold_metrics.csv       R2/MAE/RMSE/CCC/Spearman per variant × fraction × fold × panel ×
                               property, in the schema of analysis_molecular.precompute_metrics_mol
                               (``method`` = '<variant>_<fraction>', ``cv_cycle`` = fold), so its
                               paired_informed_vs_blind_table / tukey_ci_mol apply per panel
  aimel_env_contrast.csv       the molecule-matched environment effect: each molecule's error
                               with its environments unseen (panel B) minus its mean error when
                               they were seen (panel A of the other folds), chemotype held out in
                               both, training size equal — per variant × fraction × property,
                               as fold-level means (n = 25) and a molecule bootstrap
  remainder_metrics.csv        deployment metrics per variant × fraction × fold × chemotype
                               stratum × reserved-environment status
  remainder_env_contrast.csv   within-molecule: mean error over the folds that reserved the
                               molecule's environments minus over the folds that did not, matched
                               on chemotype status; molecule bootstrap and fold-level mean ± SE
  remainder_dose_response.csv  error vs training support of the molecule's rarest environment

Errors are expressed both in physical units and normalised by the property's standard deviation
over AIMEl, so properties can be averaged.

Usage:
    python analysis/molecular/env_holdout_analysis.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'analysis', 'atomic'))

import result_analysis as ra  # noqa: E402  (QT-Net: analysis/atomic/result_analysis.py)

PROPS = ['alpha', 'gap', 'U0', 'Cv']
VARIANTS = ['informed', 'blind']
PANELS = ['A', 'B', 'C', 'D']
RUNS_DIR = os.path.join(REPO_ROOT, 'experiments', 'molecular', 'env_holdout')
OUT_DIR = os.path.join(HERE, 'env_holdout_results')
AIMEL_PKL = os.path.join(REPO_ROOT, 'data_curation', 'molecular', 'aimel_clustered_molecular.pkl')
# Training atoms of the molecule's rarest labelled environment; -1 = every atom is HDBSCAN noise
# (no labelled environment), kept apart from a genuine zero.
SUPPORT_BINS = [-1.5, -0.5, 0, 9, 99, 999, np.inf]
SUPPORT_LABELS = ['no labelled atom', '0', '1-9', '10-99', '100-999', '>=1000']


def property_scales(props=PROPS) -> dict:
    """Standard deviation of each target over AIMEl (the normalisation of errors)."""
    df = pd.read_pickle(AIMEL_PKL)
    return {p: float(df[p].std()) for p in props}


def _runs(base, name):
    for p in sorted(glob.glob(os.path.join(base, '*', 'fold_*', 'frac_*', name))):
        variant, fold, frac = p.split(os.sep)[-4:-1]
        yield variant, int(fold.split('_')[1]), float(frac.split('_')[1]), p


# ── AIMEl ─────────────────────────────────────────────────────────────────────

def aimel_fold_metrics(runs_dir=RUNS_DIR, min_mols=30) -> pd.DataFrame:
    """precompute_metrics_mol schema, one row per (method, fold, panel, property);
    panel 'all' is the whole test set.  Cells below ``min_mols`` molecules are dropped."""
    rows = []
    for variant, fold, frac, p in _runs(runs_dir, 'test_preds.pkl'):
        df = pd.read_pickle(p)
        for panel, sub in [('all', df)] + list(df.groupby('panel')):
            if len(sub) < min_mols:
                continue
            for prop in PROPS:
                rec = {'method': f'{variant}_{frac}', 'variant': variant, 'fraction': frac,
                       'cv_cycle': fold, 'repeat': fold // 5, 'split': 'test', 'property': prop,
                       'panel': panel, 'n_mols': len(sub)}
                rec.update(ra._compute_metrics_dict(sub[f'target_{prop}'].to_numpy(float),
                                                    sub[f'pred_{prop}'].to_numpy(float)))
                rows.append(rec)
    return pd.DataFrame(rows)


def molecule_errors(runs_dir=RUNS_DIR, scales=None) -> pd.DataFrame:
    """Long table: variant, fraction, fold, index, panel, property, err (physical), nerr (scaled)."""
    scales = scales or property_scales()
    parts = []
    for variant, fold, frac, p in _runs(runs_dir, 'test_preds.pkl'):
        df = pd.read_pickle(p)
        for prop in PROPS:
            e = (df[f'pred_{prop}'] - df[f'target_{prop}']).abs().to_numpy()
            parts.append(pd.DataFrame({'variant': variant, 'fraction': frac, 'fold': fold,
                                       'index': df['index'].to_numpy(), 'panel': df['panel'].to_numpy(),
                                       'property': prop, 'err': e, 'nerr': e / scales[prop]}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _bootstrap_mean(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aimel_env_contrast(errors: pd.DataFrame) -> pd.DataFrame:
    """Molecule-matched environment effect on AIMEl.

    For each (variant, fraction, property) and each molecule scored at least once in B (its
    environments unseen) and at least once in A (seen), chemotype held out both times:
        d = mean nerr over its B predictions − mean nerr over its A predictions.
    Reported as (i) the mean over molecules with a molecule-bootstrap 95 % CI and (ii) the
    fold-level mean (each fold's B molecules, n ≤ 25 folds) with its t-interval.
    """
    rows = []
    ab = errors[errors.panel.isin(['A', 'B'])]
    for (variant, frac, prop), g in ab.groupby(['variant', 'fraction', 'property']):
        m = g.groupby(['index', 'panel'])['nerr'].mean().unstack('panel').dropna()
        if m.empty:
            continue
        d = (m['B'] - m['A'])
        lo, hi = _bootstrap_mean(d.to_numpy())
        b_rows = g[g.panel == 'B'].merge(m['A'].rename('a_mean'), left_on='index', right_index=True)
        per_fold = (b_rows['nerr'] - b_rows['a_mean']).groupby(b_rows['fold']).mean()
        se = per_fold.std(ddof=1) / np.sqrt(len(per_fold)) if len(per_fold) > 1 else np.nan
        rows.append({'variant': variant, 'fraction': frac, 'property': prop,
                     'n_molecules': len(d), 'mean_nerr_seen': float(m['A'].mean()),
                     'mean_nerr_unseen': float(m['B'].mean()), 'delta_env': float(d.mean()),
                     'delta_env_ci_lo': lo, 'delta_env_ci_hi': hi,
                     'n_folds': len(per_fold), 'delta_env_fold_mean': float(per_fold.mean()),
                     'delta_env_fold_se': float(se)})
    return pd.DataFrame(rows)


# ── QM9 remainder ─────────────────────────────────────────────────────────────

def remainder_long(runs_dir=RUNS_DIR, scales=None) -> pd.DataFrame:
    scales = scales or property_scales()
    parts = []
    for variant, fold, frac, p in _runs(runs_dir, 'remainder_preds.pkl'):
        df = pd.read_pickle(p)
        base = df[['index', 'chemotype', 'reserved_env', 'min_env_train_atoms',
                   'n_unseen_env_atoms', 'remainder_only_env']]
        for prop in PROPS:
            e = (df[f'pred_{prop}'] - df[f'target_{prop}']).abs().to_numpy()
            parts.append(base.assign(variant=variant, fraction=frac, fold=fold, property=prop,
                                     target=df[f'target_{prop}'].to_numpy(),
                                     pred=df[f'pred_{prop}'].to_numpy(), err=e, nerr=e / scales[prop]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def remainder_metrics(long: pd.DataFrame, min_mols=30) -> pd.DataFrame:
    rows = []
    for key, sub in long.groupby(['variant', 'fraction', 'fold', 'property', 'chemotype', 'reserved_env']):
        if len(sub) < min_mols:
            continue
        rec = dict(zip(['variant', 'fraction', 'fold', 'property', 'chemotype', 'reserved_env'], key))
        rec['n_mols'] = len(sub)
        rec.update(ra._compute_metrics_dict(sub['target'].to_numpy(), sub['pred'].to_numpy()))
        rows.append(rec)
    return pd.DataFrame(rows)


def remainder_env_contrast(long: pd.DataFrame) -> pd.DataFrame:
    """Within molecule: mean nerr over folds where its environments were reserved (unseen)
    minus over folds where they were not (seen), separately for each chemotype status.
    Reported, as on AIMEl, as the molecule mean with a molecule-bootstrap 95 % CI and as the
    fold-level mean (each fold's reserved-environment molecules, n ≤ 25) with its SE: a molecule's
    unseen predictions come from the few models that reserved its environments, so only the
    fold-level interval carries the model-to-model variation."""
    rows = []
    for (variant, frac, prop, chem), g in long.groupby(['variant', 'fraction', 'property', 'chemotype']):
        m = g.groupby(['index', 'reserved_env'])['nerr'].mean().unstack('reserved_env')
        if True not in m or False not in m:
            continue
        m = m.dropna()
        d = m[True] - m[False]
        lo, hi = _bootstrap_mean(d.to_numpy())
        u_rows = g[g.reserved_env].merge(m[False].rename('seen_mean'), left_on='index', right_index=True)
        per_fold = (u_rows['nerr'] - u_rows['seen_mean']).groupby(u_rows['fold']).mean()
        se = per_fold.std(ddof=1) / np.sqrt(len(per_fold)) if len(per_fold) > 1 else np.nan
        rows.append({'variant': variant, 'fraction': frac, 'property': prop, 'chemotype': chem,
                     'n_molecules': len(d), 'mean_nerr_seen': float(m[False].mean()),
                     'mean_nerr_unseen': float(m[True].mean()), 'delta_env': float(d.mean()),
                     'delta_env_ci_lo': lo, 'delta_env_ci_hi': hi,
                     'n_folds': len(per_fold), 'delta_env_fold_mean': float(per_fold.mean()),
                     'delta_env_fold_se': float(se)})
    return pd.DataFrame(rows)


def remainder_dose_response(long: pd.DataFrame) -> pd.DataFrame:
    long = long.assign(support=pd.cut(long['min_env_train_atoms'], SUPPORT_BINS, labels=SUPPORT_LABELS))
    return (long.groupby(['variant', 'fraction', 'property', 'support'], observed=True)
                .agg(n=('nerr', 'size'), mean_nerr=('nerr', 'mean'), median_nerr=('nerr', 'median'))
                .reset_index())


def main():
    if next(_runs(RUNS_DIR, 'test_preds.pkl'), None) is None:
        print(f"no test_preds.pkl under {RUNS_DIR}: this step needs the run outputs, nothing written")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    scales = property_scales()
    tables = {}
    if os.path.isdir(RUNS_DIR):
        tables['aimel_fold_metrics'] = aimel_fold_metrics()
        tables['aimel_env_contrast'] = aimel_env_contrast(molecule_errors(scales=scales))
        long = remainder_long(scales=scales)
        if not long.empty:
            tables['remainder_metrics'] = remainder_metrics(long)
            tables['remainder_env_contrast'] = remainder_env_contrast(long)
            tables['remainder_dose_response'] = remainder_dose_response(long)
    for name, t in tables.items():
        path = os.path.join(OUT_DIR, f'{name}.csv')
        t.to_csv(path, index=False)
        print(f"wrote {path} ({len(t):,} rows)")
    with open(os.path.join(OUT_DIR, 'property_scales.json'), 'w') as f:
        json.dump(scales, f, indent=2)


if __name__ == '__main__':
    main()
