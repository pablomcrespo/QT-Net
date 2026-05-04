#!/usr/bin/env python3
"""
Ensemble inference with pretrained XGBoost molecular property models.

Loads all fold checkpoints at a given training fraction from
experiments/molecular/xgboost/<variant>/ and produces ensemble-averaged
predictions.  SMILES featurization is performed once and shared across
all fold models.

If the input DataFrame contains the target property columns, the script
prints a performance summary covering MAE, MSE, RMSE, R², ECE and MCE
for each property and their macro-average.

ECE / MCE measure regression calibration: for each confidence level p the
ensemble std is used to build a symmetric Gaussian prediction interval, and
we check what fraction of true values falls inside.  ECE is the mean
absolute gap between expected and actual coverage; MCE is the maximum.

Usage:
  python predict_xgboost.py --input-pkl data.pkl --fraction 1.0
  python predict_xgboost.py --input-pkl data.pkl --fraction 0.5 --variant blind \\
      --output-pkl preds.pkl --batch-size 1024
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

from rdkit import RDLogger
import numpy as np
import pandas as pd
from scipy import stats

RDLogger.DisableLog('rdApp.info')

# ---------------------------------------------------------------------------
# Repo / project root — mirrors train_xgboost.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start_dir: Path) -> Path:
    cur = start_dir.resolve()
    while True:
        if (cur / 'data_curation').is_dir():
            return cur
        parent = cur.parent
        if parent == cur:
            return (start_dir / '..' / '..' / '..').resolve()
        cur = parent


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for _root, _dirs, _ in os.walk(str(REPO_ROOT)):
    if 'qtnet' in _dirs:
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break

_proj_candidate = REPO_ROOT.parent
if (
    (_proj_candidate / 'experiments').is_dir()
    or (_proj_candidate / 'pyproject.toml').exists()
):
    PROJECT_ROOT = _proj_candidate
else:
    PROJECT_ROOT = REPO_ROOT

from qtnet.data_utils import MOLECULAR_PROPERTIES_PRED
from qtnet.xgb_models.models import XGBMolPropertyRegressor

MOLECULAR_PROPERTIES: List[str] = MOLECULAR_PROPERTIES_PRED  # alpha, gap, U0, Cv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fold discovery
# ---------------------------------------------------------------------------

def discover_folds(
    experiments_root: Path,
    variant: str,
    fraction: float,
) -> List[Tuple[int, Path]]:
    """Return sorted ``(fold_index, checkpoint_dir)`` for all available folds."""
    variant_dir = experiments_root / variant
    if not variant_dir.is_dir():
        raise FileNotFoundError(
            f"Variant directory not found: {variant_dir}"
        )
    frac_tag = f"frac_{fraction}"
    folds = []
    for fold_dir in variant_dir.iterdir():
        if not fold_dir.is_dir() or not fold_dir.name.startswith('fold_'):
            continue
        try:
            fold_idx = int(fold_dir.name.split('_', 1)[1])
        except ValueError:
            continue
        ckpt_dir = fold_dir / frac_tag / 'checkpoints'
        if (ckpt_dir / 'model.skops').exists():
            folds.append((fold_idx, ckpt_dir))
    return sorted(folds, key=lambda pair: pair[0])


# ---------------------------------------------------------------------------
# Calibration + scalar metrics
# ---------------------------------------------------------------------------

def _calibration_bin_errors(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """Per-bin |actual_coverage - expected_coverage| under Gaussian predictions."""
    y_std_safe = np.maximum(y_std, 1e-8)
    confidence_levels = np.linspace(0.1, 1.0, n_bins)
    errors = np.zeros(n_bins)
    for i, p in enumerate(confidence_levels):
        z = stats.norm.ppf((1.0 + p) / 2.0)
        actual_p = float(np.mean(np.abs(y_true - y_mean) <= z * y_std_safe))
        errors[i] = abs(actual_p - p)
    return errors


def compute_metrics(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> Dict[str, float]:
    """Return MAE, MSE, RMSE, R², ECE, MCE for a single property."""
    mae = float(np.mean(np.abs(y_true - y_mean)))
    mse = float(np.mean((y_true - y_mean) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_mean) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float('nan')
    bin_errors = _calibration_bin_errors(y_true, y_mean, y_std)
    return {
        'MAE':  mae,
        'MSE':  mse,
        'RMSE': rmse,
        'R2':   r2,
        'ECE':  float(bin_errors.mean()),
        'MCE':  float(bin_errors.max()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='XGBoost ensemble inference on a pickled DataFrame.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_xgboost.py --input-pkl data.pkl --fraction 1.0
  python predict_xgboost.py --input-pkl data.pkl --fraction 0.5 \\
      --output-pkl preds.pkl --batch-size 1024
        """,
    )
    parser.add_argument(
        '--input-pkl', type=Path, required=True,
        help='Pickled DataFrame with at least a "smiles" column.',
    )
    parser.add_argument(
        '--fraction', type=float, required=True,
        help='Training fraction used when the models were trained (e.g. 1.0).',
    )
    parser.add_argument(
        '--variant', type=str, default='blind', choices=['blind'],
        help='Model variant (default: blind).',
    )
    parser.add_argument(
        '--experiments-root', type=Path,
        default=PROJECT_ROOT / 'experiments' / 'molecular' / 'xgboost',
        help='Root directory containing fold checkpoints.',
    )
    parser.add_argument(
        '--smiles-col', type=str, default='smiles',
        help='Column name containing SMILES strings (default: smiles).',
    )
    parser.add_argument(
        '--output-pkl', type=Path, default=None,
        help='Optional path to write the predictions DataFrame.',
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size for inference (default: all at once).',
    )
    parser.add_argument(
        '--n-jobs', type=int, default=None,
        help=(
            'Number of XGBoost inference threads. '
            'None (default) sets n_jobs=None on each loaded model, '
            'which lets XGBoost use all available threads.'
        ),
    )
    args = parser.parse_args()

    log.info('=' * 72)
    log.info('XGBoost ensemble inference')
    log.info('  variant=%s  fraction=%s', args.variant, args.fraction)
    log.info('=' * 72)

    # ------------------------------------------------------------------
    # Load DataFrame
    # ------------------------------------------------------------------
    log.info('Loading %s ...', args.input_pkl)
    df = pd.read_pickle(args.input_pkl)
    log.info('  %d molecules', len(df))

    if args.smiles_col not in df.columns:
        raise ValueError(
            f"Column '{args.smiles_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    smiles_list: List[str] = df[args.smiles_col].tolist()

    missing_targets = [p for p in MOLECULAR_PROPERTIES if p not in df.columns]
    has_targets = not missing_targets
    if not has_targets:
        log.warning(
            'Target columns missing (%s); metric computation skipped.',
            missing_targets,
        )

    # ------------------------------------------------------------------
    # Discover fold checkpoints
    # ------------------------------------------------------------------
    folds = discover_folds(args.experiments_root, args.variant, args.fraction)
    if not folds:
        raise FileNotFoundError(
            f"No fold checkpoints found under "
            f"{args.experiments_root / args.variant}/ "
            f"for fraction={args.fraction}. "
            "Check --experiments-root and --fraction."
        )
    log.info(
        'Found %d folds: %s',
        len(folds),
        [fold_idx for fold_idx, _ in folds],
    )

    # ------------------------------------------------------------------
    # Featurize SMILES once using the first model's RDKit settings
    # ------------------------------------------------------------------
    first_fold_idx, first_ckpt_dir = folds[0]
    first_ckpt_path = str(first_ckpt_dir / 'model')
    log.info('Loading fold %d to featurize SMILES ...', first_fold_idx)
    first_model = XGBMolPropertyRegressor.load(first_ckpt_path)

    log.info(
        'Featurizing %d molecules (fp_size=%d, radius=%d, '
        'use_fp=%s, use_descriptors=%s) ...',
        len(smiles_list),
        first_model.fp_size,
        first_model.radius,
        first_model.use_fp,
        first_model.use_descriptors,
    )
    
    X_raw: np.ndarray = first_model.featurize(smiles_list, n_jobs=args.n_jobs)
    log.info('  Raw feature matrix: %s', X_raw.shape)

    # ------------------------------------------------------------------
    # Run inference across all folds (pre-featurized path)
    # ------------------------------------------------------------------
    fold_preds: List[np.ndarray] = []

    for fold_idx, ckpt_dir in folds:
        ckpt_path = str(ckpt_dir / 'model')
        log.info('fold %d: loading checkpoint and predicting ...', fold_idx)
        model = XGBMolPropertyRegressor.load(ckpt_path)
        model.set_n_jobs(args.n_jobs)
        preds = model.predict(X_raw, batch_size=args.batch_size)
        log.info('  predictions shape: %s', preds.shape)
        fold_preds.append(preds)

    # ------------------------------------------------------------------
    # Ensemble statistics
    # ------------------------------------------------------------------
    stack = np.stack(fold_preds, axis=0)   # (n_folds, n_mols, n_props)
    mean_preds = stack.mean(axis=0)        # (n_mols, n_props)
    std_preds = stack.std(axis=0)          # (n_mols, n_props)

    # ------------------------------------------------------------------
    # Build output DataFrame
    # ------------------------------------------------------------------
    out = df.copy()
    tag = f"{args.variant}_{args.fraction}"
    for i, prop in enumerate(MOLECULAR_PROPERTIES):
        out[f'{tag}_pred_{prop}'] = mean_preds[:, i]
        out[f'{tag}_std_{prop}'] = std_preds[:, i]

    if args.output_pkl is not None:
        args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
        out.to_pickle(args.output_pkl)
        log.info('Saved predictions to %s', args.output_pkl)

    # ------------------------------------------------------------------
    # Performance summary
    # ------------------------------------------------------------------
    if not has_targets:
        log.info('No target columns found — skipping metric summary.')
        return

    metric_names = ['MAE', 'MSE', 'RMSE', 'R2', 'ECE', 'MCE']
    all_metrics: Dict[str, Dict[str, float]] = {}

    log.info('')
    log.info('%s', '=' * 72)
    log.info(
        'Performance summary  (%d molecules, %d folds)',
        len(df),
        len(folds),
    )
    log.info('%s', '=' * 72)
    header = f"  {'prop':>6} | " + '  '.join(f'{k:>8}' for k in metric_names)
    log.info(header)
    log.info('  %s', '-' * (len(header) - 2))

    for i, prop in enumerate(MOLECULAR_PROPERTIES):
        y_true = df[prop].to_numpy(dtype=np.float64)
        y_mean = mean_preds[:, i].astype(np.float64)
        y_std = std_preds[:, i].astype(np.float64)
        m = compute_metrics(y_true, y_mean, y_std)
        all_metrics[prop] = m
        vals = '  '.join(f'{m[k]:>8.4f}' for k in metric_names)
        log.info('  %6s | %s', prop, vals)

    log.info('  %s', '-' * (len(header) - 2))

    avg = {
        k: float(np.mean([all_metrics[p][k] for p in MOLECULAR_PROPERTIES]))
        for k in metric_names
    }
    std_across = {
        k: float(np.std([all_metrics[p][k] for p in MOLECULAR_PROPERTIES]))
        for k in metric_names
    }
    log.info(
        '  %6s | %s', 'mean',
        '  '.join(f'{avg[k]:>8.4f}' for k in metric_names),
    )
    log.info(
        '  %6s | %s', 'std',
        '  '.join(f'{std_across[k]:>8.4f}' for k in metric_names),
    )
    log.info('%s', '=' * 72)
    log.info('Done.')


if __name__ == '__main__':
    main()
