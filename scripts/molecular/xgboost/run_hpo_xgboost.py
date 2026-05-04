#!/usr/bin/env python3
"""
Optuna HPO for XGBoost molecular property prediction.

Searches over XGBoost tree hyperparameters (max_depth, learning_rate,
subsample, …) and TruncatedSVD components. Uses fold 0 (first fold of
first repeat), consistent with run_hpo_chemprop.py and run_hpo_molecular.py.

The objective minimises negative mean R² across the four target properties.
Top-N trial configs are saved to XGBoost_blind_optuna.json in the same
format as the other HPO scripts.

Usage examples:
    python run_hpo_xgboost.py
    python run_hpo_xgboost.py --n-trials 60 --group-col none
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

# ---------------------------------------------------------------------------
# Locate repository root so that qtnet is importable regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, 'data_curation')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(os.path.join(start_dir, '..', '..', '..'))
        cur = parent


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
for _root, _dirs, _ in os.walk(REPO_ROOT):
    if 'qtnet' in _dirs:
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break

_proj_candidate = os.path.dirname(REPO_ROOT)
if (os.path.isdir(os.path.join(_proj_candidate, 'experiments'))
        or os.path.exists(os.path.join(_proj_candidate, 'pyproject.toml'))):
    PROJECT_ROOT = _proj_candidate
else:
    PROJECT_ROOT = REPO_ROOT

from qtnet.data_utils import create_cv_splits, MOLECULAR_PROPERTIES_PRED
from qtnet.xgb_models.train_utils import fit_xgboost

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COLUMNS = MOLECULAR_PROPERTIES_PRED   # ['alpha', 'gap', 'U0', 'Cv']
DEFAULT_PKL = os.path.join(
    REPO_ROOT, 'data_curation', 'molecular', 'aimel_clustered_molecular.pkl'
)
DEFAULT_N_TRIALS = 50


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_fold0_split(
    df: pd.DataFrame,
    n_splits: int,
    group_col: str,
    base_seed: int,
) -> tuple:
    """Return (train_df, val_df) for fold 0, repeat 0 with a 10 % inner val."""
    gcol = None if group_col.lower() == 'none' else group_col
    for split_info in create_cv_splits(
        df,
        n_splits=n_splits,
        n_repeats=1,
        group_col=gcol,
        base_seed=base_seed,
        training_fractions=[1.0],
        val_fraction=0.1,
    ):
        if split_info['fold'] == 0:
            train_idx = split_info['train_idx']
            val_idx   = split_info['val_idx']
            return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    raise RuntimeError("Fold 0 not found — check n_splits >= 1")


# ---------------------------------------------------------------------------
# XGBoostHPO
# ---------------------------------------------------------------------------

class XGBoostHPO:
    """
    Optuna HPO for XGBMolPropertyRegressor.

    Each trial calls fit_xgboost with Optuna-suggested hyperparameters
    (max_depth, learning_rate, min_child_weight, subsample, colsample_bytree,
    reg_alpha, reg_lambda, svd_components) and evaluates mean R² on val.

    Args:
        train_df    : Training DataFrame with 'smiles' and target columns.
        val_df      : Validation DataFrame for scoring.
        smiles_col  : Name of the SMILES column.
        target_cols : Target column names.
        n_trials    : Number of Optuna trials.
        fp_size     : Morgan fingerprint size (fixed across trials).
        radius      : Morgan fingerprint radius (fixed across trials).
        random_state: Random seed for model reproducibility.
        seed        : Random seed for Optuna sampler.
        verbose     : Print per-trial summaries.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        smiles_col: str = 'smiles',
        target_cols: list = None,
        n_trials: int = DEFAULT_N_TRIALS,
        n_jobs: int = 2,
        random_state: int = 0,
        seed: int = 0,
        verbose: bool = True,
    ):
        self.train_df     = train_df
        self.val_df       = val_df
        self.smiles_col   = smiles_col
        self.target_cols  = target_cols or TARGET_COLUMNS
        self.n_trials     = n_trials
        self.n_jobs       = n_jobs
        self.random_state = random_state
        self.seed         = seed
        self.verbose      = verbose
        self._val_smiles  = val_df[smiles_col].tolist()
        self._val_y       = val_df[self.target_cols].values.astype(np.float32)

    @property
    def study_name(self) -> str:
        return 'XGBoost_blind'

    def _objective(self, trial: optuna.Trial) -> float:
        fixed_kwargs = {
            'random_state': self.random_state,
            'xgb_params': {'n_jobs': self.n_jobs},
        }
        try:
            model = fit_xgboost(
                train_df=self.train_df,
                val_df=self.val_df,
                smiles_col=self.smiles_col,
                target_cols=self.target_cols,
                trial=trial,
                **fixed_kwargs,
            )
        except Exception as exc:
            warnings.warn(f"Trial {trial.number} failed: {exc}")
            raise optuna.exceptions.TrialPruned()

        mean_r2 = model.score(self._val_smiles, self._val_y)

        if self.verbose:
            hp_str = " | ".join(f"{k}={v}" for k, v in trial.params.items())
            print(
                f"[{self.study_name}] trial {trial.number:>3d} | "
                f"val_r2={mean_r2:.4f} | {hp_str}"
            )

        return -mean_r2   # minimise negative R²

    def run(self, save_top_n: int = 5, save_dir: str = '') -> optuna.Study:
        """Run all trials and return the finished study."""
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            sampler=TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=0, interval_steps=1
            ),
        )
        study.optimize(self._objective, n_trials=self.n_trials)

        best_r2 = -study.best_value
        print(f"\n{'='*60}")
        print(f"[{self.study_name}] HPO complete — best val R²: {best_r2:.4f}")
        print(f"  Best params: {study.best_params}")
        print(f"{'='*60}\n")

        if save_top_n > 0:
            trials = sorted(
                [t for t in study.trials if t.value is not None],
                key=lambda t: t.value,
            )[:save_top_n]
            records = [
                {
                    'trial':  t.number,
                    'value':  t.value,
                    'val_r2': -t.value,
                    'params': t.params,
                }
                for t in trials
            ]
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{self.study_name}_optuna.json"
            fname = os.path.join(save_dir, fname)
            with open(fname, 'w') as fp:
                json.dump(records, fp, indent=2)
            print(f"Saved top-{save_top_n} trial info to {fname}")

        return study


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna HPO for XGBoost molecular property prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Target properties: alpha, gap, U0, Cv

The search uses fold 0 (= fold 1, repeat 1 in 1-indexed notation).
        """,
    )
    p.add_argument('--pkl-file', default=DEFAULT_PKL,
                   help='Path to the molecular DataFrame pickle')
    p.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS,
                   help=f'Number of Optuna trials (default: {DEFAULT_N_TRIALS})')
    p.add_argument('--save-top-n', type=int, default=5,
                   help='Number of best-trial configs to dump to JSON (default: 5)')
    p.add_argument('--seed', type=int, default=0,
                   help='Base random seed (default: 0)')
    p.add_argument('--n-splits', type=int, default=5,
                   help='Number of CV folds (default: 5)')
    p.add_argument('--group-col', type=str, default='Murcko_Scaffold',
                   help="DataFrame column for grouped k-fold; 'none' for plain KFold")
    p.add_argument('--n_jobs', type=int, default=2,
                   help='Number of parallel threads for XGBoost (default: 2)')
    p.add_argument('--optuna-dir', type=str,
                   default=os.path.join(REPO_ROOT, 'optimal_hyperparams'),
                   help=f'Directory where to save optimized hyperparameters (default: {os.path.join(REPO_ROOT, 'optimal_hyperparams')})')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.pkl_file} ...")
    t0 = time.perf_counter()
    df = pd.read_pickle(args.pkl_file)
    print(f"  {len(df)} molecules ({time.perf_counter() - t0:.1f}s)")

    missing = [c for c in ['smiles'] + TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    print(f"\nBuilding fold 0 split (group_col='{args.group_col}') ...")
    train_df, val_df = get_fold0_split(
        df,
        n_splits=args.n_splits,
        group_col=args.group_col,
        base_seed=args.seed,
    )
    print(f"  train={len(train_df)}  val={len(val_df)}")

    print(f"\n{'='*60}")
    print(f"HPO: XGBoost (blind)")
    print(f"  n_trials={args.n_trials}  seed={args.seed}")
    print(f"  fp_size: Optuna suggests [512, 1024]  radius: Optuna suggests [2, 4, 6]")
    print(f"{'='*60}\n")

    hpo = XGBoostHPO(
        train_df=train_df,
        val_df=val_df,
        target_cols=TARGET_COLUMNS,
        n_trials=args.n_trials,
        random_state=args.seed,
        seed=args.seed,
    )
    hpo.run(save_top_n=args.save_top_n)


if __name__ == '__main__':
    main()
