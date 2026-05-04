#!/usr/bin/env python3
"""
Optuna HPO for ChemProp molecular property prediction.

Searches over message-passing depth, hidden dimensions, dropout, and
learning rate. Uses fold 0 (first fold of first repeat) for consistency
with run_hpo_molecular.py.

Val R² (mean across targets) is reported at every trial so the
MedianPruner can cut unpromising trials early.

Usage examples:
    python run_hpo_chemprop.py
    python run_hpo_chemprop.py --n-trials 50 --epochs-per-trial 100
    python run_hpo_chemprop.py --group-col none
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
from sklearn.metrics import r2_score

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

from qtnet.data_utils import create_cv_splits, MOLECULAR_PROPERTIES_PRED
from qtnet.chemprop_models.train_utils import fit_chemprop

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COLUMNS = MOLECULAR_PROPERTIES_PRED   # ['alpha', 'gap', 'U0', 'Cv']
DEFAULT_PKL = os.path.join(
    REPO_ROOT, 'data_curation', 'molecular', 'aimel_clustered_molecular.pkl'
)
DEFAULT_N_TRIALS = 30
DEFAULT_EPOCHS   = 50


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
# ChemPropHPO
# ---------------------------------------------------------------------------

class ChemPropHPO:
    """
    Optuna HPO for ChemPropPredictor.

    Each trial calls fit_chemprop, evaluates mean R² on val, and reports
    to Optuna. The MedianPruner is not applicable here (no per-epoch
    intermediate reporting), so pruning happens between trials only.

    Args:
        train_df         : Training DataFrame with 'smiles' and target columns.
        val_df           : Validation DataFrame for scoring.
        smiles_col       : Name of the SMILES column.
        target_cols      : Target column names.
        n_trials         : Number of Optuna trials.
        epochs_per_trial : max_epochs for each ChemProp trial.
        batch_size       : Mini-batch size.
        warmup_epochs    : LR warm-up epochs (passed to ChemPropPredictor).
        seed             : Random seed for Optuna sampler.
        verbose          : Print per-trial summaries.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        smiles_col: str = 'smiles',
        target_cols: list = None,
        n_trials: int = DEFAULT_N_TRIALS,
        epochs_per_trial: int = DEFAULT_EPOCHS,
        batch_size: int = 64,
        warmup_epochs: int = 2,
        seed: int = 0,
        verbose: bool = True,
    ):
        self.train_df        = train_df
        self.val_df          = val_df
        self.smiles_col      = smiles_col
        self.target_cols     = target_cols or TARGET_COLUMNS
        self.n_trials        = n_trials
        self.epochs_per_trial = epochs_per_trial
        self.batch_size      = batch_size
        self.warmup_epochs   = warmup_epochs
        self.seed            = seed
        self.verbose         = verbose
        self._val_smiles     = val_df[smiles_col].tolist()
        self._val_y          = val_df[self.target_cols].values.astype(np.float32)

    @property
    def study_name(self) -> str:
        return 'ChemProp_blind'

    def _objective(self, trial: optuna.Trial) -> float:
        fixed_kwargs = {
            'max_epochs':    self.epochs_per_trial,
            'batch_size':    self.batch_size,
            'warmup_epochs': self.warmup_epochs,
            'accelerator':   'auto',
        }
        try:
            predictor = fit_chemprop(
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

        preds  = predictor.predict(self._val_smiles)
        r2_per = [
            r2_score(self._val_y[:, i], preds[:, i])
            for i in range(self._val_y.shape[1])
        ]
        mean_r2 = float(np.mean(r2_per))

        if self.verbose:
            hp_str = " | ".join(
                f"{k}={v}" for k, v in trial.params.items()
            )
            print(
                f"[{self.study_name}] trial {trial.number:>3d} | "
                f"val_r2={mean_r2:.4f} | {hp_str}"
            )

        # Minimise negative R²
        return -mean_r2

    def run(self, save_top_n: int = 5) -> optuna.Study:
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
                    'value':  t.value,        # negative mean R²
                    'val_r2': -t.value,
                    'params': t.params,
                }
                for t in trials
            ]
            fname = f"{self.study_name}_optuna.json"
            with open(fname, 'w') as fp:
                json.dump(records, fp, indent=2)
            print(f"Saved top-{save_top_n} trial info to {fname}")

        return study


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna HPO for ChemProp molecular property prediction",
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
    p.add_argument('--epochs-per-trial', type=int, default=DEFAULT_EPOCHS,
                   help=f'max_epochs per ChemProp trial (default: {DEFAULT_EPOCHS})')
    p.add_argument('--batch-size', type=int, default=64,
                   help='Mini-batch size (default: 64)')
    p.add_argument('--warmup-epochs', type=int, default=2,
                   help='LR warm-up epochs (default: 2)')
    p.add_argument('--save-top-n', type=int, default=5,
                   help='Number of best-trial configs to dump to JSON (default: 5)')
    p.add_argument('--seed', type=int, default=0,
                   help='Base random seed (default: 0)')
    p.add_argument('--n-splits', type=int, default=5,
                   help='Number of CV folds (default: 5)')
    p.add_argument('--group-col', type=str, default='Murcko_Scaffold',
                   help="DataFrame column for grouped k-fold; 'none' for plain KFold")
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
    print(f"HPO: ChemProp (blind)")
    print(f"  n_trials={args.n_trials}  epochs_per_trial={args.epochs_per_trial}")
    print(f"  batch_size={args.batch_size}  warmup_epochs={args.warmup_epochs}")
    print(f"  seed={args.seed}")
    print(f"{'='*60}\n")

    hpo = ChemPropHPO(
        train_df=train_df,
        val_df=val_df,
        target_cols=TARGET_COLUMNS,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
    )
    hpo.run(save_top_n=args.save_top_n)


if __name__ == '__main__':
    main()
