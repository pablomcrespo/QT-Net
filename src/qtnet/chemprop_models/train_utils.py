"""
Training utilities for ChemProp molecular property regression.

Provides:
  - fit_chemprop: single-run training from DataFrames
  - optimize_fit_chemprop: Optuna HPO wrapping fit_chemprop
"""

from typing import Optional, List

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from qtnet.chemprop_models.models import ChemPropPredictor


def fit_chemprop(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    smiles_col: str = "smiles",
    target_cols: Optional[List[str]] = None,
    test_df: Optional[pd.DataFrame] = None,
    trial: Optional[optuna.Trial] = None,
    **kwargs,
) -> ChemPropPredictor:
    """
    Fit a ChemPropPredictor from DataFrames.

    Args:
        train_df: Training data with a SMILES column and target columns.
        val_df: Validation data (used for metric monitoring during training).
        smiles_col: Name of the SMILES column (default: 'smiles').
        target_cols: Target column names. Defaults to all non-SMILES columns.
        test_df: Unused at training time; accepted for API consistency.
        trial: If provided, architecture hyperparameters are suggested via
            Optuna. Fixed training arguments (max_epochs, accelerator, …)
            should be passed via **kwargs.
        **kwargs: Extra arguments forwarded to ChemPropPredictor
            (e.g. max_epochs=100, accelerator='gpu').

    Returns:
        Fitted ChemPropPredictor.
    """
    if target_cols is None:
        target_cols = [c for c in train_df.columns if c != smiles_col]

    if trial is not None:
        kwargs.update({
            "d_h":            trial.suggest_int("d_h", 100, 600, step=100),
            "depth":          trial.suggest_int("depth", 2, 6),
            "ffn_hidden_dim": trial.suggest_int("ffn_hidden_dim", 100, 600, step=100),
            "ffn_n_layers":   trial.suggest_int("ffn_n_layers", 1, 4),
            "dropout":        trial.suggest_float("dropout", 0.0, 0.4, step=0.1),
            "max_lr":         trial.suggest_float("max_lr", 1e-4, 1e-2, log=True),
        })

    predictor = ChemPropPredictor(n_targets=len(target_cols), **kwargs)
    predictor.fit(
        train_df[smiles_col].tolist(),
        train_df[target_cols].values.astype(np.float32),
        val_smiles=val_df[smiles_col].tolist(),
        val_y=val_df[target_cols].values.astype(np.float32),
    )
    return predictor


def optimize_fit_chemprop(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    smiles_col: str = "smiles",
    target_cols: Optional[List[str]] = None,
    n_trials: int = 30,
    test_df: Optional[pd.DataFrame] = None,
    fixed_kwargs: Optional[dict] = None,
) -> dict:
    """
    Run Optuna HPO for ChemPropPredictor.

    The objective minimises the negative mean R² on the validation set,
    computed after each trial by calling ``predictor.predict()`` and
    comparing against the true validation targets.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame (scored after each trial).
        smiles_col: SMILES column name (default: 'smiles').
        target_cols: Target column names; defaults to all non-SMILES columns.
        n_trials: Number of Optuna trials (default: 30).
        test_df: Unused, accepted for API symmetry.
        fixed_kwargs: Arguments fixed across all trials that are forwarded to
            fit_chemprop in every trial and NOT tuned by Optuna
            (e.g. ``{'max_epochs': 100, 'accelerator': 'gpu'}``).

    Returns:
        Dict mapping hyperparameter names to their best found values.
    """
    from sklearn.metrics import r2_score  # local import to keep top-level lean

    if target_cols is None:
        target_cols = [c for c in train_df.columns if c != smiles_col]

    val_smiles = val_df[smiles_col].tolist()
    val_y = val_df[target_cols].values.astype(np.float32)
    fixed = fixed_kwargs or {}

    def objective(trial: optuna.Trial) -> float:
        predictor = fit_chemprop(
            train_df, val_df,
            smiles_col=smiles_col,
            target_cols=target_cols,
            trial=trial,
            **fixed,
        )
        preds = predictor.predict(val_smiles)
        # Mean R² across targets; minimise negative
        r2 = float(np.mean([
            r2_score(val_y[:, i], preds[:, i]) for i in range(val_y.shape[1])
        ]))
        return -r2

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=3),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
