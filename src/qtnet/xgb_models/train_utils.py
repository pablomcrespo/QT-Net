"""
Training utilities for XGBoost molecular property regression.

Provides:
  - fit_xgboost: single-run training from DataFrames
"""

from typing import Optional, List

import numpy as np
import pandas as pd
import optuna

from qtnet.xgb_models.models import XGBMolPropertyRegressor


def fit_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    smiles_col: str = 'smiles',
    target_cols: Optional[List[str]] = None,
    test_df: Optional[pd.DataFrame] = None,
    trial: Optional[optuna.Trial] = None,
    **kwargs,
) -> XGBMolPropertyRegressor:
    """
    Fit an XGBMolPropertyRegressor from DataFrames.

    Args:
        train_df: Training data with a SMILES column and target columns.
        val_df: Validation data (used for early stopping in XGBoost).
        smiles_col: Name of the SMILES column (default: 'smiles').
        target_cols: Target column names. Defaults to all non-SMILES columns.
        test_df: Unused at training time; accepted for API consistency.
        trial: If provided, hyperparameters are suggested via Optuna for HPO.
        **kwargs: Extra keyword arguments forwarded to XGBMolPropertyRegressor.

    Returns:
        Fitted XGBMolPropertyRegressor.
    """
    if target_cols is None:
        target_cols = [c for c in train_df.columns if c != smiles_col]

    if trial is not None:
        xgb_params = dict(kwargs.pop('xgb_params', {}))
        xgb_params.update({
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'n_estimators':     trial.suggest_int('n_estimators', 1000, 3000),
            'multi_strategy':   trial.suggest_categorical('multi_strategy', ['one_output_per_tree', 'multi_output_tree']),
        })
        kwargs['xgb_params'] = xgb_params
        kwargs['svd_components'] = trial.suggest_int('svd_components', 0, 128)

        feat_combo = trial.suggest_categorical(
            'feat_combo', ['fp_only', 'desc_only', 'both']
        )
        kwargs['use_fp'] = feat_combo in ('fp_only', 'both')
        kwargs['use_descriptors'] = feat_combo in ('desc_only', 'both')

        kwargs['fp_size'] = trial.suggest_categorical('fp_size', [512, 1024])
        kwargs['radius'] = trial.suggest_categorical('radius', [2, 4, 6])

    model = XGBMolPropertyRegressor(**kwargs)
    model.fit(
        train_df[smiles_col].tolist(),
        train_df[target_cols].values.astype(np.float32),
        smiles_val=val_df[smiles_col].tolist(),
        y_val=val_df[target_cols].values.astype(np.float32),
    )
    return model