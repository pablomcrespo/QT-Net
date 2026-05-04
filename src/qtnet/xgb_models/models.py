"""
XGBMolPropertyRegressor: A Sklearn-compatible wrapper for XGBoost molecular property prediction.
Accepts raw SMILES strings as input, computes Morgan fingerprints and RDKit descriptors, and applies a preprocessing pipeline before fitting an XGBoost regressor.
"""
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Union, List, Any, Optional

import numpy as np
import skops.io as sio
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
    FunctionTransformer,
    QuantileTransformer,
    StandardScaler,
)
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score


def _standardize_mol(smiles: str, uncharge: bool) -> Optional[Chem.Mol]:
    """Parse SMILES, keep largest fragment, optionally uncharge, add explicit Hs."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        if uncharge:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        return None
    return Chem.AddHs(mol)


def _compute_desc_array(
    mol: Optional[Chem.Mol],
    descriptors_list: List[str],
) -> np.ndarray:
    """Compute RDKit descriptor values for one molecule, clipped to [-1e3, 1e3].

    Follows the same iteration order as ``get_mol_descriptors`` (i.e.
    ``Descriptors._descList`` order filtered by ``descriptors_list``) so the
    resulting array is consistent with the fitted preprocessor's expectations.
    """
    n = len(descriptors_list)
    if mol is None:
        return np.full(n, np.nan, dtype=np.float32)
    desc_set = set(descriptors_list)
    result = np.zeros(n, dtype=np.float32)
    i = 0
    for name, fn in Descriptors._descList:
        if name not in desc_set:
            continue
        try:
            result[i] = fn(mol)
        except Exception:
            result[i] = np.nan
        i += 1
    return np.clip(result, a_min=-1e3, a_max=1e3)


class XGBMolPropertyRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for XGBoost molecular property prediction.
    Accepts raw SMILES strings as input.

    Inheriting BaseEstimator gives you get_params() / set_params() for free,
    which Optuna and sklearn's cross_val_score expect.
    Inheriting RegressorMixin gives you a default score() method (R²).
    
    Args:
        fp_size (int): Size of the Morgan fingerprint (default: 512).
        radius (int): Radius for Morgan fingerprint (default: 2).
        svd_components (int): Number of components for TruncatedSVD on fingerprints (default: 64). If 0 or None, no SVD is applied.
        xgb_params (dict): Additional parameters for XGBoost regressor (default: None, which uses internal defaults).
        descriptors_list (List[str]): List of RDKit descriptor names to compute (default: all available descriptors).
        random_state (int): Random seed for reproducibility (default: 42).
    """

    def __init__(
        self,
        fp_size: int = 512,
        radius: int = 2,
        svd_components: int = 64,
        xgb_params: dict = None,
        descriptors_list: List[str] = [name for name, _ in Descriptors._descList],
        use_fp: bool = True,
        use_descriptors: bool = True,
        uncharge: bool = False,
        random_state: int = 42,
    ):
        self.fp_size = fp_size
        self.radius = radius
        self.svd_components = svd_components
        self.xgb_params = xgb_params or {}
        self.descriptors_list = descriptors_list
        self.use_fp = use_fp
        self.use_descriptors = use_descriptors
        self.uncharge = uncharge
        self.random_state = random_state
        

    @property
    def fpgen(self):
        """ Reconstruct fpgen on-the-fly from serializable params. Never stored.
            Slow, but allows us to save the model without worrying about RDKit's
            non-serializable objects. """
        return AllChem.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.fp_size,
            includeChirality=True,
        )

    @property
    def lfg(self):
        """ Reconstruct LargestFragmentChooser on-the-fly. Never stored.
            Slow, but allows us to save the model without worrying about RDKit's
            non-serializable objects. """
        return rdMolStandardize.LargestFragmentChooser()
    
    @property
    def uncharger(self):
        """ Reconstruct Uncharger on-the-fly. Never stored.
            Slow, but allows us to save the model without worrying about RDKit's
            non-serializable objects. """
        return rdMolStandardize.Uncharger()

    def featurize(
        self,
        smiles_list: List[str],
        n_jobs: Optional[int] = None,
    ) -> np.ndarray:
        """SMILES → raw feature matrix (n_samples, n_features).

        Standardization (largest fragment, optional uncharge, add Hs) and
        descriptor computation run in parallel via ``multiprocessing.Pool``.
        Fingerprint calculation runs single-threaded (already vectorised by
        RDKit and typically faster without IPC overhead).

        Args:
            smiles_list: List of SMILES strings.
            n_jobs: Worker processes for the parallel steps.  ``None`` uses
                all available CPUs; ``1`` runs sequentially.
        """
        n_workers = cpu_count() if n_jobs is None else n_jobs

        # Step 1 (parallel): parse + largest fragment + uncharge + add Hs
        standardize_fn = partial(_standardize_mol, uncharge=self.uncharge)
        if n_workers > 1:
            with Pool(processes=n_workers) as pool:
                mols = pool.map(standardize_fn, smiles_list)
        else:
            mols = [standardize_fn(s) for s in smiles_list]

        parts = []

        # Step 2 (single-threaded): Morgan fingerprints
        if self.use_fp:
            fpgen = self.fpgen  # reconstruct once outside the loop
            fp_list = []
            for mol in mols:
                if mol is None:
                    fp_list.append(np.zeros(self.fp_size, dtype=np.uint8))
                else:
                    fp_list.append(fpgen.GetFingerprintAsNumPy(mol))
            parts.append(np.array(fp_list))

        # Step 3 (parallel): RDKit descriptors
        if self.use_descriptors:
            desc_fn = partial(
                _compute_desc_array, descriptors_list=self.descriptors_list
            )
            if n_workers > 1:
                with Pool(processes=n_workers) as pool:
                    desc_list = pool.map(desc_fn, mols)
            else:
                desc_list = [desc_fn(mol) for mol in mols]
            parts.append(np.array(desc_list, dtype=np.float32))

        return np.concatenate(parts, axis=1)
        

    def fit(
        self,
        smiles_list: List[str],
        y: np.ndarray,
        smiles_val: Optional[List[str]] = None,
        y_val: Optional[np.ndarray] = None,
     ) -> 'XGBMolPropertyRegressor':
        """
        Fit the model to the training data.
        
        smiles_val / y_val are optional — if provided, used for early stopping.
        Not part of the standard sklearn fit() signature, but acceptable for a
        custom wrapper. If you need strict sklearn compatibility (e.g., for
        cross_val_score), omit them and disable early stopping.

        Args:
            smiles_list (List[str]): List of SMILES strings for training.
            y (np.ndarray): Target values for training (shape: [n_samples, n_targets]).
            smiles_val (Optional[List[str]]): List of SMILES strings for validation (for early stopping).
            y_val (Optional[np.ndarray]): Target values for validation (for early stopping).
            
        Returns:
            self: The fitted model instance.
        """
        if not self.use_fp and not self.use_descriptors:
            raise ValueError(
                "At least one of 'use_fp' or 'use_descriptors' must be True."
            )

        X = self.featurize(smiles_list)
        print('All SMILES featurized.')

        # NOTE: If we were NOT to "break" the pipeline this way (for example by
        # using a TransformedTargetRegressor to combine a QuantileTransformer
        # and the SkPipeline), the fused pipeline would change the shape of the
        # training data, but not the ones used for early stopping, throwing an
        # error! We need early stopping for making a more performing model and
        # for "pruning" early during HPO.
        desc_pipe = SkPipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('var_thresh', VarianceThreshold(threshold=0.0)),
            ('scale', StandardScaler()),
        ])

        if self.use_fp and self.use_descriptors:
            fp_proc = (
                TruncatedSVD(
                    n_components=self.svd_components,
                    random_state=self.random_state,
                )
                if self.svd_components else 'passthrough'
            )
            self.preprocessor_ = SkPipeline([
                ('ct', ColumnTransformer([
                    ('fp', fp_proc, slice(0, self.fp_size)),
                    ('desc', desc_pipe, slice(self.fp_size, None)),
                ])),
            ])
        elif self.use_fp:
            if self.svd_components:
                self.preprocessor_ = SkPipeline([
                    ('svd', TruncatedSVD(
                        n_components=self.svd_components,
                        random_state=self.random_state,
                    ))
                ])
            else:
                self.preprocessor_ = SkPipeline([('id', FunctionTransformer())])
        else:
            self.preprocessor_ = desc_pipe

        self.target_transformer_ = QuantileTransformer(
            output_distribution='normal', random_state=self.random_state
        )

        X_proc   = self.preprocessor_.fit_transform(X)
        y_transf = self.target_transformer_.fit_transform(y)

        default_xgb = dict(
            tree_method='hist',
            objective='reg:pseudohubererror',
            multi_strategy='one_output_per_tree',
            n_estimators=2000,
            early_stopping_rounds=50 if smiles_val is not None else None,
            random_state=self.random_state,
            n_jobs=2,
        )
        final_params = {**default_xgb, **self.xgb_params}

        self.model_ = xgb.XGBRegressor(**final_params)

        if smiles_val is not None:
            X_val_proc = self.preprocessor_.transform(self.featurize(smiles_val))
            y_val_t    = self.target_transformer_.transform(y_val)
            self.model_.fit(
                X_proc, y_transf,
                eval_set=[(X_val_proc, y_val_t)],
                verbose=False,
            )
        else:
            self.model_.fit(X_proc, y_transf)

        return self

    def predict(
        self,
        smiles_list: Union[List[str], np.ndarray],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Predict target values for SMILES strings or a pre-featurized matrix.

        Passing a pre-featurized ``np.ndarray`` (from :meth:`featurize`)
        skips the expensive RDKit step, which is useful when the same
        feature matrix is reused across an ensemble of models.

        Args:
            smiles_list: List of SMILES strings, or a raw feature array of
                shape ``(n_samples, n_features)`` produced by
                :meth:`featurize`.  Each model still applies its own fitted
                preprocessor, so only the RDKit step is shared.
            batch_size: If set, process this many samples per chunk to
                reduce peak memory usage.  Defaults to all at once.

        Returns:
            np.ndarray: Predicted target values (n_samples, n_targets) in
            the original (un-normalised) scale.
        """
        is_prefeaturized = isinstance(smiles_list, np.ndarray)
        n = len(smiles_list)

        if batch_size is None:
            X_raw = (
                smiles_list if is_prefeaturized
                else self.featurize(smiles_list)
            )
            X_proc = self.preprocessor_.transform(X_raw)
            return self.target_transformer_.inverse_transform(
                self.model_.predict(X_proc)
            )

        chunks = []
        for start in range(0, n, batch_size):
            chunk = smiles_list[start:start + batch_size]
            if not is_prefeaturized:
                chunk = self.featurize(chunk)
            X_proc = self.preprocessor_.transform(chunk)
            chunks.append(
                self.target_transformer_.inverse_transform(
                    self.model_.predict(X_proc)
                )
            )
        return np.concatenate(chunks, axis=0)

    def score(self, smiles_list: List[str], y: np.ndarray) -> float:
        """ Returns mean R² across targets (used by RegressorMixin).
        
        Args:
            smiles_list (List[str]): List of SMILES strings for evaluation.
            y (np.ndarray): True target values (shape: [n_samples, n_targets]).
            
        Returns:
            float: Mean R² score across targets.
        """
        y_pred = self.predict(smiles_list)
        return float(np.mean([
            r2_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])
        ]))

    def compute_molecular_features(self, smiles: str) -> np.ndarray:
        """ Compute Morgan fingerprints and RDKit descriptors from SMILES.
        
        Args:
            smiles (str): The SMILES string of the molecule.
            
        Returns:
            np.ndarray: Combined feature vector of fingerprints and descriptors.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Standardize molecule: keep largest fragment and DO NOT uncharge.
        try:        
            mol = self.lfg.choose(mol)
            if self.uncharge:
                mol = self.uncharger.uncharge(mol)
        except:
            return None

        # Add explicit hydrogens
        mol = Chem.AddHs(mol)

        features = []
        if self.use_fp:
            features.append(self.fpgen.GetFingerprintAsNumPy(mol))

        if self.use_descriptors:
            desc_array = np.array(
                list(self.get_mol_descriptors(
                    mol, descriptors_list=self.descriptors_list
                ).values()),
                dtype=np.float32,
            )
            desc_array = np.clip(desc_array, a_min=-1e3, a_max=1e3)
            features.append(desc_array)

        return np.concatenate(features)

    @staticmethod
    def get_mol_descriptors(
        mol: Union[str, Chem.Mol],
        missing: Any = np.nan,
        descriptors_list: List[str] = [name for name, _ in Descriptors._descList],
    ) -> dict:
        """ Calculate the full list of descriptors for a molecule.
        
        Args:
            mol (Union[str, Chem.Mol]): The molecule as a SMILES string or an RDKit Mol object.
            missing (Any): Value to use if a descriptor cannot be calculated.
            descriptors_list (List[str]): List of descriptor names to calculate. Defaults to all available descriptors.
            
        Returns:
            dict: A dictionary mapping descriptor names to their calculated values.
        """
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        desc2val = {}
        for name, fn in Descriptors._descList:
            if name not in descriptors_list:
                continue
            # Some of the descriptor functions can throw errors if they fail,
            # catch those here:
            try:
                val = fn(mol)
            except:
                # And set the descriptor value to whatever `missing` is
                val = missing
            desc2val[name] = val
        return desc2val

    def save(self, path: str) -> None:
        """
        Save the model to disk. Splits into two files:
        - {path}.skops  : sklearn components (preprocessor + target transformer)
        - {path}.ubj    : XGBoost model in binary JSON format (its native format)

        RDKit's fpgen is intentionally excluded — it is reconstructed from
        (fp_size, radius) which are already stored as init params in the skops file.

        Args:
            path: Base path without extension, e.g. 'models/mol_regressor'
        """
        # Raise an error if the model hasn't been fitted yet
        if not hasattr(self, 'model_'):
            raise ValueError("Cannot save an unfitted model. Please call fit() before saving.")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Temporarily detach the XGBoost model so skops only sees sklearn objects
        xgb_model = self.model_
        self.model_ = None
        sio.dump(self, f"{path}.skops")
        self.model_ = xgb_model  # restore immediately

        # Save XGBoost separately in its own stable binary format
        xgb_model.save_model(f"{path}.ubj")

    @classmethod
    def load(cls, path: str) -> 'XGBMolPropertyRegressor':
        """
        Load a saved model from disk.

        Args:
            path: Base path without extension, matching what was used in save().

        Returns:
            Fully restored XGBMolPropertyRegressor instance.
        """
        unknown_types = sio.get_untrusted_types(file=f"{path}.skops")
        
        for t in unknown_types:
            if not ('XGBMolPropertyRegressor' in t or 'numpy.dtype' in t):
                raise ValueError(f"Untrusted type '{t}' found in skops file. Aborting load for safety.")
        
        # Inspect unknown_types before trusting in production
        instance = sio.load(f"{path}.skops", trusted=unknown_types)

        # Restore XGBoost model
        instance.model_ = xgb.XGBRegressor()
        instance.model_.load_model(f"{path}.ubj")

        return instance

    def set_n_jobs(self, n_jobs: Optional[int]) -> None:
        """Set the number of threads used by XGBoost for inference.

        Args:
            n_jobs: Number of threads.  ``None`` lets XGBoost use all
                available threads (its default when the parameter is unset).
        """
        if not hasattr(self, 'model_') or self.model_ is None:
            raise ValueError(
                "Cannot set n_jobs on an unfitted model."
            )
        self.model_.set_params(n_jobs=n_jobs)


class XGBAtomPropertyRegressor(BaseEstimator, RegressorMixin):
    
    """
    Placeholder for a future implementation of an atom-level property regressor using XGBoost.
    This would involve featurizing individual atoms (e.g., using atom descriptors and local environment features)
    and training a model to predict properties at the atom level, which could then be aggregated for molecule-level predictions.
    """
    pass
