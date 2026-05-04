"""
ChemProp molecular property prediction — three-tier design:

  MolPropDataModule   — LightningDataModule that owns the datasets and
                        all target normalisation statistics. Can be saved
                        and loaded independently so that the same
                        normalisation is applied to future predictions
                        without re-fitting.

  MolPropModule       — LightningModule that subclasses ChemProp's MPNN.
                        Adds TorchMetrics validation logging (RMSE, MAE, R²)
                        in the original (un-normalised) target space.
                        predict_step() is inherited unchanged from MPNN and
                        returns predictions in the original scale because the
                        UnscaleTransform baked into the FFN is activated
                        automatically in eval mode.

  ChemPropPredictor   — High-level helper that ties the DataModule and the
                        LightningModule together for fit / predict / save /
                        load. The model weights and the DataModule state are
                        saved to separate files so normalisation statistics
                        survive independently of the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import lightning as L
from rdkit import Chem

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.nn.transforms import UnscaleTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smiles_to_datapoints(
    smiles_list: List[str],
    y: Optional[np.ndarray] = None,
) -> List[MoleculeDatapoint]:
    """Convert SMILES strings to MoleculeDatapoints, skipping invalid SMILES."""
    datapoints = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        yi = y[i].astype(np.float32) if y is not None else None
        datapoints.append(MoleculeDatapoint(mol=mol, y=yi))
    return datapoints


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class MolPropDataModule(L.LightningDataModule):
    """
    LightningDataModule for molecular property prediction.

    Responsibilities:
      - Compute and store target normalisation statistics (mean, std) from
        training data.
      - Provide normalised targets to the training and validation dataloaders.
      - Provide un-labelled batches for the prediction dataloader.
      - Serialise / deserialise normalisation statistics independently of the
        model, so predictions on new data can use the same scale as training.

    Typical lifecycle::

        dm = MolPropDataModule(batch_size=64)
        dm.setup_fit(train_smiles, train_y, val_smiles, val_y)
        # ... train with trainer.fit(model, dm) ...
        dm.save("run/datamodule.pt")

        # Later, for prediction on new data:
        dm2 = MolPropDataModule.load("run/datamodule.pt")
        dm2.setup_predict(new_smiles)
        preds = trainer.predict(model, dm2)
    """

    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.batch_size = batch_size
        # Set by setup_fit(); also restored by load_state_dict()
        self.y_mean_: Optional[np.ndarray] = None
        self.y_std_: Optional[np.ndarray] = None
        self._train_dataset: Optional[MoleculeDataset] = None
        self._val_dataset: Optional[MoleculeDataset] = None
        self._pred_dataset: Optional[MoleculeDataset] = None

    # ------------------------------------------------------------------ setup

    def setup_fit(
        self,
        train_smiles: List[str],
        train_y: np.ndarray,
        val_smiles: Optional[List[str]] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> None:
        """
        Compute normalisation statistics from training targets and build
        training / validation datasets with z-scored targets.

        Call this before trainer.fit().
        """
        train_y = np.atleast_2d(np.array(train_y, dtype=np.float32))
        if train_y.shape[0] == 1:               # (1, N) → (N, 1)
            train_y = train_y.T

        self.y_mean_ = train_y.mean(axis=0)
        self.y_std_ = train_y.std(axis=0)
        self.y_std_[self.y_std_ == 0] = 1.0    # avoid divide-by-zero

        train_y_norm = (train_y - self.y_mean_) / self.y_std_
        self._train_dataset = MoleculeDataset(
            _smiles_to_datapoints(train_smiles, train_y_norm)
        )

        if val_smiles is not None and val_y is not None:
            val_y = np.atleast_2d(np.array(val_y, dtype=np.float32))
            if val_y.shape[0] == 1:
                val_y = val_y.T
            val_y_norm = (val_y - self.y_mean_) / self.y_std_
            self._val_dataset = MoleculeDataset(
                _smiles_to_datapoints(val_smiles, val_y_norm)
            )

    def setup_predict(self, smiles_list: List[str]) -> None:
        """Build an un-labelled dataset for inference. Call before trainer.predict()."""
        self._pred_dataset = MoleculeDataset(_smiles_to_datapoints(smiles_list))

    # ------------------------------------------------------------------ dataloaders

    def train_dataloader(self):
        return build_dataloader(
            self._train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        if self._val_dataset is None:
            return None
        return build_dataloader(
            self._val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def predict_dataloader(self):
        return build_dataloader(
            self._pred_dataset, batch_size=self.batch_size, shuffle=False
        )

    # ------------------------------------------------------------------ serialisation

    def state_dict(self) -> dict:
        """Return normalistion statistics as a plain dict (numpy arrays)."""
        return {"y_mean": self.y_mean_, "y_std": self.y_std_}

    def load_state_dict(self, state: dict) -> None:
        """Restore normalisation statistics from a state dict."""
        self.y_mean_ = np.array(state["y_mean"], dtype=np.float32)
        self.y_std_ = np.array(state["y_std"], dtype=np.float32)

    def save(self, path: str) -> None:
        """Save normalisation statistics to *path* (single .pt file)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, batch_size: int = 64) -> "MolPropDataModule":
        """Load a DataModule from a previously saved .pt file."""
        dm = cls(batch_size=batch_size)
        dm.load_state_dict(torch.load(path, weights_only=False))
        return dm


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class MolPropModule(MPNN):
    """
    Subclass of ChemProp's MPNN that adds TorchMetrics validation logging.

    Architecture hyperparameters and target normalisation statistics are stored
    in ``arch_hparams`` so the module can be reconstructed from scratch (needed
    for loading saved weights).

    Training and validation steps operate in the normalised (z-scored) target
    space.  The ``UnscaleTransform`` attached to the FFN output layer is only
    active in eval mode, so ``predict_step()`` (inherited from MPNN) returns
    predictions in the original target scale automatically.

    Validation metrics (RMSE, MAE, R²) are computed in the original scale by
    un-normalising both predictions and targets inside ``validation_step``.
    """

    def __init__(
        self,
        n_targets: int,
        d_h: int = 300,
        depth: int = 3,
        ffn_hidden_dim: int = 300,
        ffn_n_layers: int = 2,
        dropout: float = 0.0,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        warmup_epochs: int = 5,
        y_mean: Optional[np.ndarray] = None,
        y_std: Optional[np.ndarray] = None,
    ):
        # Default normalisation to identity if not provided
        y_mean = np.zeros(n_targets, dtype=np.float32) if y_mean is None else np.array(y_mean, dtype=np.float32)
        y_std  = np.ones(n_targets, dtype=np.float32)  if y_std  is None else np.array(y_std,  dtype=np.float32)

        # Build ChemProp components
        mp = BondMessagePassing(d_h=d_h, depth=depth, dropout=dropout, undirected=True)
        agg = MeanAggregation()
        # UnscaleTransform: identity in train mode, applies X*std+mean in eval mode
        output_transform = UnscaleTransform(mean=y_mean.tolist(), scale=y_std.tolist())
        ffn = RegressionFFN(
            n_tasks=n_targets,
            input_dim=d_h,
            hidden_dim=ffn_hidden_dim,
            n_layers=ffn_n_layers,
            dropout=dropout,
            output_transform=output_transform,
        )
        super().__init__(
            message_passing=mp,
            agg=agg,
            predictor=ffn,
            init_lr=init_lr,
            max_lr=max_lr,
            final_lr=final_lr,
            warmup_epochs=warmup_epochs,
        )

        # Persist architecture params for reconstruction after loading weights
        self.arch_hparams: dict = dict(
            n_targets=n_targets, d_h=d_h, depth=depth,
            ffn_hidden_dim=ffn_hidden_dim, ffn_n_layers=ffn_n_layers,
            dropout=dropout, init_lr=init_lr, max_lr=max_lr, final_lr=final_lr,
            warmup_epochs=warmup_epochs,
        )

        # Register normalisation as buffers: saved in state_dict, moved with .to(device)
        self.register_buffer("y_mean", torch.tensor(y_mean))
        self.register_buffer("y_std",  torch.tensor(y_std))

        # TorchMetrics — updated per batch, computed and logged per epoch.
        # num_outputs=1 + flattened preds/targets gives a single scalar per metric,
        # which Lightning logs cleanly.  R2Score with uniform_average averages per target.
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_mae  = torchmetrics.MeanAbsoluteError()
        self.val_r2   = torchmetrics.R2Score(multioutput="uniform_average")

    # ------------------------------------------------------------------ steps

    def validation_step(self, batch, batch_idx: int = 0):
        """
        Validation step with TorchMetrics in the original (un-normalised) scale.

        The model is in eval mode here, so the UnscaleTransform is active and
        ``predictor(Z)`` already returns un-normalised predictions.
        Targets arrive as z-scored values from the DataModule, so we
        un-normalise them before passing to the metrics.
        """
        bmg, V_d, X_d, targets_norm, weights, lt_mask, gt_mask = batch
        mask = targets_norm.isfinite()
        targets_norm = targets_norm.nan_to_num(nan=0.0)

        # Predictions are un-normalised (eval mode activates UnscaleTransform)
        Z    = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor(Z)                         # shape: (B, n_targets)

        # Un-normalise targets to match prediction scale
        targets_orig = targets_norm * self.y_std + self.y_mean

        # Val loss: z-scored space (consistent with training loss)
        preds_norm = (preds - self.y_mean) / self.y_std
        val_loss = F.mse_loss(
            torch.where(mask, preds_norm, targets_norm),
            targets_norm,
        )

        # TorchMetrics: flatten to 1D so single-scalar metrics work for any n_targets
        preds_flat   = preds.reshape(-1)
        targets_flat = targets_orig.reshape(-1)
        self.val_rmse.update(preds_flat, targets_flat)
        self.val_mae.update(preds_flat, targets_flat)
        self.val_r2.update(preds, targets_orig)           # 2D for uniform_average R²

        batch_size = targets_norm.shape[0]
        self.log("val_loss", val_loss,     prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("val_rmse", self.val_rmse, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("val_mae",  self.val_mae,  on_epoch=True, batch_size=batch_size)
        self.log("val_r2",   self.val_r2,   prog_bar=True, on_epoch=True, batch_size=batch_size)

    # predict_step() is inherited from MPNN:
    #   def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #       bmg, V_d, X_d, *_ = batch
    #       return self(bmg, V_d, X_d)   ← eval-mode forward, UnscaleTransform active


# ---------------------------------------------------------------------------
# High-level predictor
# ---------------------------------------------------------------------------

class ChemPropPredictor:
    """
    High-level helper that orchestrates training and inference.

    Manages a ``MolPropModule`` and a ``MolPropDataModule``.  The two are saved
    to separate files so normalisation statistics can be loaded independently
    from model weights, and new data can be predicted without re-fitting.

    Saved files::

        {path}_model.pt     — MolPropModule state dict (weights + buffers)
        {path}_hparams.pt   — architecture hyperparameters + y_mean / y_std
        {path}_datamodule.pt — DataModule state dict (y_mean / y_std only)

    Args:
        n_targets:     Number of regression targets.
        d_h:           Hidden dimension of the message passing network.
        depth:         Number of message passing steps.
        ffn_hidden_dim: Hidden dimension of the FFN prediction head.
        ffn_n_layers:  Number of FFN layers.
        dropout:       Dropout applied in message passing and FFN.
        batch_size:    Mini-batch size for training and inference.
        max_epochs:    Maximum training epochs.
        init_lr, max_lr, final_lr: Endpoints of ChemProp's NoamLike LR schedule.
        accelerator:   Lightning accelerator ('auto', 'cpu', 'gpu', …).
    """

    def __init__(
        self,
        n_targets: int = 1,
        d_h: int = 300,
        depth: int = 3,
        ffn_hidden_dim: int = 300,
        ffn_n_layers: int = 2,
        dropout: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 50,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        warmup_epochs: int = 5,
        accelerator: str = "auto",
    ):
        self.n_targets = n_targets
        self.d_h = d_h
        self.depth = depth
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_n_layers = ffn_n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.accelerator = accelerator

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        train_smiles: List[str],
        train_y: np.ndarray,
        val_smiles: Optional[List[str]] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> "ChemPropPredictor":
        """
        Train the model.

        Creates a fresh DataModule (computes normalisation statistics from
        training targets) and a fresh MolPropModule, then runs a Lightning
        Trainer. Both are stored as ``self.dm_`` and ``self.model_``.

        Args:
            train_smiles: Training SMILES strings.
            train_y: Training targets, shape [n_samples, n_targets] or [n_samples].
            val_smiles: Optional validation SMILES (for metric monitoring).
            val_y: Optional validation targets.

        Returns:
            self
        """
        self.dm_ = MolPropDataModule(batch_size=self.batch_size)
        self.dm_.setup_fit(train_smiles, train_y, val_smiles, val_y)

        self.model_ = MolPropModule(
            n_targets=self.n_targets,
            d_h=self.d_h,
            depth=self.depth,
            ffn_hidden_dim=self.ffn_hidden_dim,
            ffn_n_layers=self.ffn_n_layers,
            dropout=self.dropout,
            init_lr=self.init_lr,
            max_lr=self.max_lr,
            final_lr=self.final_lr,
            warmup_epochs=self.warmup_epochs,
            y_mean=self.dm_.y_mean_,
            y_std=self.dm_.y_std_,
        )

        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            enable_model_summary=False,
            enable_progress_bar=True,
            logger=True,
        )
        trainer.fit(self.model_, self.dm_)
        return self

    # ------------------------------------------------------------------ predict

    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """
        Return predictions in the original (un-normalised) target scale.

        A fresh Trainer is created for each call so that this method works
        after load() without requiring a previously fitted Trainer instance.

        Args:
            smiles_list: SMILES strings to predict.

        Returns:
            np.ndarray of shape [n_samples, n_targets].
        """
        pred_dm = MolPropDataModule(batch_size=self.batch_size)
        pred_dm.load_state_dict(self.dm_.state_dict())
        pred_dm.setup_predict(smiles_list)

        # Pass the dataloader directly instead of the DataModule to avoid
        # Lightning trying to re-instantiate ChemProp's custom DataLoader
        # (which conflicts with Lightning's sampler-injection mechanism).
        pred_dl = pred_dm.predict_dataloader()

        trainer = L.Trainer(
            accelerator=self.accelerator,
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
        )
        preds = trainer.predict(self.model_, dataloaders=pred_dl)
        return torch.cat(preds, dim=0).cpu().numpy()

    # ------------------------------------------------------------------ save / load

    def save(self, path: str) -> None:
        """
        Save model weights, architecture hyperparameters, and DataModule
        normalisation statistics to disk.

        Three files are written:
          - ``{path}_model.pt``      — model weights (state_dict)
          - ``{path}_hparams.pt``    — architecture hyperparameters and normalisation
          - ``{path}_datamodule.pt`` — DataModule state (normalisation statistics)

        The DataModule file can be loaded independently via
        ``MolPropDataModule.load("{path}_datamodule.pt")``.
        """
        if not hasattr(self, "model_"):
            raise ValueError("No fitted model found. Call fit() before save().")
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model_.state_dict(), f"{path}_model.pt")

        torch.save(
            {
                **self.model_.arch_hparams,
                "y_mean": self.dm_.y_mean_,
                "y_std":  self.dm_.y_std_,
            },
            f"{path}_hparams.pt",
        )

        self.dm_.save(f"{path}_datamodule.pt")

    @classmethod
    def load(cls, path: str, batch_size: int = 64, accelerator: str = "auto") -> "ChemPropPredictor":
        """
        Restore a ChemPropPredictor from disk.

        Args:
            path:        Base path without suffix, matching what was used in save().
            batch_size:  Batch size for predict(); does not affect model weights.
            accelerator: Lightning accelerator for predict().

        Returns:
            Restored ChemPropPredictor ready for predict().
        """
        hparams = torch.load(f"{path}_hparams.pt", weights_only=False)
        y_mean = hparams.pop("y_mean")
        y_std  = hparams.pop("y_std")

        instance = cls(**hparams, batch_size=batch_size, accelerator=accelerator)

        # Restore DataModule (normalisation statistics)
        instance.dm_ = MolPropDataModule.load(f"{path}_datamodule.pt", batch_size=batch_size)

        # Rebuild model architecture and load weights
        instance.model_ = MolPropModule(**hparams, y_mean=y_mean, y_std=y_std)
        instance.model_.load_state_dict(
            torch.load(f"{path}_model.pt", weights_only=True)
        )
        instance.model_.eval()

        return instance
