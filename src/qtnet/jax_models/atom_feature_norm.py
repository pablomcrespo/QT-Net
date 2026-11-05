"""
Per-element normalisation of the atomic INPUT features of the informed
molecular model (N, LI, Mu_X/Y/Z, Q_XY/XZ/YZ/aniso/ZZ).

Why it exists
-------------
The raw features differ in scale by orders of magnitude between elements (N of
an H basin ≈ 1 e, of an O basin ≈ 9 e), so a pooled statistic is a distortion,
not a rescaling.  The scheme mirrors ``data_utils.apply_z_regularization`` (the
atomic study's target convention), with statistics from
``data_utils.compute_per_atom_stats``:

  * N, LI                  → per-element z-score
  * Mu_* and Q_* components → divided by the per-element RMS of the vector /
                              tensor, WITHOUT mean-centring

Subtracting a mean from a covariant quantity is not rotation-equivariant (it
adds a fixed lab-frame vector); a pure scale commutes with rotation, so
``augmentation.augment_molecular_batches`` stays valid on normalised batches.

Which data the statistics come from
-----------------------------------
A model must keep being fed the scales it was trained with.  The statistics are
fit on the TRAINING split of each fold only, stored in the run's
``config.json``, and re-applied unchanged when the trained model is served other
data (e.g. inferred properties on the QM9 remainder) — never re-fit on the
population being served.
"""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from qtnet.data_utils import compute_per_atom_stats

ELEMENTS = ('H', 'C', 'N', 'O')          # species-index order of the molecular complexes

_ZSCORE = ('N', 'LI')
_MU_SCALED = ('Mu_X', 'Mu_Y', 'Mu_Z')
_Q_SCALED = ('Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ')
_EPS = 1e-8


def atom_feature_norm_stats(df_train: pd.DataFrame,
                            elements: Sequence[str] = ELEMENTS) -> Dict[str, Dict[str, float]]:
    """Per-element statistics, fit on ``df_train`` (the training split) only.

    Returns a JSON-serialisable dict
    ``{element: {'N_mean', 'N_std', 'LI_mean', 'LI_std', 'Mu_rms', 'Q_rms'}}``.
    An element absent from ``df_train`` gets the identity transform.
    """
    cols = ['atom', 'N', 'LI', *_MU_SCALED, *_Q_SCALED]
    per_atom = compute_per_atom_stats(df_train[cols])
    stats: Dict[str, Dict[str, float]] = {}
    for el in elements:
        if el in per_atom.index:
            row = per_atom.loc[el]
            s = {f'{k}_{m}': float(row[f'{k}_{m}']) for k in _ZSCORE for m in ('mean', 'std')}
            s['Mu_rms'] = float(row['Mu_rms'])
            s['Q_rms'] = float(row['Q_rms'])
        else:
            s = {'N_mean': 0.0, 'N_std': 1.0, 'LI_mean': 0.0, 'LI_std': 1.0,
                 'Mu_rms': 1.0, 'Q_rms': 1.0}
        for k, v in list(s.items()):
            if not np.isfinite(v) or (k.endswith(('_std', '_rms')) and abs(v) < _EPS):
                s[k] = 1.0 if k.endswith(('_std', '_rms')) else 0.0
        stats[el] = s
    return stats


def _per_species(stats: Dict[str, Dict[str, float]], elements: Sequence[str],
                 key: str, default: float) -> np.ndarray:
    """Vector of ``key`` indexed by node_static['Z'] (last slot: unknown species)."""
    out = [float(stats.get(el, {}).get(key, default)) for el in elements]
    out.append(default)
    return np.asarray(out, dtype=np.float32)


def normalize_atom_feature_batches(batches: List, stats: Dict[str, Dict[str, float]],
                                   elements: Sequence[str] = ELEMENTS) -> List:
    """Normalise the atomic feature channels of padded molecular batches, in place.

    Operates on the padded batches (freshly allocated by
    ``prepare_padded_batches``), never on the shared complex cache, so no
    statistics can leak between folds.  Call ONCE per split, not per epoch:
    augmentation rotates the same arrays in place.  No-op for blind batches.
    """
    if not batches or 'N' not in batches[0].cochain_batches[0].static:
        return batches

    means = {k: _per_species(stats, elements, f'{k}_mean', 0.0) for k in _ZSCORE}
    stds = {k: _per_species(stats, elements, f'{k}_std', 1.0) for k in _ZSCORE}
    mu_rms = _per_species(stats, elements, 'Mu_rms', 1.0)
    q_rms = _per_species(stats, elements, 'Q_rms', 1.0)

    for batch in batches:
        node_static = batch.cochain_batches[0].static
        z = np.asarray(node_static['Z'], dtype=np.int32)
        # Padding rows carry Z=0 and zero features; keep them exactly zero.
        pad = np.asarray(batch.cochain_batches[0].x_mask).reshape(z.shape) == 0
        for k in _ZSCORE:
            v = (np.asarray(node_static[k], dtype=np.float32) - means[k][z]) / stds[k][z]
            node_static[k] = np.where(pad, 0.0, v).astype(np.float32)
        for keys, scale in ((_MU_SCALED, mu_rms), (_Q_SCALED, q_rms)):
            for k in keys:
                v = np.asarray(node_static[k], dtype=np.float32) / scale[z]
                node_static[k] = np.where(pad, 0.0, v).astype(np.float32)
    return batches
