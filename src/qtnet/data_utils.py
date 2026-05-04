"""
Data utilities for molecular datasets, including CV splits, statistics
computation, and normalization.
"""
import math
import warnings
from typing import Optional, List, Tuple, Any
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDetermineBonds import DetermineConnectivity, DetermineBondOrders, DetermineBonds
from rdkit.Chem.Scaffolds.MurckoScaffold import (
    MurckoScaffoldSmiles,
    MakeScaffoldGeneric,
)

#import networkx as nx
#from networkx.algorithms import isomorphism

# Frobenius norm weights for 5-component representation:
# ||T||²_F = 2*xy² + 2*xz² + 2*yz² + 2*aniso² + 1.5*zz²
FROBENIUS_WEIGHTS_L2 = np.array([2.0, 2.0, 2.0, 2.0, 1.5], dtype=np.float32)
BOHR_TO_ANGSTROM = 0.529177210544
ATOM_TARGETS = ["N", "LI", "Mu_X", "Mu_Y", "Mu_Z", "Q_XY", "Q_XZ",
                "Q_YZ", "Q_aniso", "Q_ZZ"]
MOLECULAR_PROPERTIES_PRED = ['alpha', 'gap', 'U0', 'Cv']



def get_scaffold(smiles: str, generic: bool = False) -> str:
    """ Generate the Bemis-Murcko scaffold for a given molecule.

    Args:
        smiles (str): A SMILES string or an RDKit molecule object representing the
                molecule for which to generate the scaffold.
        generic (bool): If True, generate a generic scaffold (all atom types replaced
                with carbon and all bond types set to single).
    
    Returns:
        str: A SMILES string representing the Bemis-Murcko scaffold of the input
             molecule. If the scaffold cannot be generated, the input SMILES
             string is returned.
    """
    if generic:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaffold = Chem.MolToSmiles(MakeScaffoldGeneric(mol))
    else:
        scaffold = MurckoScaffoldSmiles(smiles)
    if len(scaffold) == 0:
        scaffold = smiles
    return scaffold


def create_cv_splits(
    ds: pd.DataFrame,
    n_splits: int = 5,
    n_repeats: int = 5,
    group_col: Optional[str] = None,
    base_seed: int = 42,
    training_fractions: Optional[List[float]] = None,
    val_fraction: float = 0.0,
):
    """Create repeated k-fold cross-validation splits.

    Args:
        ds: Dataset or DataFrame to split.
        n_splits: Number of folds.
        n_repeats: Number of repeats.
        group_col: Column name for grouping (if None, uses standard K-Fold).
        base_seed: Base random seed.
        training_fractions: List of training-set fractions to subsample at each
            fold (e.g. [0.1, 0.5, 1.0]).  When val_fraction > 0 the fraction
            is applied to the *trainable* pool (full_train minus val).
            Subsampling is deterministic:
            seed = base_seed + global_fold * 1000 + int(frac * 1000).
            Defaults to [1.0] (full trainable set, backward-compatible).
        val_fraction: Fraction of the fold's full training pool to reserve as a
            **fold-specific validation set**, carved out before fraction
            subsampling.  The same ``val_idx`` is returned for every training
            fraction within a fold.
            seed = base_seed + global_fold * 100 + 7.
            Defaults to 0.0 (backward-compatible: no val split, ``val_idx``
            is an empty array and ``train_idx`` == ``full_train_idx`` at
            fraction 1.0).

    Yields:
        Dictionary with repeat, cv_fold, fold, fraction, train_idx,
        full_train_idx, val_idx, test_idx.
        - ``full_train_idx`` : all non-test indices for this fold (unchanged).
        - ``val_idx``        : val_fraction of full_train_idx; empty array
                               when val_fraction == 0.
        - ``train_idx``      : ``fraction`` of the trainable pool
                               (full_train minus val); equals the full
                               trainable pool when fraction == 1.0.
    """
    if training_fractions is None:
        training_fractions = [1.0]

    if group_col is None:
        groups = np.zeros(len(ds))
    else:
        groups = np.array(ds[group_col])

    for repeat in range(n_repeats):
        seed = base_seed + repeat
        if group_col is None:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            kf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold, (full_train_idx, test_idx) in enumerate(
            kf.split(X=np.zeros(len(groups)), groups=groups)
        ):
            global_fold = repeat * n_splits + fold

            # Optionally carve a fold-specific validation set.
            # val_idx is identical for every training fraction within this fold.
            if val_fraction > 0.0:
                val_seed = base_seed + global_fold * 100 + 7
                rng_val = np.random.default_rng(val_seed)
                n_val = max(1, int(len(full_train_idx) * val_fraction))
                val_idx = np.sort(
                    rng_val.choice(full_train_idx, size=n_val, replace=False)
                )
                val_set = set(val_idx.tolist())
                trainable_idx = np.array(
                    [i for i in full_train_idx if i not in val_set],
                    dtype=full_train_idx.dtype,
                )
            else:
                val_idx = np.array([], dtype=full_train_idx.dtype)
                trainable_idx = full_train_idx

            for fraction in training_fractions:
                if fraction < 1.0:
                    frac_seed = base_seed + global_fold * 1000 + int(fraction * 1000)
                    rng = np.random.default_rng(frac_seed)
                    n_train = max(1, int(len(trainable_idx) * fraction))
                    train_idx = np.sort(
                        rng.choice(trainable_idx, size=n_train, replace=False)
                    )
                else:
                    train_idx = trainable_idx
                yield {
                    "repeat": repeat,
                    "cv_fold": fold,
                    "fold": global_fold,
                    "fraction": fraction,
                    "train_idx": train_idx,
                    "full_train_idx": full_train_idx,
                    "val_idx": val_idx,
                    "test_idx": test_idx,
                }


def compute_molecular_stats(ds: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Compute mean and standard deviation for the specified columns.

    Args:
        ds: pandas DataFrame containing the data.
        columns: list of column names to compute stats for. If None,
                 defaults to ['gap', 'alpha', 'U0', 'Cv'].

    Returns:
        DataFrame with index ['mean', 'std'] and columns equal to `columns`.
    """
    if columns is None:
        columns = ['gap', 'alpha', 'U0', 'Cv']

    missing = [c for c in columns if c not in ds.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    stats = ds[columns].agg(['mean', 'std'])
    return stats


def compute_per_atom_stats(
    ds: pd.DataFrame,
    atom_col: str = 'atoms',
    n_col: str = 'N',
    li_col: str = 'LI',
    mu_prefix: str = 'Mu_',
    q_prefix: str = 'Q_',
) -> pd.DataFrame:
    """
    For each atom type (values found in `atom_col`) compute:
      - mean & std of `n_col` and `li_col`
      - RMS of per-atom `Mu_` vectors (RMS of vector magnitude)
      - RMS of per-atom `Q_` tensors (RMS of Frobenius norm)

    The function will automatically fall back to a column named `'atom'` if
    `atom_col='atoms'` is not present in `ds`.
    """

    # allow either 'atoms' or 'atom' (common dataset variants)
    if atom_col not in ds.columns:
        if 'atom' in ds.columns:
            atom_col = 'atom'
        elif 'atoms' in ds.columns:
            atom_col = 'atoms'
        else:
            raise KeyError(f"Required column not found in DataFrame: {atom_col} (also checked 'atom' and 'atoms')")

    # detect Mu_ and Q_ columns
    mu_cols = [c for c in ds.columns if c.startswith(mu_prefix)]
    q_cols = [c for c in ds.columns if c.startswith(q_prefix)]

    # checks for required scalar-per-atom columns
    for required in (n_col, li_col):
        if required not in ds.columns:
            raise KeyError(f"Required column not found in DataFrame: {required}")

    # accumulators per atom type
    N_vals = defaultdict(list)
    LI_vals = defaultdict(list)
    mu_norm_sq_vals = defaultdict(list)
    q_frob_sq_vals = defaultdict(list)

    for idx, row in ds.iterrows():
        atoms = list(row[atom_col])
        n_arr = list(row[n_col])
        li_arr = list(row[li_col])

        L = len(atoms)
        if not (len(n_arr) == len(li_arr) == L):
            raise ValueError(f"Per-atom arrays length mismatch at index {idx}: atoms({L}), N({len(n_arr)}), LI({len(li_arr)})")

        # prepare mu and q component arrays for this row
        mu_comp_arrays = [list(row[c]) for c in mu_cols] if mu_cols else []
        q_comp_arrays = [list(row[c]) for c in q_cols] if q_cols else []

        # sanity for mu/q lengths
        for arr in mu_comp_arrays + q_comp_arrays:
            if len(arr) != L:
                raise ValueError(f"Per-atom component length mismatch at index {idx}")

        for i, at in enumerate(atoms):
            # N and LI
            try:
                N_vals[at].append(float(n_arr[i]))
            except Exception:
                N_vals[at].append(float('nan'))
            try:
                LI_vals[at].append(float(li_arr[i]))
            except Exception:
                LI_vals[at].append(float('nan'))

            # Mu vector norm-square (sum of squares over Mu components)
            if mu_comp_arrays:
                mu_sq = 0.0
                for comp in mu_comp_arrays:
                    v = comp[i]
                    mu_sq += float(v) * float(v)
                mu_norm_sq_vals[at].append(mu_sq)

            # Q tensor Frobenius-norm-square using proper 5-component weights
            # first collect values by name for this atom
            if q_comp_arrays:
                comp_map = {name: comp[i] for name, comp in zip(q_cols, q_comp_arrays)}

                # helper to safely pull float or 0
                def _get(name):
                    return float(comp_map.get(name, 0.0))

                # components
                xy = _get('Q_XY')
                xz = _get('Q_XZ')
                yz = _get('Q_YZ')
                # anisotropy: either provided or computed from Q_XX/Q_YY
                if 'Q_aniso' in comp_map:
                    aniso = _get('Q_aniso')
                else:
                    aniso = (_get('Q_XX') - _get('Q_YY')) / 2.0
                zz = _get('Q_ZZ')

                tensor = np.array([xy,xz,yz,aniso,zz])
                # weighted Frobenius norm squared (using same weights as FROBENIUS_WEIGHTS_L2)
                q = compute_frobenius_norm(tensor)
                q_sq = q**2
                q_frob_sq_vals[at].append(q_sq)

    # build result
    rows = []
    atom_types = sorted(set(list(N_vals.keys()) + list(LI_vals.keys())))
    for at in atom_types:
        n_series = pd.Series(N_vals[at]).dropna()
        li_series = pd.Series(LI_vals[at]).dropna()

        mu_list = mu_norm_sq_vals.get(at, [])
        q_list = q_frob_sq_vals.get(at, [])

        mu_rms = math.sqrt(float(pd.Series(mu_list).mean())) if mu_list else float('nan')
        q_rms = math.sqrt(2/3*float(pd.Series(q_list).mean())) if q_list else float('nan')

        rows.append(
            {
                'atom': at,
                'count': int(n_series.shape[0]),
                'N_mean': float(n_series.mean()) if not n_series.empty else float('nan'),
                'N_std': float(n_series.std()) if not n_series.empty else float('nan'),
                'LI_mean': float(li_series.mean()) if not li_series.empty else float('nan'),
                'LI_std': float(li_series.std()) if not li_series.empty else float('nan'),
                'Mu_rms': mu_rms,
                'Q_rms': q_rms,
            }
        )

    result = pd.DataFrame(rows).set_index('atom')

    # Compute element-frequency weights: w_el = sqrt(total / count_el),
    # normalised so that the mean weight across elements is 1.
    total = result['count'].sum()
    raw_w = np.sqrt(total / result['count'].clip(lower=1).values)
    result['weight'] = raw_w / raw_w.mean()

    return result


def apply_z_regularization(
    ds: pd.DataFrame,
    mol_stats: pd.DataFrame,
    per_atom_stats: pd.DataFrame,
    mol_cols: Optional[List[str]] = None,
    atom_col: str = 'atoms',
    n_col: str = 'N',
    li_col: str = 'LI',
    mu_prefix: str = 'Mu_',
    q_prefix: str = 'Q_',
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Normalize dataset using provided statistics.

    - Z-regularize molecular-level scalar columns using `mol_stats` (index: 'mean','std').
    - For each atom in the per-row `atoms` list, z-regularize `N` and `LI` using
      per-atom means/stds from `per_atom_stats` (columns: 'N_mean','N_std','LI_mean','LI_std').
    - Divide each per-atom `Mu_*` component by the corresponding per-atom `Mu_rms`.
    - Divide each per-atom `Q_*` component by the corresponding per-atom `Q_rms` (where `Q_rms` is
      the root-mean-square Frobenius norm computed from the 5-component representation
      [xy,xz,yz,aniso,zz], ensuring correct off-diagonal weighting).

    Args:
        ds: DataFrame containing molecular scalars and per-atom list-columns.
        mol_stats: output from `compute_molecular_stats` (index must include 'mean' and 'std').
        per_atom_stats: output from `compute_per_atom_stats` (index=atom types).
        mol_cols: which molecular columns to z-normalize (defaults to columns in `mol_stats`).
        inplace: modify `ds` in-place when True (default).

    Returns:
        The normalized DataFrame (same object if `inplace=True`, otherwise a copy).

    Behavior notes:
      - If a per-atom std or rms is zero/NaN, that value/component becomes 0.0 (avoids divide-by-zero).
      - If an atom type is missing in `per_atom_stats` its per-atom entries become NaN and a
        single warning is emitted listing the missing types.
    """
    if not inplace:
        ds = ds.copy()

    # validate mol_stats
    if 'mean' not in mol_stats.index or 'std' not in mol_stats.index:
        raise ValueError("`mol_stats` must have index ['mean','std']")

    if mol_cols is None:
        mol_cols = list(mol_stats.columns)

    # molecular-level z-score (vectorized)
    for col in mol_cols:
        if col not in ds.columns:
            continue
        mean = float(mol_stats.at['mean', col])
        std = float(mol_stats.at['std', col]) if not pd.isna(mol_stats.at['std', col]) else 0.0
        if std == 0.0:
            ds[col] = ds[col].apply(lambda _: 0.0 if pd.notna(_) else _)
        else:
            ds[col] = (ds[col] - mean) / std

    # resolve atom column name if needed
    if atom_col not in ds.columns:
        if 'atom' in ds.columns:
            atom_col = 'atom'
        elif 'atoms' in ds.columns:
            atom_col = 'atoms'
        else:
            raise KeyError(f"Neither '{atom_col}' nor 'atom'/'atoms' found in DataFrame")

    # required columns in per_atom_stats
    required = {'N_mean', 'N_std', 'LI_mean', 'LI_std', 'Mu_rms', 'Q_rms'}
    if not required.issubset(set(per_atom_stats.columns)):
        missing = required - set(per_atom_stats.columns)
        raise KeyError(f"per_atom_stats missing required columns: {sorted(missing)}")

    mu_cols = [c for c in ds.columns if c.startswith(mu_prefix)]
    q_cols = [c for c in ds.columns if c.startswith(q_prefix)]

    missing_atom_types = set()

    # normalize per-row per-atom arrays
    for idx, row in ds.iterrows():
        atoms = list(row[atom_col])

        # N
        if n_col in ds.columns:
            n_vals = list(row[n_col])
            if len(n_vals) != len(atoms):
                raise ValueError(f"Row {idx}: length mismatch between '{atom_col}' and '{n_col}'")
            new_n = []
            for at, v in zip(atoms, n_vals):
                if at in per_atom_stats.index:
                    mean = per_atom_stats.at[at, 'N_mean']
                    std = per_atom_stats.at[at, 'N_std']
                    if pd.isna(std) or std == 0:
                        new_n.append(0.0)
                    else:
                        new_n.append((float(v) - float(mean)) / float(std))
                else:
                    missing_atom_types.add(at)
                    new_n.append(float('nan'))
            ds.at[idx, n_col] = new_n

        # LI
        if li_col in ds.columns:
            li_vals = list(row[li_col])
            if len(li_vals) != len(atoms):
                raise ValueError(f"Row {idx}: length mismatch between '{atom_col}' and '{li_col}'")
            new_li = []
            for at, v in zip(atoms, li_vals):
                if at in per_atom_stats.index:
                    mean = per_atom_stats.at[at, 'LI_mean']
                    std = per_atom_stats.at[at, 'LI_std']
                    if pd.isna(std) or std == 0:
                        new_li.append(0.0)
                    else:
                        new_li.append((float(v) - float(mean)) / float(std))
                else:
                    missing_atom_types.add(at)
                    new_li.append(float('nan'))
            ds.at[idx, li_col] = new_li

        # Mu components
        for mu_c in mu_cols:
            comp_vals = list(row[mu_c])
            if len(comp_vals) != len(atoms):
                raise ValueError(f"Row {idx}: length mismatch between '{atom_col}' and '{mu_c}'")
            new_comp = []
            for at, v in zip(atoms, comp_vals):
                if at in per_atom_stats.index:
                    rms = per_atom_stats.at[at, 'Mu_rms']
                    if pd.isna(rms) or rms == 0:
                        new_comp.append(0.0)
                    else:
                        new_comp.append(float(v) / float(rms))
                else:
                    missing_atom_types.add(at)
                    new_comp.append(float('nan'))
            ds.at[idx, mu_c] = new_comp

        # Q components
        for q_c in q_cols:
            comp_vals = list(row[q_c])
            if len(comp_vals) != len(atoms):
                raise ValueError(f"Row {idx}: length mismatch between '{atom_col}' and '{q_c}'")
            new_comp = []
            for at, v in zip(atoms, comp_vals):
                if at in per_atom_stats.index:
                    rms = per_atom_stats.at[at, 'Q_rms']
                    if pd.isna(rms) or rms == 0:
                        new_comp.append(0.0)
                    else:
                        new_comp.append(float(v) / float(rms))
                else:
                    missing_atom_types.add(at)
                    new_comp.append(float('nan'))
            ds.at[idx, q_c] = new_comp

    if missing_atom_types:
        warnings.warn(
            f"apply_z_regularization: missing per-atom stats for atom types: {sorted(missing_atom_types)}",
            UserWarning,
        )

    return ds


# =============================================================================
# L=2 Tensor Representation Utilities
# =============================================================================

def cartesian_to_5comp(Q_XX, Q_XY, Q_XZ, Q_YY, Q_YZ, Q_ZZ):
    """
    Convert Cartesian 6-component quadrupole to 5-component L=2 representation.
    
    The 5-component representation [xy, xz, yz, (xx-yy)/2, zz] is derived from
    a 3x3 symmetric traceless tensor. It is inherently traceless.
    
    Args:
        Q_XX, Q_XY, Q_XZ, Q_YY, Q_YZ, Q_ZZ: Cartesian components (arrays or scalars)
        
    Returns:
        Array of shape (..., 5) with [xy, xz, yz, aniso, zz]
        where aniso = (xx - yy) / 2
    """
    xy = Q_XY
    xz = Q_XZ
    yz = Q_YZ
    aniso = (Q_XX - Q_YY) / 2
    zz = Q_ZZ
    return np.stack([xy, xz, yz, aniso, zz], axis=-1)


def comp5_to_cartesian(tensor_5):
    """
    Convert 5-component L=2 representation to Cartesian 6-component.
    
    Reconstructs the full traceless symmetric tensor from the 5-component
    representation using:
        xx = aniso - zz/2
        yy = -aniso - zz/2
        (which satisfies xx + yy + zz = 0)
    
    Args:
        tensor_5: Array of shape (..., 5) with [xy, xz, yz, aniso, zz]
        
    Returns:
        Array of shape (..., 6) with [XX, XY, XZ, YY, YZ, ZZ]
    """
    xy = tensor_5[..., 0]
    xz = tensor_5[..., 1]
    yz = tensor_5[..., 2]
    aniso = tensor_5[..., 3]
    zz = tensor_5[..., 4]
    
    # Reconstruct diagonal: xx = aniso - zz/2, yy = -aniso - zz/2
    xx = aniso - zz / 2
    yy = -aniso - zz / 2
    
    return np.stack([xx, xy, xz, yy, yz, zz], axis=-1)


def compute_frobenius_norm(
    tensor: np.ndarray,
) -> np.ndarray:
    """
    Compute Frobenius norm of L=2 tensor in 5-component representation.
    
    Args:
        tensor: Tensor of shape (..., 5) in [xy, xz, yz, (xx-yy)/2, zz] format
        weighted: Whether to use proper Frobenius weights (default: True)
        
    Returns:
        Frobenius norm of shape (...)
    """
    weighted_sq = FROBENIUS_WEIGHTS_L2 * tensor ** 2
    return np.sqrt(np.sum(weighted_sq, axis=-1) + 1e-8)


# ==============================================================================
# Utilities for building RDKit mols from QM9 rows, including bond inference and
# SMILES consistency checks. This will be used to "explode" the dataset with one
# atom per row and also to build 3D mols for visualization and sanity checks.
# ==============================================================================

def canonical_smiles(smi: str) -> str:
    """Canonicalize a SMILES string, stripping stereochemistry for comparison."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    # Remove stereo for comparison — QM9 SMILES and inferred ones may differ
    # on stereo annotation without being chemically wrong
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def mol_from_qm9_row(row: pd.Series) -> Chem.Mol:
    """
    Build an RDKit mol from a QM9 dataframe row, preserving QM9 atom ordering.
    Coordinates are converted from Bohr to Angstrom before bond detection.

    Tier 1: DetermineBonds from geometry (handles ~99% of QM9)
    Tier 2: Connectivity from geometry + bond orders from SMILES via
            graph isomorphism (fallback for unusual bonding patterns)

    Raises ValueError with a descriptive message if both tiers fail or if the
    inferred SMILES does not match the reference after canonicalization.
    """
    elements = list(row['atom'])
    coords   = list(zip(
        np.array(row['position_x']) * BOHR_TO_ANGSTROM,
        np.array(row['position_y']) * BOHR_TO_ANGSTROM,
        np.array(row['position_z']) * BOHR_TO_ANGSTROM,
    ))
    #TODO: if distances are converted to Å, dipole and quadrupole need to be
    #converted too! dipole *= BOHR_TO_ANGSTROM, quad *= BOHR_TO_ANGSTROM**2

    # ── Build mol with atoms in QM9 order ─────────────────────────────────────
    em = Chem.EditableMol(Chem.Mol())
    for symbol in elements:
        em.AddAtom(Chem.Atom(symbol))
    rdk_mol = em.GetMol()

    conf = Chem.Conformer(rdk_mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (x, y, z))
    rdk_mol.AddConformer(conf, assignId=True)

    # ── Tier 1: full bond determination from geometry ─────────────────────────
    mol_charge = Chem.GetFormalCharge(Chem.MolFromSmiles(row['smiles']))
    tier1_error = None
    try:
        mol_out = Chem.RWMol(rdk_mol)
        DetermineBonds(mol_out, charge=mol_charge)
        Chem.SanitizeMol(mol_out)
        result = mol_out.GetMol()
    except Exception as e:
        tier1_error = e
        result = None

    # ── Tier 2: topology from geometry + bond orders from SMILES ──────────────
    tier2_error = None
    if result is None:
        try:
            mol_topo = Chem.RWMol(rdk_mol)
            DetermineConnectivity(mol_topo)
            mol_smi = Chem.AddHs(Chem.MolFromSmiles(row['smiles']))
            result = _transfer_bond_orders(mol_topo.GetMol(), mol_smi)
        except Exception as e:
            tier2_error = e

    if result is None:
        raise ValueError(
            f"All attempts failed for mol {row.name}.\n"
            f"  Tier 1: {tier1_error}\n"
            f"  Tier 2: {tier2_error}"
        )

    # ── SMILES consistency check ───────────────────────────────────────────────
    inferred_smi = canonical_smiles(Chem.MolToSmiles(result))
    reference_smi = canonical_smiles(row['smiles'])
    if inferred_smi != reference_smi:
        raise ValueError(
            f"SMILES mismatch for mol {row.name}.\n"
            f"  Inferred : {inferred_smi}\n"
            f"  Reference: {reference_smi}"
        )

    return result


def _transfer_bond_orders(mol_topo: Chem.Mol, mol_smi: Chem.Mol) -> Chem.Mol:
    """ Transfer bond orders from mol_smi to mol_topo by finding the graph
        isomorphism between them using only element labels (no bond orders
        needed).
        
    Args:
        mol_topo (Chem.Mol): Has correct QM9 atom ordering + connectivity but wrong bond orders.
        mol_smi (Chem.Mol): Has correct bond orders but SMILES atom ordering.
    
    Returns:
        A mol with QM9 ordering AND correct bond orders.
    """
    # Build networkx graphs labeled by element only
    def to_nx(mol):
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), el=atom.GetSymbol())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        return G

    G_topo = to_nx(mol_topo)
    G_smi  = to_nx(mol_smi)

    nm = isomorphism.categorical_node_match('el', 'C')
    gm = isomorphism.GraphMatcher(G_topo, G_smi, node_match=nm)

    if not gm.is_isomorphic():
        raise ValueError("Graph isomorphism failed — topology/SMILES mismatch")

    # iso[qm9_idx] = smiles_idx
    iso = next(gm.isomorphisms_iter())

    rw = Chem.RWMol(mol_topo)
    for bond in mol_smi.GetBonds():
        i_smi = bond.GetBeginAtomIdx()
        j_smi = bond.GetEndAtomIdx()
        # Invert iso to get smiles_idx → qm9_idx
        inv = {v: k for k, v in iso.items()}
        if i_smi in inv and j_smi in inv:
            b = rw.GetBondBetweenAtoms(inv[i_smi], inv[j_smi])
            if b is not None:
                b.SetBondType(bond.GetBondType())

    Chem.SanitizeMol(rw)
    return rw.GetMol()


def verify_atom_order(mol: Chem.Mol, row: pd.Series) -> bool:
    """
    Sanity check: confirm that element symbols in the RDKit mol
    match the QM9 `atom` column element-by-element.
    Always call this after mol_from_qm9_coords during development.
    """
    rdkit_elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    qm9_elements   = list(row['atom'])

    if rdkit_elements != qm9_elements:
        raise ValueError(
            f"Atom order mismatch for mol {row.name}:\n"
            f"  RDKit: {rdkit_elements}\n"
            f"  QM9:   {qm9_elements}"
        )
    return True


def atom_features(atom: Any, elem_vocab: List[str]) -> List[float]:
    """ Compute a feature vector for an atom, including:
        - One-hot encoding of the atom type (based on the element vocabulary)
        - Atomic number, degree, formal charge, number of implicit Hs, aromaticity, ring membership, mass, total valence, and hybridization (SP, SP2, SP3)
        
        The element vocabulary should be built from the dataset to ensure consistent ordering.
        
    Args:
        atom (RDKit Atom): The atom for which to compute features.
        elem_vocab (List[str]): List of element symbols for one-hot encoding.
        
    Returns:
        List[float]: The feature vector for the atom.
    """
    atom_symbol = atom.GetSymbol()
    return [
        *[int(atom_symbol == el) for el in elem_vocab], # List of 0/1
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetNumImplicitHs(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetMass(),
        atom.GetTotalValence(),
        int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP),
        int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
        int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3),
    ]


def build_atom_dataset(
        df: pd.DataFrame,
        fp_size: int = 1024,
        radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Build an atom-level dataset from the QM9 DataFrame, where each row
        corresponds to one atom. The feature vector for each atom includes:
          - One-hot encoding of the atom type (based on the element vocabulary)
          - Atomic number, degree, formal charge, number of implicit Hs, aromaticity, ring membership, mass, total valence, and hybridization (SP, SP2, SP3)
          - 3D coordinates (x, y, z) from the QM9 dataset
          - Global Morgan fingerprint of the molecule (computed from the SMILES)
          
        The target vector for each atom includes the per-atom properties (N, LI, Mu_*, Q_*) aligned with the atom ordering in the QM9 dataset.
        
    Args:
        df (pd.DataFrame): QM9 dataset as a DataFrame.
        fp_size (int): Size of the Morgan fingerprint.
        radius (int): Radius for the Morgan fingerprint.
        
    Returns:
        Tuple of (X, y, mol_ids):
          - X: np.ndarray of shape (num_atoms, num_features) containing the feature vectors
          - y: np.ndarray of shape (num_atoms, num_targets) containing the target vectors
          - mol_ids: np.ndarray of shape (num_atoms,) containing the molecule ID for each atom (corresponding to the index in the original DataFrame)    
    """
    # Build element vocabulary from the dataset (sorted for consistency)
    elem_vocab = sorted(set(
        [str(el) for row in df['atom'] for el in row]
    ))

    # Initialize Morgan fingerprint generator with chirality
    fpgen = AllChem.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
        includeChirality=True,
    )
    
    X_rows, y_rows, mol_ids = [], [], []

    for idx, row in df.iterrows():
        # ── Mol preserving QM9 atom ordering ──────────────────────────────
        try:
            mol_3d = mol_from_qm9_row(row)
            verify_atom_order(mol_3d, row)
        except Exception as e:
            print(f"Skipping mol {idx}: {e}")
            continue

        # ── SMILES mol used ONLY for the global Morgan fingerprint ─────────
        mol_smi = Chem.MolFromSmiles(row['smiles'])
        if mol_smi is None:
            continue
        
        global_fp = fpgen.GetFingerprintAsNumPy(mol_smi)

        # ── Targets — index i aligns with mol_3d atom i by construction ───
        targets = {t: np.array(row[t]) for t in ATOM_TARGETS if t in row}
        xs = np.array(row['position_x'])
        ys = np.array(row['position_y'])
        zs = np.array(row['position_z'])

        # For each atom, build feature vector and target vector
        for ai in range(mol_3d.GetNumAtoms()):
            atom = mol_3d.GetAtomWithIdx(ai)
            coords = [xs[ai], ys[ai], zs[ai]]
            x = np.concatenate([
                atom_features(atom, elem_vocab),
                coords,
                global_fp,
            ])
            y = [targets[t][ai] if t in targets else np.nan for t in ATOM_TARGETS]
            X_rows.append(x)
            y_rows.append(y)
            mol_ids.append(idx)

    return np.array(X_rows), np.array(y_rows), np.array(mol_ids)