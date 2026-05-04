#!/usr/bin/env python3
"""
Reconstruct molecular dipole magnitudes from inferred per-atom AIM properties.

Uses the QTAIM decomposition (atomic units, positions in Bohr):

    mu_mol = sum_i (Z_i - N_i) * R_i   +   sum_i mu_i_atomic

where:
  Z_i        = nuclear charge (from element symbol)
  N_i        = electron population (inferred or ground-truth)
  R_i        = atom position vector (Bohr)
  mu_i       = per-atom intrinsic dipole vector (e·Bohr)

For neutral molecules sum_i(Z_i - N_i) = 0, so the charge-transfer term is
origin-independent.  The result is converted to Debye and compared to QM9's
ground-truth 'mu' column (B3LYP/6-31G(2df,p) total dipole magnitude).

Supports three modes:
  1. Single pkl  (default)         : computes 'mu_inferred'
  2. --per-architecture            : computes 'mu_inferred_SGN2' and
                                     'mu_inferred_EGNX' from two separate pkls
                                     (each produced by infer_QTAIM_QM9.py with
                                     --model-type SGN2 / EGNX respectively)
  3. --aimel                       : run on aimel_clustered_molecular.pkl using
                                     true AIM properties to calibrate the formula

Usage:
  # Default — single ensemble pkl
  python molecule_dipoles_from_inferred.py

  # Per-architecture comparison
  python molecule_dipoles_from_inferred.py --per-architecture \\
      --sgn2-pkl data_curation/molecular/qm9_inferred_SGN2.pkl \\
      --egnx-pkl data_curation/molecular/qm9_inferred_EGNX.pkl

  # Calibration against true AIM (sanity check)
  python molecule_dipoles_from_inferred.py --aimel \\
      --inferred-pkl data_curation/molecular/aimel_clustered_molecular.pkl \\
      --output-pkl /tmp/aimel_dipoles.pkl
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repo root (needed only for default path construction)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, 'data_curation')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(os.path.join(start_dir, '..', '..'))
        cur = parent


REPO_ROOT = _find_repo_root(SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
# 1 e·Bohr = 2.541746473 Debye  (CODATA 2018)
EBOHR_TO_DEBYE = 2.541746473

# Nuclear charges for QM9 / AIMEl elements
NUCLEAR_CHARGE = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_mu_inferred(row, atom_col: str) -> float:
    """Compute molecular dipole magnitude (Debye) from one DataFrame row.

    Expects per-atom list columns: N, Mu_X, Mu_Y, Mu_Z.
    Expects per-atom list column for element symbols: atom_col.
    Expects per-atom list columns: position_x, position_y, position_z (Bohr).

    Returns NaN on any error.
    """
    try:
        elements = list(row[atom_col])
        Z = np.array([NUCLEAR_CHARGE[e] for e in elements], dtype=np.float64)
        N = np.array(list(row['N']),     dtype=np.float64)
        Rx = np.array(list(row['position_x']), dtype=np.float64)
        Ry = np.array(list(row['position_y']), dtype=np.float64)
        Rz = np.array(list(row['position_z']), dtype=np.float64)
        mux = np.array(list(row['Mu_X']), dtype=np.float64)
        muy = np.array(list(row['Mu_Y']), dtype=np.float64)
        muz = np.array(list(row['Mu_Z']), dtype=np.float64)

        # Charge-transfer term: sum_i q_i * R_i  (q_i = Z_i - N_i)
        q = Z - N
        ct = np.array([
            (q * Rx).sum(),
            (q * Ry).sum(),
            (q * Rz).sum(),
        ])

        # Atomic-polarisation term: sum_i mu_i
        ap = np.array([mux.sum(), muy.sum(), muz.sum()])

        mu_total_au = ct + ap
        return float(np.linalg.norm(mu_total_au) * EBOHR_TO_DEBYE)

    except Exception:
        return float('nan')


def _atom_col(df: pd.DataFrame) -> str:
    """Return the name of the per-atom element column."""
    for candidate in ('atom', 'elements', 'element'):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"No element column found in DataFrame. "
        f"Expected one of 'atom', 'elements', 'element'. "
        f"Got: {list(df.columns)}"
    )


def add_mu_inferred(df: pd.DataFrame, col_name: str = 'mu_inferred') -> pd.DataFrame:
    """Compute mu_inferred for every row and attach as a new column."""
    ac = _atom_col(df)
    mu_vals = [compute_mu_inferred(row, ac) for _, row in df.iterrows()]
    df = df.copy()
    df[col_name] = mu_vals
    return df


def print_summary(df: pd.DataFrame, pred_col: str, ref_col: str = 'mu'):
    """Print MAE and Pearson r between pred_col and ref_col."""
    if ref_col not in df.columns:
        print(f"  Reference column '{ref_col}' not found; skipping summary.")
        return
    mask = df[pred_col].notna() & df[ref_col].notna()
    pred = df.loc[mask, pred_col].values.astype(float)
    ref  = df.loc[mask, ref_col].values.astype(float)
    mae = np.abs(pred - ref).mean()
    r   = float(np.corrcoef(pred, ref)[0, 1]) if len(pred) > 1 else float('nan')
    print(f"  {pred_col} vs {ref_col}  (n={mask.sum():,}):  "
          f"MAE={mae:.4f} D   Pearson r={r:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--inferred-pkl', type=str,
        default=os.path.join(
            REPO_ROOT, 'data_curation', 'molecular', 'qm9_inferred.pkl'),
        help='Input pkl with N, Mu_X/Y/Z, position_* columns '
             '(default: qm9_inferred.pkl)',
    )
    parser.add_argument(
        '--output-pkl', type=str, default=None,
        help='Output pkl path (default: overwrite --inferred-pkl in place)',
    )
    parser.add_argument(
        '--per-architecture', action='store_true',
        help='Write separate mu_inferred_SGN2 and mu_inferred_EGNX columns '
             'from two distinct pkl files',
    )
    parser.add_argument(
        '--sgn2-pkl', type=str,
        default=os.path.join(
            REPO_ROOT, 'data_curation', 'molecular', 'qm9_inferred_SGN2.pkl'),
        help='SGN2 inferred pkl (used with --per-architecture)',
    )
    parser.add_argument(
        '--egnx-pkl', type=str,
        default=os.path.join(
            REPO_ROOT, 'data_curation', 'molecular', 'qm9_inferred_EGNX.pkl'),
        help='EGNX inferred pkl (used with --per-architecture)',
    )
    parser.add_argument(
        '--aimel', action='store_true',
        help='Use ground-truth AIM columns (calibration / sanity check mode). '
             'Point --inferred-pkl at aimel_clustered_molecular.pkl.',
    )

    args = parser.parse_args()

    # Resolve output path
    output_pkl = args.output_pkl or args.inferred_pkl

    print("=" * 72)
    print("Molecular dipole reconstruction from per-atom AIM properties")
    print(f"  EBOHR_TO_DEBYE = {EBOHR_TO_DEBYE}")
    print("=" * 72)

    if args.per_architecture:
        # ── Per-architecture mode ────────────────────────────────────────
        print(f"\nPer-architecture mode:")
        print(f"  SGN2 pkl: {args.sgn2_pkl}")
        print(f"  EGNX pkl: {args.egnx_pkl}")

        print("\nLoading SGN2 pkl ...")
        df_sgn2 = pd.read_pickle(args.sgn2_pkl)
        print(f"  {len(df_sgn2)} molecules")
        print("Computing mu_inferred_SGN2 ...")
        df_sgn2 = add_mu_inferred(df_sgn2, col_name='mu_inferred_SGN2')
        print_summary(df_sgn2, 'mu_inferred_SGN2')

        print("\nLoading EGNX pkl ...")
        df_egnx = pd.read_pickle(args.egnx_pkl)
        print(f"  {len(df_egnx)} molecules")
        print("Computing mu_inferred_EGNX ...")
        df_egnx = add_mu_inferred(df_egnx, col_name='mu_inferred_EGNX')
        print_summary(df_egnx, 'mu_inferred_EGNX')

        # Merge both dipole columns onto the SGN2 frame (aligned by index)
        out = df_sgn2.copy()
        out['mu_inferred_EGNX'] = df_egnx['mu_inferred_EGNX']

        # Ensemble mean of the two architectures as a bonus column
        out['mu_inferred'] = (
            out['mu_inferred_SGN2'].fillna(float('nan')) +
            out['mu_inferred_EGNX'].fillna(float('nan'))
        ) / 2.0
        print("\nEnsemble mean (mu_inferred = mean of SGN2 + EGNX):")
        print_summary(out, 'mu_inferred')

    else:
        # ── Single-pkl mode ──────────────────────────────────────────────
        col_name = 'mu_inferred'
        if args.aimel:
            print("\nCalibration mode: using ground-truth AIM properties.")
            col_name = 'mu_reconstructed'

        print(f"\nLoading {args.inferred_pkl} ...")
        out = pd.read_pickle(args.inferred_pkl)
        print(f"  {len(out)} molecules")

        print(f"Computing {col_name} ...")
        out = add_mu_inferred(out, col_name=col_name)

        n_valid = out[col_name].notna().sum()
        n_nan   = out[col_name].isna().sum()
        print(f"  {n_valid:,} valid   {n_nan:,} NaN")
        print_summary(out, col_name)

    # Save
    out.to_pickle(output_pkl)
    print(f"\nSaved → {output_pkl}")
    print("=" * 72)
    print("Done.")


if __name__ == '__main__':
    main()
