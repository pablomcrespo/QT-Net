#!/usr/bin/env python3
"""
Environment folds: co-occurrence-closure holdouts over the whole-QM9 6-Bohr
atomic-environment labels (``qm9_environment_clustering.py``), designed on the
AIMEl molecules.

A fold is a closure around ONE seed environment of a scarce element (N or O):
  * the C and H environments whose atoms co-occur with the seed in >= 95 % of
    the seed atoms' molecules (atom-weighted), the most specific one each, and
  * the environment of the other scarce element that is most absorbed by the
    resulting set of molecules.
Every molecule carrying one of the fold's labels is held out, so the fold's
environments have zero atoms in training.  N and O are seeded SEPARATELY: a C
partner at 0.95 co-occurrence with N alone can drop to ~0.3 once O's atoms are
mixed in, and a combined "closure" would describe neither motif (and leak).
Seeds are taken rarest-molecule-first from the N / O environments carrying
1–15 % of that element's AIMEl atoms, interleaving N and O; a candidate is
rejected if no tight C/H partner exists, if it would hold out more than 25 %
of AIMEl, if it reuses a label of an accepted fold, if a selected label has
< 50 atoms, or if it overlaps accepted folds by more than half.  HDBSCAN noise
atoms (label ``X_-1``) carry no environment and are left out of the DESIGN
only.

The defaults reproduce the five folds used by the environment-holdout study:
    0 N_65 + H_29, C_52, O_10   2,653 molecules (8.6 %)
    1 O_51 + H_21, C_72, N_26   3,497 (11.3 %)
    2 N_27 + H_13, C_34, O_36   2,664 (8.6 %)
    3 N_81 + H_28, C_84, O_16   6,700 (21.7 %)
    4 N_24 + H_12, C_79, O_19   5,065 (16.4 %)

Output: ``qm9_env_folds_6b_rerun.json`` — the design parameters and, per fold, its
labels, seed, co-occurrences, sizes and the QM9 ids of the AIMEl molecules it
holds out (a cross-check; consumers recompute membership from the labels).  The
shipped ``qm9_env_folds_6b.json`` defines the splits and is not overwritten by default.

Usage:
    python qm9_environment_folds.py
    python qm9_environment_folds.py --dry-run
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ELEMENTS = ('H', 'C', 'N', 'O')


def load_atoms(labels_file: str) -> pd.DataFrame:
    """One row per labelled AIMEl atom: mol_pos (row of the label table), qm9_index,
    element, cluster, label.  HDBSCAN noise is dropped (design only)."""
    lab = pd.read_pickle(labels_file)
    lab = lab.assign(mol_pos=np.arange(len(lab)))
    lab = lab[lab['in_aimel']]
    atoms = lab[['mol_pos', 'index', 'atom_env_labels_6b']].explode('atom_env_labels_6b')
    atoms = atoms.rename(columns={'index': 'qm9_index', 'atom_env_labels_6b': 'label'})
    atoms['element'] = atoms['label'].str.split('_').str[0]
    atoms['cluster'] = atoms['label'].str.split('_').str[1].astype(int)
    return atoms[atoms['cluster'] >= 0].reset_index(drop=True)


def build_index(atoms: pd.DataFrame):
    """label -> set(mol_pos), label -> n_atoms, and the molecule universe."""
    mol_of = atoms.groupby('label').mol_pos.apply(set).to_dict()
    n_atoms = atoms.groupby('label').size().to_dict()
    return mol_of, n_atoms, set(atoms.mol_pos.unique())


def cooccurrence(seed, cand_labels, atoms, mol_of) -> pd.Series:
    """Atom-weighted co-occurrence f(cand | seed): the fraction of the SEED's atoms
    whose molecule also contains ``cand``."""
    seed_atom_mols = atoms.loc[atoms.label == seed, 'mol_pos'].to_numpy()
    if len(seed_atom_mols) == 0:
        return pd.Series(dtype=float)
    out = {c: float(np.mean([m in mol_of[c] for m in seed_atom_mols])) for c in cand_labels}
    return pd.Series(out).sort_values(ascending=False)


def closure_set(seed, atoms, mol_of, n_atoms, labels_by_el, cooc_thresh=0.95,
                min_label_atoms=0, seed_el='N'):
    """{N, C, H, O} closure around one seed of element ``seed_el``, or None when no
    C/H partner reaches ``cooc_thresh`` (no well-defined motif)."""
    other_el = 'O' if seed_el == 'N' else 'N'
    sel = {seed_el: seed, f'{seed_el}_cooc': 1.0}
    for el in ('C', 'H'):
        f = cooccurrence(seed, labels_by_el[el], atoms, mol_of)
        ok = [l for l in f.index if f[l] >= cooc_thresh]
        if not ok:
            return None
        sel[el] = min(ok, key=lambda l: len(mol_of[l]))     # most specific partner
        sel[f'{el}_cooc'] = float(f[sel[el]])
    union = set().union(*[mol_of[sel[e]] for e in ELEMENTS if e in sel])
    cands = [l for l in labels_by_el[other_el] if n_atoms[l] >= min_label_atoms] or labels_by_el[other_el]
    absorbed = {l: len(mol_of[l] & union) / len(mol_of[l]) for l in cands}
    sel[other_el] = max(absorbed, key=absorbed.get)
    sel[f'{other_el}_cooc'] = float(absorbed[sel[other_el]])
    chosen = [sel[e] for e in ELEMENTS]
    union = set().union(*[mol_of[l] for l in chosen])
    return {'labels': chosen, 'seed': seed, 'seed_mode': seed_el,
            'cooc': {e: round(sel[f'{e}_cooc'], 3) for e in ELEMENTS},
            'mols': union, 'n_mols': len(union),
            'n_eval_atoms': int(sum(n_atoms[l] for l in chosen)),
            'per_label_atoms': {l: int(n_atoms[l]) for l in chosen}}


def design_folds(atoms, n_folds, cooc_thresh, max_mol_frac, min_label_atoms,
                 seed_frac_lo, seed_frac_hi, o_frac_lo, o_frac_hi, max_overlap=0.5):
    mol_of, n_atoms, all_mols = build_index(atoms)
    labels_by_el = {el: [l for l in mol_of if l.startswith(el + '_')] for el in ELEMENTS}
    budget = int(max_mol_frac * len(all_mols))

    def _band(el, lo, hi):
        tot = sum(v for k, v in n_atoms.items() if k.startswith(el + '_'))
        sel = sorted((l for l in labels_by_el[el] if lo * tot <= n_atoms[l] <= hi * tot),
                     key=lambda l: len(mol_of[l]))          # cheapest holdout first
        print(f"[design] {len(sel):>2} {el} seeds carrying {100*lo:.0f}-{100*hi:.0f}% of the {tot:,} {el} atoms")
        return sel

    pools = {'N': _band('N', seed_frac_lo, seed_frac_hi), 'O': _band('O', o_frac_lo, o_frac_hi)}
    seeds = [(pools[el][i], el) for i in range(max(len(v) for v in pools.values()))
             for el in ('N', 'O') if i < len(pools[el])]
    print(f"[design] per-fold budget {budget:,} molecules ({100*max_mol_frac:.0f}% of {len(all_mols):,}); "
          f"cooc >= {cooc_thresh}")
    folds, used_mols, used_labels = [], set(), set()
    rej = {'no_closure': 0, 'too_big': 0, 'label_reuse': 0, 'overlap': 0, 'thin_label': 0}
    for s_lab, s_el in seeds:
        if len(folds) >= n_folds:
            break
        if s_lab in used_labels:
            continue
        cs = closure_set(s_lab, atoms, mol_of, n_atoms, labels_by_el, cooc_thresh,
                         min_label_atoms=min_label_atoms, seed_el=s_el)
        if cs is None:
            rej['no_closure'] += 1; continue
        if cs['n_mols'] > budget:
            rej['too_big'] += 1; continue
        if any(l in used_labels for l in cs['labels']):
            rej['label_reuse'] += 1; continue
        if min(cs['per_label_atoms'].values()) < min_label_atoms:
            rej['thin_label'] += 1; continue
        overlap = len(cs['mols'] & used_mols) / max(1, cs['n_mols'])
        if overlap > max_overlap:
            rej['overlap'] += 1; continue
        cs['overlap_with_prev'] = round(overlap, 3)
        sub = atoms[atoms.mol_pos.isin(cs['mols'])]
        cs['selected_vs_total'] = {el: {'selected': int(n_atoms[cs['labels'][i]]),
                                        'total_in_holdout': int((sub.element == el).sum())}
                                   for i, el in enumerate(ELEMENTS)}
        folds.append(cs)
        used_mols |= cs['mols']
        used_labels.update(cs['labels'])
    print(f"[design] rejected seeds: {rej}")
    return folds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', default=os.path.join(SCRIPT_DIR, 'qm9_env_labels_6b.pkl.gz'))
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'qm9_env_folds_6b_rerun.json'),
                    help='output folds (default does NOT overwrite the shipped qm9_env_folds_6b.json)')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--cooc-thresh', type=float, default=0.95)
    ap.add_argument('--max-mol-frac', type=float, default=0.25)
    ap.add_argument('--min-label-atoms', type=int, default=50)
    ap.add_argument('--seed-frac-lo', type=float, default=0.01, help='N seed: min share of N atoms')
    ap.add_argument('--seed-frac-hi', type=float, default=0.15, help='N seed: max share of N atoms')
    ap.add_argument('--o-frac-lo', type=float, default=0.01, help='O seed: min share of O atoms')
    ap.add_argument('--o-frac-hi', type=float, default=0.15, help='O seed: max share of O atoms')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    atoms = load_atoms(a.labels)
    n_mol = atoms.mol_pos.nunique()
    print(f"[data] labelled AIMEl atoms {len(atoms):,} in {n_mol:,} molecules")
    folds = design_folds(atoms, a.n_folds, a.cooc_thresh, a.max_mol_frac, a.min_label_atoms,
                         a.seed_frac_lo, a.seed_frac_hi, a.o_frac_lo, a.o_frac_hi)
    for i, f in enumerate(folds):
        cooc = '/'.join(f"{f['cooc'][e]:.2f}" for e in ELEMENTS)
        print(f"{i} [{f['seed_mode']}-seed {f['seed']:>5}] {','.join(f['labels']):<24} "
              f"{f['n_mols']:>6,} mols ({100*f['n_mols']/n_mol:4.1f}%) | eval atoms {f['n_eval_atoms']:>6,} | "
              f"cooc H/C/N/O {cooc}")
    if a.dry_run or not folds:
        return
    mol2qm9 = atoms.drop_duplicates('mol_pos').set_index('mol_pos')['qm9_index']
    payload = {
        'description': 'co-occurrence-closure environment folds over the 6-Bohr labels, designed on AIMEl',
        'radius_bohr': 6.0,
        'design': {k: getattr(a, k) for k in ('n_folds', 'cooc_thresh', 'max_mol_frac', 'min_label_atoms',
                                              'seed_frac_lo', 'seed_frac_hi', 'o_frac_lo', 'o_frac_hi')},
        'folds': [{'fold': i, **{k: v for k, v in f.items() if k != 'mols'},
                   'qm9_index': sorted(int(q) for q in mol2qm9.loc[sorted(f['mols'])])}
                  for i, f in enumerate(folds)],
    }
    with open(a.out, 'w') as fh:
        json.dump(payload, fh, indent=1)
    print(f"-> wrote {a.out}")


if __name__ == '__main__':
    main()
