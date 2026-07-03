# Learnable activations (soft dynActivation) — atomic QTAIM ablation

Quick check of whether the **soft dynActivation** family from
*"dynActivation: A Trainable Activation Family for Adaptive Nonlinearity"*
(arXiv:2603.22154) helps on QT‑Net's atomic targets, for the equivariant GNN
(EGNN) and the scalar GNN (SGNN), both with **12 neighbours**.

## What was implemented

`src/qtnet/jax_models/dynamic_activations.py` adds the **soft (smooth)** variant.
The paper derives the family from a smooth sigmoid gate,
`σ(x)·x·(α−β) + β·x`; since `σ(x)·x = SiLU/Swish`, the smooth member is

```
dynActivation_soft(x) = SiLU(x) · (α − β) + β · x
```

with two learnable per‑site scalars `α, β`, initialised `α=1, β=0` so it
reproduces QT‑Net's existing static SiLU exactly (an exact no‑op at init). SiLU
is C‑∞, so the unit is smooth for every `α, β` (a ReLU‑like base would leave a
kink). Every SiLU site in `scalar_layers.py` and `equivariant_layers.py` is
routed through it (63 sites in SGNN → +126 params; 42 in EGNN → +84 params —
negligible).

Toggle at construction time via `set_dynamic_activations(True/False)`; off by
default, so unchanged models are byte‑for‑byte the old SiLU networks.

## Setup (small, CPU‑only)

This environment has **no GPU**, so the experiment is deliberately small — a
directional signal, not paper‑grade numbers.

- fold‑0 scaffold split, **350 train / 120 val** molecules (single full batch)
- SGNN: 120 epochs, seeds 0/1/2 · EGNN: 100 epochs, seeds 0/1
- identical setup per pair; the *only* difference is static SiLU vs soft dynAct
- metric: multitask validation loss (`total`), best over epochs (early stopping)

Reproduce: `bash run_experiments.sh` then
`python scripts/atomic/plot_learnable_activations.py`.

## Result — dynActivation helps both models

| model | nbrs | static SiLU (best val) | soft dynAct (best val) | rel. improvement |
|---|---|---|---|---|
| SGNN | 12 | 0.1882 ± 0.0078 | **0.1759 ± 0.0114** | **+6.6 %** |
| EGNN | 12 | 0.4255 ± 0.0371 | **0.3701 ± 0.0282** | **+13.0 %** |

Per‑seed best validation loss:

| model | seed | SiLU | dynAct |
|---|---|---|---|
| SGNN | 0 | 0.1871 | 0.1894 |
| SGNN | 1 | 0.1982 | 0.1765 |
| SGNN | 2 | 0.1792 | 0.1616 |
| EGNN | 0 | 0.4626 | 0.3983 |
| EGNN | 1 | 0.3884 | 0.3419 |

- **EGNN**: both seeds improve (+13 % mean). The larger gain fits the paper's
  story that trainable activations help more in the more expressive / harder‑to‑
  optimise model.
- **SGNN**: +6.6 % mean; 2 of 3 seeds clearly better, seed 0 marginally worse
  (within noise).
- **Training dynamics** (`learnable_activations.png`): SGNN dynAct reaches its
  best at the *final* epoch with no late overfitting (`final == best`), whereas
  static SiLU peaks earlier and then degrades. EGNN's SiLU shows sharp
  instability spikes (epochs ~58, ~90) while dynAct trains more smoothly — the
  learnable linear path behaves like a light regulariser.

## Caveats

- Small data (350 molecules, fold 0) and few seeds (2–3) → treat effect sizes as
  indicative; SGNN std bands overlap at best‑val.
- Hyperparameters were **not** re‑tuned for dynActivation (same lr/wd as the SiLU
  baseline); a short HPO could widen the gap.
- Full verdict needs the standard 5×5 CV at scale on GPU.

**Bottom line:** on this task, making the activations learnable (soft
dynActivation) is a small, cheap change that consistently helped the
equivariant GNN (+13 %) and helped the scalar GNN on most seeds (+6.6 %) — worth
a proper GPU run over the full CV.
