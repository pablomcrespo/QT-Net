"""Learnable (trainable) activation functions for QT-Net.

Implements the ``dynActivation`` family from

    "dynActivation: A Trainable Activation Family for Adaptive Nonlinearity"
    (arXiv:2603.22154)

The idea is to keep the inductive bias of an established base activation while
giving every layer two extra learnable scalars that let it interpolate between
the base nonlinearity and a purely linear path:

    dynActivation(x) = BaseAct(x) * (alpha - beta) + beta * x

with per-layer scalars ``alpha`` and ``beta``.  Initialising ``alpha = 1`` and
``beta = 0`` recovers ``BaseAct`` exactly, so a dynActivation network starts
identical to its static-activation counterpart and can only deviate if the
data pushes it to.  Setting ``alpha = beta = 1`` collapses the unit to the
identity ``x``; the paper reports that deeper layers tend to linearise this way
during training.  Only two scalars are added per activation site, so the
parameter overhead is negligible.

Soft (smooth) variant
---------------------
The paper *derives* the family from a smooth sigmoid gate,

    sigma(x) = 1 / (1 + e^-x)
    dynActivation(x) = sigma(x) * x * (alpha - beta) + beta * x ,

and only afterwards generalises ``sigma(x) * x`` to an arbitrary ``BaseAct``.
Because ``sigma(x) * x`` is exactly **SiLU/Swish**, the smooth ("soft") member
of the family is the SiLU-based one:

    dynActivation_soft(x) = SiLU(x) * (alpha - beta) + beta * x .

SiLU is C-infinity, so this activation is smooth for *every* value of ``alpha``
and ``beta`` -- unlike ReLU-like bases, which leave a kink at the origin.  We
use this soft variant throughout: it also happens to match QT-Net's existing
static activation (SiLU), which makes the ``alpha=1, beta=0`` initialisation an
exact no-op and keeps the comparison to the baseline clean.

Wiring
------
Every MLP in ``scalar_layers`` and ``equivariant_layers`` builds its
nonlinearity through :func:`activation` rather than referencing ``nnx.silu``
directly.  A process-wide switch selects the behaviour at construction time:

    >>> from qtnet.jax_models import dynamic_activations as da
    >>> da.set_dynamic_activations(True)   # subsequently-built models learn acts
    >>> model = ScalarGNN(...)
    >>> da.set_dynamic_activations(False)  # back to plain SiLU

The switch is only read when a model is *constructed*; toggling it afterwards
does not change already-built modules.  This keeps the change unobtrusive: with
the switch off (the default) the models are byte-for-byte the SiLU networks
they were before.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import flax.nnx as nnx


# ---------------------------------------------------------------------------
# Process-wide switch
# ---------------------------------------------------------------------------
_USE_DYNAMIC_ACTIVATIONS: bool = False


def set_dynamic_activations(enabled: bool) -> None:
    """Enable/disable dynActivation for *subsequently constructed* models."""
    global _USE_DYNAMIC_ACTIVATIONS
    _USE_DYNAMIC_ACTIVATIONS = bool(enabled)


def dynamic_activations_enabled() -> bool:
    """Return whether newly built models will use dynActivation."""
    return _USE_DYNAMIC_ACTIVATIONS


# ---------------------------------------------------------------------------
# dynActivation module
# ---------------------------------------------------------------------------
class DynActivation(nnx.Module):
    """Trainable activation ``BaseAct(x) * (alpha - beta) + beta * x``.

    Defaults to the paper's smooth ("soft") variant with ``base = SiLU``
    (SiLU(x) = x * sigmoid(x)), which is C-infinity for all ``alpha, beta``.

    Args:
        base: Base activation callable (defaults to SiLU, the smooth variant).
        alpha_init: Initial value of the ``alpha`` scalar (1.0 recovers base).
        beta_init: Initial value of the ``beta`` scalar (0.0 recovers base).
    """

    def __init__(
        self,
        base: Callable[[jnp.ndarray], jnp.ndarray] = nnx.silu,
        alpha_init: float = 1.0,
        beta_init: float = 0.0,
    ):
        self.base = base
        # Two learnable scalars per activation site.
        self.alpha = nnx.Param(jnp.asarray(alpha_init, dtype=jnp.float32))
        self.beta = nnx.Param(jnp.asarray(beta_init, dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.base(x) * (self.alpha.value - self.beta.value) + self.beta.value * x


# ---------------------------------------------------------------------------
# Factory used by the layer modules
# ---------------------------------------------------------------------------
def activation() -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return the activation to place inside an ``nnx.Sequential``.

    ``DynActivation`` (a fresh instance, so its scalars are learned per site)
    when the switch is on, otherwise the plain ``nnx.silu`` function.
    """
    if _USE_DYNAMIC_ACTIVATIONS:
        return DynActivation()
    return nnx.silu
