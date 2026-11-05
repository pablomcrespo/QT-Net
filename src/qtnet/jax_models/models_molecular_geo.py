"""
Geometry-aware molecular property model for the environment-holdout study.

``GeoScalarGNNMolecular`` is the geometry swap of ``ScalarGNNMolecular``
(``models_molecular.py``): the same complex-level readout and layer layout, but
the node / edge encoders and messages are the geometry-gated layers of
``scalar_layers`` (messages gated by ``MLP(chemistry) * tanh(filter(geometry))``
on the displacement ``pos`` of the nodes and the gyration tensor ``G`` of the
edges), read from the cochain ``static`` dicts written by
``representations.row_to_complex``.

Why geometry is given to BOTH variants
--------------------------------------
The per-atom QTAIM properties carry geometric information implicitly (the Mu /
Q components are expressed in the molecular frame).  A topology-only blind
model therefore cannot separate "the informed model wins because of electronic
information" from "the informed model wins because the features smuggle in
geometry".  Here the blind model sees the geometry explicitly, so the only
difference between the two variants is the 10 atomic input columns.

The two variants share ONE embedding architecture (``LadderChemicalEmbedding``:
one-hot(Z) [⊕ atomic features] → Linear) so their parameterisation differs only
by those input columns.  (``scalar_layers.ChemicalEmbedding`` switches between an
embedding lookup and a linear projection depending on ``use_atom_features``.)

Complexes: ``row_to_complex(..., include_atom_features=<variant>, max_dim=1)``;
the model reads nodes and edges only (no bags / rings).
"""

from typing import Dict, Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from qtnet.jax_models.representations import ComplexBatch
from qtnet.jax_models.scalar_layers import (
    ScalarChemicalReminder,
    ScalarEdgeBoundaryMessages,
    ScalarEdgeDownMessages,
    ScalarEdgeEncoder,
    ScalarEdgeFFN,
    ScalarGatedResidual,
    ScalarNodeCoboundaryMessages,
    ScalarNodeEncoder,
    ScalarNodeUpMessages,
)
from qtnet.jax_models.scalar_layers_molecular import (
    ComplexMultiPropertyReadout,
    UpdateFromCochains,
)

# The informed variant's per-atom inputs: exactly the 10 QT-Net targets, in the
# order ScalarGNNMolecular reads them and row_to_complex writes them.
ATOM_FEATURE_KEYS = ('N', 'LI', 'Mu_X', 'Mu_Y', 'Mu_Z',
                     'Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ')

_ModuleList = getattr(nnx, 'List', list)


# =============================================================================
# Chemical embedding shared by both variants
# =============================================================================

class LadderChemicalEmbedding(nnx.Module):
    """one-hot(Z) [⊕ atomic features] → Linear(embedding_dim).

    blind    : one-hot(Z)
    informed : one-hot(Z) ⊕ the ATOM_FEATURE_KEYS columns
    """

    def __init__(
        self,
        num_species: int,
        embedding_dim: int,
        use_atom_features: bool = False,
        num_atom_features: int = len(ATOM_FEATURE_KEYS),
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.num_species = num_species
        self.use_atom_features = use_atom_features
        input_dim = num_species + (num_atom_features if use_atom_features else 0)
        self.projection = nnx.Linear(input_dim, embedding_dim, rngs=rngs)

    def __call__(self, species_indices: jnp.ndarray,
                 atom_features: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        blocks = [jax.nn.one_hot(species_indices, self.num_species)]
        if self.use_atom_features:
            if atom_features is None:
                raise ValueError("atom_features are required when use_atom_features=True")
            blocks.append(atom_features)
        return self.projection(jnp.concatenate(blocks, axis=-1))


# =============================================================================
# Geometry-aware layer (layers > 0)
# =============================================================================

class GeoScalarGNNLayer(nnx.Module):
    """Geometry swap of ``models_molecular.ScalarGNNLayer``."""

    def __init__(self, num_node_scalars, num_edge_scalars, embedding_dim,
                 hidden_dim=64, geometric_filter_dim=None, rngs=None):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.node_up        = ScalarNodeUpMessages(num_node_scalars, hidden_dim, num_edge_scalars, geometric_filter_dim, rngs)
        self.node_up_block  = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs)
        self.edge_bd        = ScalarEdgeBoundaryMessages(num_node_scalars, num_edge_scalars, hidden_dim, geometric_filter_dim, rngs)
        self.edge_bd_block  = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)
        self.edge_dn        = ScalarEdgeDownMessages(num_edge_scalars, num_node_scalars, hidden_dim, geometric_filter_dim, rngs)
        self.edge_dn_block  = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)
        self.edge_ffn       = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs)
        self.edge_ffn_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)
        self.node_cb        = ScalarNodeCoboundaryMessages(num_node_scalars, num_edge_scalars, hidden_dim, rngs)
        self.node_cb_block  = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs)
        self.chem_reminder  = ScalarChemicalReminder(num_node_scalars, embedding_dim, hidden_dim, rngs)

    def __call__(self, node_features, edge_features, chem_embeddings, node_batch, edge_batch):
        x_mask, ex_mask = node_batch.x_mask, edge_batch.x_mask

        normed = self.node_up_block.pre_norm(node_features)
        up = self.node_up(normed, x_mask, node_batch.up_senders, node_batch.up_receivers,
                          node_batch.up_mask, static=node_batch.static,
                          edge_features=edge_features, up_intermediaries=node_batch.up_intermediaries)
        node_features = self.node_up_block.residual(node_features, up, x_mask)

        normed_e = self.edge_bd_block.pre_norm(edge_features)
        bd = self.edge_bd(normed_e, ex_mask, node_features,
                          edge_batch.boundary_senders, edge_batch.boundary_receivers,
                          edge_batch.boundary_mask, static=edge_batch.static)
        edge_features = self.edge_bd_block.residual(edge_features, bd, ex_mask)

        normed_e = self.edge_dn_block.pre_norm(edge_features)
        dn = self.edge_dn(normed_e, ex_mask, node_features,
                          edge_batch.down_senders, edge_batch.down_receivers,
                          edge_batch.down_intermediaries, edge_batch.down_mask, static=edge_batch.static)
        edge_features = self.edge_dn_block.residual(edge_features, dn, ex_mask)

        normed_e = self.edge_ffn_block.pre_norm(edge_features)
        ffn = self.edge_ffn(normed_e, ex_mask)
        edge_features = self.edge_ffn_block.residual(edge_features, ffn, ex_mask)

        normed = self.node_cb_block.pre_norm(node_features)
        cb = self.node_cb(normed, x_mask, edge_features,
                          node_batch.coboundary_senders, node_batch.coboundary_receivers,
                          node_batch.coboundary_mask)
        node_features = self.node_cb_block.residual(node_features, cb, x_mask)

        node_features = self.chem_reminder(node_features, chem_embeddings, x_mask)
        return node_features, edge_features


# =============================================================================
# Model
# =============================================================================

class GeoScalarGNNMolecular(nnx.Module):
    """
    Geometry-aware scalar GNN (nodes + edges, no rings) for molecular property
    prediction.  Same architecture as ``ScalarGNNMolecular`` with the geometry
    layers swapped in; ``use_atom_features`` selects the blind / informed
    variant.

    Args:
        num_species          : Vocabulary size for the element one-hot.
        num_outputs          : Number of molecular properties (default 4).
        num_node_scalars     : Node feature width.
        num_edge_scalars     : Edge feature width.
        num_complex_scalars  : Graph-level feature width (default = num_node_scalars).
        embedding_dim        : Chemical embedding width.
        hidden_dim           : MLP hidden width.
        num_layers           : Total layers including layer 0.
        geometric_filter_dim : Width of the geometry filter (None → layer default).
        use_atom_features    : Read the ATOM_FEATURE_KEYS columns (informed variant).
        rngs                 : Flax RNG container.
    """

    def __init__(
        self,
        num_species: int,
        num_outputs: int = 4,
        num_node_scalars: int = 64,
        num_edge_scalars: int = 64,
        num_complex_scalars: int = 64,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 3,
        geometric_filter_dim: Optional[int] = None,
        use_atom_features: bool = False,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        if num_complex_scalars is None:
            num_complex_scalars = num_node_scalars
        self.num_layers = num_layers
        self.use_atom_features = use_atom_features

        self.chemical_embedding = LadderChemicalEmbedding(
            num_species, embedding_dim, use_atom_features=use_atom_features, rngs=rngs)

        self.cx_encoder      = nnx.Linear(embedding_dim, num_complex_scalars, rngs=rngs)
        self.cx_encoder_norm = nnx.LayerNorm(num_features=num_complex_scalars, rngs=rngs)

        # Layer 0: geometry encoders
        self.node_encoder      = ScalarNodeEncoder(num_node_scalars, embedding_dim, hidden_dim, geometric_filter_dim, rngs)
        self.node_encoder_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        self.edge_encoder      = ScalarEdgeEncoder(num_node_scalars, num_edge_scalars, hidden_dim, geometric_filter_dim, rngs)
        self.edge_encoder_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)

        # Layer 0: remaining sub-layers
        self.l0_edge_dn        = ScalarEdgeDownMessages(num_edge_scalars, num_node_scalars, hidden_dim, geometric_filter_dim, rngs)
        self.l0_edge_dn_block  = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)
        self.l0_edge_ffn       = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs)
        self.l0_edge_ffn_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)
        self.l0_node_cb        = ScalarNodeCoboundaryMessages(num_node_scalars, num_edge_scalars, hidden_dim, rngs)
        self.l0_node_cb_block  = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs)
        self.l0_chem_reminder  = ScalarChemicalReminder(num_node_scalars, embedding_dim, hidden_dim, rngs)

        # flax >= 0.12 rejects a plain list of parameter-holding modules as a
        # static attribute; older flax has no nnx.List.
        self.layers = _ModuleList([
            GeoScalarGNNLayer(num_node_scalars, num_edge_scalars, embedding_dim,
                              hidden_dim, geometric_filter_dim, rngs)
            for _ in range(num_layers - 1)
        ])
        self.cx_updates = _ModuleList([
            UpdateFromCochains(num_complex_scalars, num_node_scalars, num_edge_scalars,
                               d_ring=None, hidden_dim=hidden_dim, rngs=rngs)
            for _ in range(num_layers)
        ])
        self.readout = ComplexMultiPropertyReadout(
            d_cx=num_complex_scalars, num_outputs=num_outputs,
            hidden_dim=hidden_dim, num_layers=num_layers, rngs=rngs)

    def _embed(self, node_batch):
        node_static = node_batch.static
        atom_features = None
        if self.use_atom_features:
            atom_features = jnp.stack([node_static[k] for k in ATOM_FEATURE_KEYS], axis=-1)
        return self.chemical_embedding(node_static['Z'], atom_features)

    def __call__(self, complex_batch: ComplexBatch,
                 graph_idx: Optional[jnp.ndarray] = None, num_graphs: int = 1) -> Dict[str, jnp.ndarray]:
        node_batch = complex_batch.cochain_batches[0]
        edge_batch = complex_batch.cochain_batches[1]

        chem_embeddings = self._embed(node_batch)

        if complex_batch.x is not None:
            cx = self.cx_encoder_norm(self.cx_encoder(complex_batch.x))
        else:
            h_init = jnp.where(node_batch.x_mask[:, None], chem_embeddings, 0.0)
            cx_pool = (jax.ops.segment_sum(h_init, graph_idx, num_graphs + 1)
                       if graph_idx is not None else h_init.sum(axis=0, keepdims=True))
            cx = self.cx_encoder_norm(self.cx_encoder(cx_pool))

        edge_graph_idx = edge_batch.owner_cochains

        node_features = self.node_encoder(
            chem_embeddings, node_batch.x_mask,
            node_batch.up_senders, node_batch.up_receivers, node_batch.up_mask,
            static=node_batch.static)
        node_features = self.node_encoder_norm(node_features)
        node_features = jnp.where(node_batch.x_mask[:, None], node_features, 0.0)

        edge_features = self.edge_encoder(
            node_features, edge_batch.boundary_senders, edge_batch.boundary_receivers,
            edge_batch.boundary_mask, edge_batch.x_mask, static=edge_batch.static)
        edge_features = self.edge_encoder_norm(edge_features)
        edge_features = jnp.where(edge_batch.x_mask[:, None], edge_features, 0.0)

        normed_e = self.l0_edge_dn_block.pre_norm(edge_features)
        dn = self.l0_edge_dn(normed_e, edge_batch.x_mask, node_features,
                             edge_batch.down_senders, edge_batch.down_receivers,
                             edge_batch.down_intermediaries, edge_batch.down_mask, static=edge_batch.static)
        edge_features = self.l0_edge_dn_block.residual(edge_features, dn, edge_batch.x_mask)

        normed_e = self.l0_edge_ffn_block.pre_norm(edge_features)
        ffn = self.l0_edge_ffn(normed_e, edge_batch.x_mask)
        edge_features = self.l0_edge_ffn_block.residual(edge_features, ffn, edge_batch.x_mask)

        normed = self.l0_node_cb_block.pre_norm(node_features)
        cb = self.l0_node_cb(normed, node_batch.x_mask, edge_features,
                             node_batch.coboundary_senders, node_batch.coboundary_receivers,
                             node_batch.coboundary_mask)
        node_features = self.l0_node_cb_block.residual(node_features, cb, node_batch.x_mask)
        node_features = self.l0_chem_reminder(node_features, chem_embeddings, node_batch.x_mask)

        cx = self.cx_updates[0](cx, node_features, edge_features,
                                node_graph_idx=graph_idx, edge_graph_idx=edge_graph_idx,
                                node_mask=node_batch.x_mask, edge_mask=edge_batch.x_mask,
                                num_graphs=num_graphs)
        complex_features = [cx]

        for i, layer in enumerate(self.layers):
            node_features, edge_features = layer(node_features, edge_features,
                                                 chem_embeddings, node_batch, edge_batch)
            cx = self.cx_updates[i + 1](cx, node_features, edge_features,
                                        node_graph_idx=graph_idx, edge_graph_idx=edge_graph_idx,
                                        node_mask=node_batch.x_mask, edge_mask=edge_batch.x_mask,
                                        num_graphs=num_graphs)
            complex_features.append(cx)

        predictions = self.readout.read_and_combine(complex_features)
        return {
            'node_features':    node_features,
            'edge_features':    edge_features,
            'complex_features': cx[:num_graphs],
            'predictions':      predictions[:num_graphs],
            'x_mask':           node_batch.x_mask,
        }
