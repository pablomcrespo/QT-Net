"""
Molecular property prediction models (topology-only, no 3D geometry).

Two model families:
  ScalarGNNMolecular    – BCP bond topology only (no rings)
  ScalarTPaiNNMolecular – BCP bond topology + RCP ring topology

Both maintain a graph-level (complex) feature vector ``cx`` that is
initialised from pooled chemical embeddings and updated at every layer by
``UpdateFromCochains``.  The per-layer history of ``cx`` feeds directly into
``ComplexMultiPropertyReadout`` for the final multi-property prediction,
making every layer contribute to the prediction and the readout robust to
depth choice.

If ``complex_batch.x`` is not None (e.g. SMILES-derived fingerprints), it
is used instead of the pooled chemical embeddings to seed ``cx``.

Output dict keys: ``node_features``, ``edge_features``, ``complex_features``
(last layer), ``predictions`` (G, num_outputs), ``x_mask``.
"""

from typing import Optional, Dict

import jax.numpy as jnp
import jax
import flax.nnx as nnx

from qtnet.jax_models.scalar_layers import (
    ChemicalEmbedding,
    ScalarGatedResidual,
    ScalarChemicalReminder,
    ScalarEdgeFFN,
    ScalarNodeCoboundaryMessages,
    ScalarBagEncoder,
    ScalarEdgeCoboundaryMessages,
    ScalarDualChannelNodeUpdate,
    ScalarNodeBoundaryMessages,
)
from qtnet.jax_models.scalar_layers_molecular import (
    UpdateFromCochains,
    ComplexMultiPropertyReadout,
    NoGeoNodeEncoder,
    NoGeoEdgeEncoder,
    NoGeoNodeMessages,
    NoGeoEdgeBoundaryMessages,
    NoGeoEdgeDownMessages,
)

from qtnet.jax_models.representations import ComplexBatch


# =============================================================================
# NO-GEOMETRY GNN LAYER  (topology only, dim-0 + dim-1, no rings in messages)
# =============================================================================

class ScalarGNNLayer(nnx.Module):
    """Single topology-only GNN layer (layers > 0).

    Message-passing cycle:
        NoGeoNodeMessages      → ScalarGatedResidual
        NoGeoEdgeBoundaryMessages → ScalarGatedResidual
        NoGeoEdgeDownMessages  → ScalarGatedResidual
        ScalarEdgeFFN          → ScalarGatedResidual
        ScalarNodeCoboundaryMessages → ScalarGatedResidual
        ScalarChemicalReminder
    """

    def __init__(
        self,
        num_node_scalars: int,
        num_edge_scalars: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.node_up        = NoGeoNodeMessages(num_node_scalars, hidden_dim, num_edge_scalars, rngs)
        self.node_up_block  = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs)

        self.edge_bd        = NoGeoEdgeBoundaryMessages(num_node_scalars, num_edge_scalars, hidden_dim, rngs)
        self.edge_bd_block  = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)

        self.edge_dn        = NoGeoEdgeDownMessages(num_edge_scalars, num_node_scalars, hidden_dim, rngs)
        self.edge_dn_block  = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)

        self.edge_ffn       = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs)
        self.edge_ffn_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)

        self.node_cb        = ScalarNodeCoboundaryMessages(num_node_scalars, num_edge_scalars, hidden_dim, rngs)
        self.node_cb_block  = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs)

        self.chem_reminder  = ScalarChemicalReminder(num_node_scalars, embedding_dim, hidden_dim, rngs)

    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_features: jnp.ndarray,
        chem_embeddings: jnp.ndarray,
        node_batch,
        edge_batch,
    ) -> tuple:
        x_mask  = node_batch.x_mask
        ex_mask = edge_batch.x_mask

        normed = self.node_up_block.pre_norm(node_features)
        up = self.node_up(
            normed, x_mask,
            node_batch.up_senders, node_batch.up_receivers, node_batch.up_mask,
            edge_features=edge_features,
            up_intermediaries=node_batch.up_intermediaries,
        )
        node_features = self.node_up_block.residual(node_features, up, x_mask)

        normed_e = self.edge_bd_block.pre_norm(edge_features)
        bd = self.edge_bd(
            normed_e, ex_mask,
            node_features, edge_batch.boundary_senders, edge_batch.boundary_receivers, edge_batch.boundary_mask,
        )
        edge_features = self.edge_bd_block.residual(edge_features, bd, ex_mask)

        normed_e = self.edge_dn_block.pre_norm(edge_features)
        dn = self.edge_dn(
            normed_e, ex_mask, node_features,
            edge_batch.down_senders, edge_batch.down_receivers,
            edge_batch.down_intermediaries, edge_batch.down_mask,
        )
        edge_features = self.edge_dn_block.residual(edge_features, dn, ex_mask)

        normed_e = self.edge_ffn_block.pre_norm(edge_features)
        ffn = self.edge_ffn(normed_e, ex_mask)
        edge_features = self.edge_ffn_block.residual(edge_features, ffn, ex_mask)

        normed = self.node_cb_block.pre_norm(node_features)
        cb = self.node_cb(
            normed, x_mask, edge_features,
            node_batch.coboundary_senders, node_batch.coboundary_receivers, node_batch.coboundary_mask,
        )
        node_features = self.node_cb_block.residual(node_features, cb, x_mask)

        node_features = self.chem_reminder(node_features, chem_embeddings, x_mask)

        return node_features, edge_features


# =============================================================================
# ScalarGNNMolecular  (no geometry, no rings, 4 outputs)
# =============================================================================

class ScalarGNNMolecular(nnx.Module):
    """
    Topology-only scalar GNN for molecular property prediction.

    Maintains a graph-level feature vector ``cx`` (shape ``(G, num_complex_scalars)``)
    that is initialised from pooled chemical embeddings and updated after every
    layer by ``UpdateFromCochains``.  The per-layer history of ``cx`` feeds
    ``ComplexMultiPropertyReadout``, so every layer directly contributes to the
    final prediction.

    Architecture
    ------------
    Init  : cx ← cx_encoder( segment_sum(chem_embeddings, graph_idx) )
    Layer 0:
        ChemicalEmbedding
        NoGeoNodeEncoder → LayerNorm
        NoGeoEdgeEncoder → LayerNorm
        NoGeoEdgeDownMessages → GatedResidual
        ScalarEdgeFFN → GatedResidual
        ScalarNodeCoboundaryMessages → GatedResidual
        ScalarChemicalReminder
        cx ← UpdateFromCochains(cx, node_features, edge_features)

    Layers 1+: ScalarGNNLayer → UpdateFromCochains

    Readout: ComplexMultiPropertyReadout(cx per layer) → (G, num_outputs)

    Args:
        num_species         : Vocabulary size for element embedding.
        num_outputs         : Number of molecular properties (default 4).
        num_node_scalars    : Node feature width.
        num_edge_scalars    : Edge feature width.
        num_complex_scalars : Graph-level feature width (default = num_node_scalars).
        embedding_dim       : Chemical embedding width.
        hidden_dim          : MLP hidden width.
        num_layers          : Total layers including layer 0.
        use_atom_features   : Whether to include pre-computed atomic properties.
        rngs                : Flax RNG container.
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

        self.chemical_embedding = ChemicalEmbedding(
            num_species=num_species,
            embedding_dim=embedding_dim,
            use_atom_features=use_atom_features,
            rngs=rngs,
        )

        # Complex feature initialisation: pool chem_embeddings → cx
        self.cx_encoder      = nnx.Linear(embedding_dim, num_complex_scalars, rngs=rngs)
        self.cx_encoder_norm = nnx.LayerNorm(num_features=num_complex_scalars, rngs=rngs)

        # Layer 0: encoders
        self.node_encoder      = NoGeoNodeEncoder(num_node_scalars, embedding_dim, hidden_dim, rngs)
        self.node_encoder_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)

        self.edge_encoder      = NoGeoEdgeEncoder(num_node_scalars, num_edge_scalars, hidden_dim, rngs)
        self.edge_encoder_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)

        # Layer 0: remaining sub-layers
        self.l0_edge_dn       = NoGeoEdgeDownMessages(num_edge_scalars, num_node_scalars, hidden_dim, rngs)
        self.l0_edge_dn_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)
        self.l0_edge_ffn      = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs)
        self.l0_edge_ffn_block= ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)
        self.l0_node_cb       = ScalarNodeCoboundaryMessages(num_node_scalars, num_edge_scalars, hidden_dim, rngs)
        self.l0_node_cb_block = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs)
        self.l0_chem_reminder = ScalarChemicalReminder(num_node_scalars, embedding_dim, hidden_dim, rngs)

        # Layers 1+
        self.layers = [
            ScalarGNNLayer(num_node_scalars, num_edge_scalars, embedding_dim, hidden_dim, rngs)
            for _ in range(num_layers - 1)
        ]

        # One UpdateFromCochains per layer (layer 0 + subsequent layers)
        self.cx_updates = [
            UpdateFromCochains(num_complex_scalars, num_node_scalars, num_edge_scalars,
                               d_ring=None, hidden_dim=hidden_dim, rngs=rngs)
            for _ in range(num_layers)
        ]

        self.readout = ComplexMultiPropertyReadout(
            d_cx=num_complex_scalars,
            num_outputs=num_outputs,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rngs=rngs,
        )

    def __call__(
        self,
        complex_batch: ComplexBatch,
        graph_idx: Optional[jnp.ndarray] = None,
        num_graphs: int = 1,
    ) -> Dict[str, jnp.ndarray]:
        node_batch = complex_batch.cochain_batches[0]
        edge_batch = complex_batch.cochain_batches[1]

        node_static = node_batch.static
        assert node_static is not None
        assert node_batch.x_mask is not None
        species_indices = node_static['Z']
        atom_features  = None
        if self.use_atom_features:
            keys = ['N', 'LI', 'Mu_X', 'Mu_Y', 'Mu_Z', 'Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ']
            atom_features = jnp.stack([node_static[k] for k in keys], axis=-1)

        chem_embeddings = self.chemical_embedding(species_indices, atom_features)

        # Initialise complex (graph-level) features.
        # cx has shape (G+1, d_cx): includes one OOB "padding complex" row at
        # index G, mirroring the mc+1 structure of cochain arrays.  Padding
        # cells own complex G so their contributions route to the OOB row.
        if complex_batch.x is not None:
            cx = self.cx_encoder_norm(self.cx_encoder(complex_batch.x))
        else:
            h_init = jnp.where(node_batch.x_mask[:, None], chem_embeddings, 0.0)
            cx_pool = (jax.ops.segment_sum(h_init, graph_idx, num_graphs + 1)
                       if graph_idx is not None else h_init.sum(axis=0, keepdims=True))
            cx = self.cx_encoder_norm(self.cx_encoder(cx_pool))  # (G+1, d_cx)

        assert edge_batch.owner_cochains is not None
        edge_graph_idx = edge_batch.owner_cochains    # (E_total,) edge → graph

        # Layer 0: node encoder
        node_features = self.node_encoder(
            chem_embeddings, node_batch.x_mask,
            node_batch.up_senders, node_batch.up_receivers, node_batch.up_mask,
        )
        node_features = self.node_encoder_norm(node_features)
        if node_batch.x_mask is not None:
            node_features = jnp.where(node_batch.x_mask[:, None], node_features, 0.0)

        # Layer 0: edge encoder
        edge_features = self.edge_encoder(
            node_features,
            edge_batch.boundary_senders, edge_batch.boundary_receivers, edge_batch.boundary_mask,
            edge_batch.x_mask,
        )
        edge_features = self.edge_encoder_norm(edge_features)
        if edge_batch.x_mask is not None:
            edge_features = jnp.where(edge_batch.x_mask[:, None], edge_features, 0.0)

        # Layer 0: edge down, FFN, node coboundary, chemical reminder
        normed_e = self.l0_edge_dn_block.pre_norm(edge_features)
        dn = self.l0_edge_dn(
            normed_e, edge_batch.x_mask, node_features,
            edge_batch.down_senders, edge_batch.down_receivers,
            edge_batch.down_intermediaries, edge_batch.down_mask,
        )
        edge_features = self.l0_edge_dn_block.residual(edge_features, dn, edge_batch.x_mask)

        normed_e = self.l0_edge_ffn_block.pre_norm(edge_features)
        ffn = self.l0_edge_ffn(normed_e, edge_batch.x_mask)
        edge_features = self.l0_edge_ffn_block.residual(edge_features, ffn, edge_batch.x_mask)

        normed = self.l0_node_cb_block.pre_norm(node_features)
        cb = self.l0_node_cb(
            normed, node_batch.x_mask, edge_features,
            node_batch.coboundary_senders, node_batch.coboundary_receivers, node_batch.coboundary_mask,
        )
        node_features = self.l0_node_cb_block.residual(node_features, cb, node_batch.x_mask)
        node_features = self.l0_chem_reminder(node_features, chem_embeddings, node_batch.x_mask)

        # Update complex features after layer 0
        cx = self.cx_updates[0](
            cx, node_features, edge_features,
            node_graph_idx=graph_idx, edge_graph_idx=edge_graph_idx,
            node_mask=node_batch.x_mask, edge_mask=edge_batch.x_mask,
            num_graphs=num_graphs,
        )
        complex_features = [cx]

        # Layers 1+
        for i, layer in enumerate(self.layers):
            node_features, edge_features = layer(
                node_features, edge_features, chem_embeddings, node_batch, edge_batch
            )
            cx = self.cx_updates[i + 1](
                cx, node_features, edge_features,
                node_graph_idx=graph_idx, edge_graph_idx=edge_graph_idx,
                node_mask=node_batch.x_mask, edge_mask=edge_batch.x_mask,
                num_graphs=num_graphs,
            )
            complex_features.append(cx)

        predictions = self.readout.read_and_combine(complex_features)

        return {
            'node_features':    node_features,
            'edge_features':    edge_features,
            'complex_features': cx[:num_graphs],            # (G, d_cx) — last layer, real graphs only
            'predictions':      predictions[:num_graphs],   # (G, num_outputs)
            'x_mask':           node_batch.x_mask,
        }


# =============================================================================
# ScalarTPaiNNMolecular  (no geometry, with rings, 4 outputs)
# =============================================================================

class ScalarTPaiNNLayer(nnx.Module):
    """Single topology-only TPaiNN layer (layers > 0).

    Extends ScalarGNNLayer with ring coboundary path:
        … edge FFN …
        ScalarBagEncoder (ring encoder)     → ring features
        ScalarEdgeCoboundaryMessages        → edge coboundary update
        ScalarEdgeFFN → GatedResidual
        ScalarDualChannelNodeUpdate         (edge-coboundary + ring-boundary channels)
    """

    def __init__(
        self,
        num_node_scalars: int,
        num_edge_scalars: int,
        num_ring_scalars: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.node_up        = NoGeoNodeMessages(num_node_scalars, hidden_dim, num_edge_scalars, rngs)
        self.node_up_block  = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs)

        self.edge_bd        = NoGeoEdgeBoundaryMessages(num_node_scalars, num_edge_scalars, hidden_dim, rngs)
        self.edge_bd_block  = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)

        self.edge_dn        = NoGeoEdgeDownMessages(num_edge_scalars, num_node_scalars, hidden_dim, rngs)
        self.edge_dn_block  = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)

        # Ring encoder + edge coboundary (ring → edge)
        self.ring_encoder   = ScalarBagEncoder(num_edge_scalars, num_ring_scalars, hidden_dim, rngs)
        self.edge_cb        = ScalarEdgeCoboundaryMessages(num_edge_scalars, num_ring_scalars, hidden_dim, rngs)
        self.edge_cb_block  = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)

        self.edge_ffn       = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs)
        self.edge_ffn_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)

        # Dual-channel node update (edge-coboundary + ring-boundary channels)
        self.node_cb        = ScalarNodeCoboundaryMessages(num_node_scalars, num_edge_scalars, hidden_dim, rngs)
        self.node_bd        = ScalarNodeBoundaryMessages(num_node_scalars, num_ring_scalars, hidden_dim, rngs)
        self.node_update    = ScalarDualChannelNodeUpdate(num_node_scalars, hidden_dim, rngs)

        self.chem_reminder  = ScalarChemicalReminder(num_node_scalars, embedding_dim, hidden_dim, rngs)

    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_features: jnp.ndarray,
        chem_embeddings: jnp.ndarray,
        node_batch,
        edge_batch,
        ring_batch,
    ) -> tuple:
        x_mask  = node_batch.x_mask
        ex_mask = edge_batch.x_mask

        normed = self.node_up_block.pre_norm(node_features)
        up = self.node_up(
            normed, x_mask,
            node_batch.up_senders, node_batch.up_receivers, node_batch.up_mask,
            edge_features=edge_features, up_intermediaries=node_batch.up_intermediaries,
        )
        node_features = self.node_up_block.residual(node_features, up, x_mask)

        normed_e = self.edge_bd_block.pre_norm(edge_features)
        bd = self.edge_bd(
            normed_e, ex_mask, node_features,
            edge_batch.boundary_senders, edge_batch.boundary_receivers, edge_batch.boundary_mask,
        )
        edge_features = self.edge_bd_block.residual(edge_features, bd, ex_mask)

        normed_e = self.edge_dn_block.pre_norm(edge_features)
        dn = self.edge_dn(
            normed_e, ex_mask, node_features,
            edge_batch.down_senders, edge_batch.down_receivers,
            edge_batch.down_intermediaries, edge_batch.down_mask,
        )
        edge_features = self.edge_dn_block.residual(edge_features, dn, ex_mask)

        ring_features = self.ring_encoder(
            edge_features,
            ring_batch.boundary_senders, ring_batch.boundary_receivers,
            ring_batch.boundary_mask, ring_batch.x_mask,
        )

        normed_e = self.edge_cb_block.pre_norm(edge_features)
        ecb = self.edge_cb(
            normed_e, ex_mask, ring_features,
            edge_batch.coboundary_senders, edge_batch.coboundary_receivers,
            edge_batch.coboundary_mask,
        )
        edge_features = self.edge_cb_block.residual(edge_features, ecb, ex_mask)

        normed_e = self.edge_ffn_block.pre_norm(edge_features)
        ffn = self.edge_ffn(normed_e, ex_mask)
        edge_features = self.edge_ffn_block.residual(edge_features, ffn, ex_mask)

        normed = self.node_update.pre_norm(node_features)
        edge_update = self.node_cb(
            normed, x_mask, edge_features,
            node_batch.coboundary_senders, node_batch.coboundary_receivers, node_batch.coboundary_mask,
        )
        ring_update = self.node_bd(
            normed, x_mask, ring_features,
            node_batch.boundary_senders, node_batch.boundary_receivers, node_batch.boundary_mask,
        )
        node_features = self.node_update.combine(node_features, edge_update, ring_update, x_mask)

        node_features = self.chem_reminder(node_features, chem_embeddings, x_mask)

        return node_features, edge_features, ring_features


class ScalarTPaiNNMolecular(nnx.Module):
    """
    Topology-only TPaiNN molecular model with ring topology and complex features.

    Like ScalarGNNMolecular but includes ring (dim-2) cells derived from RCP
    connectivity.  Rings contribute a third gated channel in ``UpdateFromCochains``.

    Architecture
    ------------
    Init  : cx ← cx_encoder( segment_sum(chem_embeddings, graph_idx) )
    Layer 0: same as ScalarGNNMolecular + ring coboundary path
             cx ← UpdateFromCochains(cx, nodes, edges, rings)
    Layers 1+: ScalarTPaiNNLayer → UpdateFromCochains(with rings)
    Readout: ComplexMultiPropertyReadout(cx per layer) → (G, num_outputs)

    Args:
        num_species         : Vocabulary size for element embedding.
        num_outputs         : Number of molecular properties (default 4).
        num_node_scalars    : Node feature width.
        num_edge_scalars    : Edge feature width.
        num_ring_scalars    : Ring feature width.
        num_complex_scalars : Graph-level feature width (default = num_node_scalars).
        embedding_dim       : Chemical embedding width.
        hidden_dim          : MLP hidden width.
        num_layers          : Total layers including layer 0.
        use_atom_features   : Whether to include pre-computed atomic properties.
        rngs                : Flax RNG container.
    """

    def __init__(
        self,
        num_species: int,
        num_outputs: int = 4,
        num_node_scalars: int = 64,
        num_edge_scalars: int = 64,
        num_ring_scalars: int = 32,
        num_complex_scalars: int = 64,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 3,
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

        self.chemical_embedding = ChemicalEmbedding(
            num_species=num_species,
            embedding_dim=embedding_dim,
            use_atom_features=use_atom_features,
            rngs=rngs,
        )

        # Complex feature initialisation
        self.cx_encoder      = nnx.Linear(embedding_dim, num_complex_scalars, rngs=rngs)
        self.cx_encoder_norm = nnx.LayerNorm(num_features=num_complex_scalars, rngs=rngs)

        # Layer 0: encoders
        self.node_encoder      = NoGeoNodeEncoder(num_node_scalars, embedding_dim, hidden_dim, rngs)
        self.node_encoder_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)
        self.edge_encoder      = NoGeoEdgeEncoder(num_node_scalars, num_edge_scalars, hidden_dim, rngs)
        self.edge_encoder_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)

        # Layer 0: edge down, ring coboundary path, dual-channel node update
        self.l0_edge_dn       = NoGeoEdgeDownMessages(num_edge_scalars, num_node_scalars, hidden_dim, rngs)
        self.l0_edge_dn_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)

        self.l0_ring_encoder  = ScalarBagEncoder(num_edge_scalars, num_ring_scalars, hidden_dim, rngs)
        self.l0_edge_cb       = ScalarEdgeCoboundaryMessages(num_edge_scalars, num_ring_scalars, hidden_dim, rngs)
        self.l0_edge_cb_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)

        self.l0_edge_ffn      = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs)
        self.l0_edge_ffn_block= ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs)

        self.l0_node_cb       = ScalarNodeCoboundaryMessages(num_node_scalars, num_edge_scalars, hidden_dim, rngs)
        self.l0_node_bd       = ScalarNodeBoundaryMessages(num_node_scalars, num_ring_scalars, hidden_dim, rngs)
        self.l0_node_update   = ScalarDualChannelNodeUpdate(num_node_scalars, hidden_dim, rngs)
        self.l0_chem_reminder = ScalarChemicalReminder(num_node_scalars, embedding_dim, hidden_dim, rngs)

        # Layers 1+
        self.layers = [
            ScalarTPaiNNLayer(
                num_node_scalars, num_edge_scalars, num_ring_scalars,
                embedding_dim, hidden_dim, rngs,
            )
            for _ in range(num_layers - 1)
        ]

        # One UpdateFromCochains per layer (with ring channel)
        self.cx_updates = [
            UpdateFromCochains(num_complex_scalars, num_node_scalars, num_edge_scalars,
                               d_ring=num_ring_scalars, hidden_dim=hidden_dim, rngs=rngs)
            for _ in range(num_layers)
        ]

        self.readout = ComplexMultiPropertyReadout(
            d_cx=num_complex_scalars,
            num_outputs=num_outputs,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rngs=rngs,
        )

    def __call__(
        self,
        complex_batch: ComplexBatch,
        graph_idx: Optional[jnp.ndarray] = None,
        num_graphs: int = 1,
    ) -> Dict[str, jnp.ndarray]:
        node_batch = complex_batch.cochain_batches[0]
        edge_batch = complex_batch.cochain_batches[1]
        ring_batch = complex_batch.cochain_batches[2]

        node_static = node_batch.static
        assert node_static is not None
        assert node_batch.x_mask is not None
        species_indices = node_static['Z']
        atom_features  = None
        if self.use_atom_features:
            keys = ['N', 'LI', 'Mu_X', 'Mu_Y', 'Mu_Z', 'Q_XY', 'Q_XZ', 'Q_YZ', 'Q_aniso', 'Q_ZZ']
            atom_features = jnp.stack([node_static[k] for k in keys], axis=-1)

        chem_embeddings = self.chemical_embedding(species_indices, atom_features)

        # Initialise complex (graph-level) features.
        # cx has shape (G+1, d_cx) including one OOB "padding complex" row.
        if complex_batch.x is not None:
            cx = self.cx_encoder_norm(self.cx_encoder(complex_batch.x))
        else:
            h_init = jnp.where(node_batch.x_mask[:, None], chem_embeddings, 0.0)
            cx_pool = (jax.ops.segment_sum(h_init, graph_idx, num_graphs + 1)
                       if graph_idx is not None else h_init.sum(axis=0, keepdims=True))
            cx = self.cx_encoder_norm(self.cx_encoder(cx_pool))  # (G+1, d_cx)

        assert edge_batch.owner_cochains is not None
        assert ring_batch.owner_cochains is not None
        edge_graph_idx = edge_batch.owner_cochains    # (E_total,) edge → graph
        ring_graph_idx = ring_batch.owner_cochains    # (R_total,) ring → graph

        # Layer 0: node encoder
        node_features = self.node_encoder(
            chem_embeddings, node_batch.x_mask,
            node_batch.up_senders, node_batch.up_receivers, node_batch.up_mask,
        )
        node_features = self.node_encoder_norm(node_features)
        if node_batch.x_mask is not None:
            node_features = jnp.where(node_batch.x_mask[:, None], node_features, 0.0)

        # Layer 0: edge encoder
        edge_features = self.edge_encoder(
            node_features,
            edge_batch.boundary_senders, edge_batch.boundary_receivers, edge_batch.boundary_mask,
            edge_batch.x_mask,
        )
        edge_features = self.edge_encoder_norm(edge_features)
        if edge_batch.x_mask is not None:
            edge_features = jnp.where(edge_batch.x_mask[:, None], edge_features, 0.0)

        # Layer 0: edge down
        normed_e = self.l0_edge_dn_block.pre_norm(edge_features)
        dn = self.l0_edge_dn(
            normed_e, edge_batch.x_mask, node_features,
            edge_batch.down_senders, edge_batch.down_receivers,
            edge_batch.down_intermediaries, edge_batch.down_mask,
        )
        edge_features = self.l0_edge_dn_block.residual(edge_features, dn, edge_batch.x_mask)

        # Layer 0: ring encoder + edge coboundary
        ring_features = self.l0_ring_encoder(
            edge_features,
            ring_batch.boundary_senders, ring_batch.boundary_receivers,
            ring_batch.boundary_mask, ring_batch.x_mask,
        )
        normed_e = self.l0_edge_cb_block.pre_norm(edge_features)
        ecb = self.l0_edge_cb(
            normed_e, edge_batch.x_mask, ring_features,
            edge_batch.coboundary_senders, edge_batch.coboundary_receivers, edge_batch.coboundary_mask,
        )
        edge_features = self.l0_edge_cb_block.residual(edge_features, ecb, edge_batch.x_mask)

        # Layer 0: edge FFN
        normed_e = self.l0_edge_ffn_block.pre_norm(edge_features)
        ffn = self.l0_edge_ffn(normed_e, edge_batch.x_mask)
        edge_features = self.l0_edge_ffn_block.residual(edge_features, ffn, edge_batch.x_mask)

        # Layer 0: dual-channel node update
        normed = self.l0_node_update.pre_norm(node_features)
        edge_update = self.l0_node_cb(
            normed, node_batch.x_mask, edge_features,
            node_batch.coboundary_senders, node_batch.coboundary_receivers, node_batch.coboundary_mask,
        )
        ring_update = self.l0_node_bd(
            normed, node_batch.x_mask, ring_features,
            node_batch.boundary_senders, node_batch.boundary_receivers, node_batch.boundary_mask,
        )
        node_features = self.l0_node_update.combine(node_features, edge_update, ring_update, node_batch.x_mask)
        node_features = self.l0_chem_reminder(node_features, chem_embeddings, node_batch.x_mask)

        # Update complex features after layer 0 (nodes + edges + rings)
        cx = self.cx_updates[0](
            cx, node_features, edge_features, ring_features,
            node_graph_idx=graph_idx, edge_graph_idx=edge_graph_idx, ring_graph_idx=ring_graph_idx,
            node_mask=node_batch.x_mask, edge_mask=edge_batch.x_mask, ring_mask=ring_batch.x_mask,
            num_graphs=num_graphs,
        )
        complex_features = [cx]

        # Layers 1+
        for i, layer in enumerate(self.layers):
            node_features, edge_features, ring_features = layer(
                node_features, edge_features, chem_embeddings, node_batch, edge_batch, ring_batch
            )
            cx = self.cx_updates[i + 1](
                cx, node_features, edge_features, ring_features,
                node_graph_idx=graph_idx, edge_graph_idx=edge_graph_idx, ring_graph_idx=ring_graph_idx,
                node_mask=node_batch.x_mask, edge_mask=edge_batch.x_mask, ring_mask=ring_batch.x_mask,
                num_graphs=num_graphs,
            )
            complex_features.append(cx)

        predictions = self.readout.read_and_combine(complex_features)

        return {
            'node_features':    node_features,
            'edge_features':    edge_features,
            'complex_features': cx[:num_graphs],            # (G, d_cx) — real graphs only
            'predictions':      predictions[:num_graphs],   # (G, num_outputs)
            'x_mask':           node_batch.x_mask,
        }
