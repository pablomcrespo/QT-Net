"""
V2 models for the inference experiment.

Three models, each built from the normalised-gate layers in layers_inference:

- **SGNN_v2**: Scalar GNN with configurable distance encoder
  (``distance_encoder='bessel'`` for cutoff, ``'gaussian'`` for FC).
- **EGNN_v2**: Equivariant GNN with GeometryReminder at NodeCoboundary
  (step 5) and EdgeGeometryReminder at EdgeDown (step 3).
"""

from typing import Optional, Dict, Tuple

import jax.numpy as jnp
import jax
import flax.nnx as nnx

# Shared building blocks from original codebase
from qtnet.jax_models.scalar_layers import (
    ChemicalEmbedding,
    ScalarChemicalReminder,
    ScalarGatedResidual,
    ScalarNodeCoboundaryMessages,
    ScalarEdgeFFN,
    ScalarAtomicHead,
    PerLayerReadout,
)
from qtnet.jax_models.equivariant_layers import (
    EquivariantNodeEncoder,
    EquivariantEdgeEncoder,
    EquivariantNodeUpMessages,
    EquivariantEdgeBoundaryMessages,
    EquivariantEdgeDownMessages,
    EquivariantNodeCoboundaryMessages,
    EquivariantNorm,
    EquivariantGatedResidual,
    EquivariantDualChannelUpdate,
    EquivariantFFN,
    ChemicalReminder,
    GeometryReminder,
    EquivariantPerLayerReadout,
)
from qtnet.jax_models.representations import ComplexBatch

# V2 layers
from qtnet.jax_models.layers_inference import (
    BesselCutoffEncoder,
    GaussianRBF,
    ScalarNodeEncoder_v2,
    ScalarNodeUpMessages_v2,
    ScalarEdgeEncoder_v2,
    ScalarEdgeBoundaryMessages_v2,
    ScalarEdgeDownMessages_v2,
    EdgeGeometryReminder,
)


# ============================================================================
# Scalar GNN Layer v2  (shared cycle for SGNN_v2)
# ============================================================================

class ScalarGNNLayer_v2(nnx.Module):
    """Cycle: NodeUp → EdgeBoundary → EdgeDown → EdgeFFN → NodeCoboundary → ChemReminder."""

    def __init__(
        self,
        num_node_scalars: int,
        num_edge_scalars: int,
        embedding_dim: int,
        distance_encoder: nnx.Module,
        hidden_dim: int = 64,
        geometric_filter_dim: int = 64,
        legendre_basis_dim: int = 8,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.node_up = ScalarNodeUpMessages_v2(
            num_scalars=num_node_scalars,
            distance_encoder=distance_encoder,
            hidden_dim=hidden_dim,
            num_edge_scalars=num_edge_scalars,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.node_up_block = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs=rngs)
        self.node_up_edge_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)

        self.edge_boundary = ScalarEdgeBoundaryMessages_v2(
            num_node_scalars=num_node_scalars,
            num_edge_scalars=num_edge_scalars,
            distance_encoder=distance_encoder,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            rngs=rngs,
        )
        self.edge_boundary_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_boundary_node_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)

        self.edge_down = ScalarEdgeDownMessages_v2(
            num_edge_scalars=num_edge_scalars,
            num_node_scalars=num_node_scalars,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            legendre_basis_dim=legendre_basis_dim,
            rngs=rngs,
        )
        self.edge_down_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_down_node_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)

        self.edge_ffn = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_ffn_block = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)

        self.node_coboundary = ScalarNodeCoboundaryMessages(
            num_node_scalars=num_node_scalars,
            num_edge_scalars=num_edge_scalars,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.node_coboundary_block = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs=rngs)
        self.node_coboundary_edge_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)

        self.chemical_reminder = ScalarChemicalReminder(
            num_scalars=num_node_scalars,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_features: jnp.ndarray,
        chem_embeddings: jnp.ndarray,
        node_adj: Dict,
        edge_adj: Dict,
        node_static: Optional[Dict] = None,
        edge_static: Optional[Dict] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # 1. NodeUp
        normed_nodes = self.node_up_block.pre_norm(node_features)
        node_up_update = self.node_up(
            x=normed_nodes, x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'),
            static=node_static,
            edge_features=self.node_up_edge_norm(edge_features),
            up_intermediaries=node_adj.get('up_intermediaries'),
        )
        node_features = self.node_up_block.residual(node_features, node_up_update, node_adj['x_mask'])

        # 2. EdgeBoundary
        normed_edges = self.edge_boundary_block.pre_norm(edge_features)
        edge_boundary_update = self.edge_boundary(
            x=normed_edges, x_mask=edge_adj['x_mask'],
            boundary_x=self.edge_boundary_node_norm(node_features),
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj.get('boundary_mask'),
            static=edge_static,
        )
        edge_features = self.edge_boundary_block.residual(edge_features, edge_boundary_update, edge_adj['x_mask'])

        # 3. EdgeDown
        normed_edges = self.edge_down_block.pre_norm(edge_features)
        edge_down_update = self.edge_down(
            x=normed_edges, x_mask=edge_adj['x_mask'],
            node_features=self.edge_down_node_norm(node_features),
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static,
        )
        edge_features = self.edge_down_block.residual(edge_features, edge_down_update, edge_adj['x_mask'])

        # 4. EdgeFFN
        normed_edges = self.edge_ffn_block.pre_norm(edge_features)
        edge_ffn_update = self.edge_ffn(normed_edges, edge_adj['x_mask'])
        edge_features = self.edge_ffn_block.residual(edge_features, edge_ffn_update, edge_adj['x_mask'])

        # 5. NodeCoboundary
        normed_nodes = self.node_coboundary_block.pre_norm(node_features)
        node_cob_update = self.node_coboundary(
            x=normed_nodes, x_mask=node_adj['x_mask'],
            coboundary_x=self.node_coboundary_edge_norm(edge_features),
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj.get('coboundary_mask'),
        )
        node_features = self.node_coboundary_block.residual(node_features, node_cob_update, node_adj['x_mask'])

        # 6. ChemReminder
        node_features = self.chemical_reminder(
            x=node_features, chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'],
        )
        return node_features, edge_features


# ============================================================================
# SGNN_v2
# ============================================================================

class SGNN_v2(nnx.Module):
    """Scalar GNN with Bessel+cutoff normalised gates.

    Same architecture as ScalarGNN but all geometric gates use
    [normalised geometry, RBF(distance)] inputs.
    """

    def __init__(
        self,
        num_species: int,
        num_node_scalars: int = 32,
        num_edge_scalars: int = 32,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        distance_encoder: str = 'bessel',
        geometric_filter_dim: int = 64,
        num_layers: int = 3,
        num_rbf_basis: int = 16,
        legendre_basis_dim: int = 8,
        cutoff: float = 8.0,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        self.num_node_scalars = num_node_scalars
        self.num_edge_scalars = num_edge_scalars
        self.num_layers = num_layers

        if distance_encoder == 'bessel':
            dist_enc = BesselCutoffEncoder(num_basis=num_rbf_basis, cutoff=cutoff)
        else:
            dist_enc = GaussianRBF(num_rbf_basis)

        self.chemical_embedding = ChemicalEmbedding(
            num_species=num_species, embedding_dim=embedding_dim, rngs=rngs)

        self.node_encoder = ScalarNodeEncoder_v2(
            num_scalar_out=num_node_scalars, embedding_dim=embedding_dim,
            distance_encoder=dist_enc, hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim, rngs=rngs,
        )
        self.node_encoder_norm = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)

        self.edge_encoder = ScalarEdgeEncoder_v2(
            num_node_scalars=num_node_scalars, num_scalar_out=num_edge_scalars,
            distance_encoder=dist_enc, hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim, rngs=rngs,
        )
        self.edge_encoder_norm = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)

        # Layer 0 tail: EdgeDown → EdgeFFN → NodeCoboundary → ChemReminder
        self.edge_down_init = ScalarEdgeDownMessages_v2(
            num_edge_scalars=num_edge_scalars, num_node_scalars=num_node_scalars,
            hidden_dim=hidden_dim, geometric_filter_dim=geometric_filter_dim,
            legendre_basis_dim=legendre_basis_dim, rngs=rngs,
        )
        self.edge_down_block_init = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_down_node_norm_init = nnx.LayerNorm(num_features=num_node_scalars, rngs=rngs)

        self.edge_ffn_init = ScalarEdgeFFN(num_edge_scalars, hidden_dim, rngs=rngs)
        self.edge_ffn_block_init = ScalarGatedResidual(num_edge_scalars, hidden_dim, rngs=rngs)

        self.node_coboundary_init = ScalarNodeCoboundaryMessages(
            num_node_scalars=num_node_scalars, num_edge_scalars=num_edge_scalars,
            hidden_dim=hidden_dim, rngs=rngs,
        )
        self.node_coboundary_block_init = ScalarGatedResidual(num_node_scalars, hidden_dim, rngs=rngs)
        self.node_coboundary_edge_norm_init = nnx.LayerNorm(num_features=num_edge_scalars, rngs=rngs)

        self.chemical_reminder_init = ScalarChemicalReminder(
            num_scalars=num_node_scalars, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, rngs=rngs,
        )

        self.layers = [
            ScalarGNNLayer_v2(
                num_node_scalars=num_node_scalars, num_edge_scalars=num_edge_scalars,
                embedding_dim=embedding_dim, distance_encoder=dist_enc,
                hidden_dim=hidden_dim, geometric_filter_dim=geometric_filter_dim,
                legendre_basis_dim=legendre_basis_dim, rngs=rngs,
            )
            for _ in range(num_layers - 1)
        ]

        self.readout = PerLayerReadout(
            num_scalars_in=num_node_scalars, hidden_dim=hidden_dim,
            num_layers=num_layers, rngs=rngs,
        )

    def __call__(self, complex_batch: ComplexBatch) -> Dict[str, jnp.ndarray]:
        node_batch = complex_batch.cochain_batches[0]
        edge_batch = complex_batch.cochain_batches[1]

        node_adj = {
            'up_senders': node_batch.up_senders,
            'up_receivers': node_batch.up_receivers,
            'up_intermediaries': node_batch.up_intermediaries,
            'up_mask': node_batch.up_mask,
            'x_mask': node_batch.x_mask,
            'coboundary_senders': node_batch.coboundary_senders,
            'coboundary_receivers': node_batch.coboundary_receivers,
            'coboundary_mask': node_batch.coboundary_mask,
        }
        edge_adj = {
            'boundary_senders': edge_batch.boundary_senders,
            'boundary_receivers': edge_batch.boundary_receivers,
            'boundary_mask': edge_batch.boundary_mask,
            'down_senders': edge_batch.down_senders,
            'down_receivers': edge_batch.down_receivers,
            'down_intermediaries': edge_batch.down_intermediaries,
            'down_mask': edge_batch.down_mask,
            'x_mask': edge_batch.x_mask,
        }
        node_static = node_batch.static
        edge_static = edge_batch.static
        species_indices = node_static['Z']

        chem_embeddings = self.chemical_embedding(species_indices)

        # --- Layer 0 ---
        node_features = self.node_encoder(
            chem_embeddings=chem_embeddings, x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'], up_receivers=node_adj['up_receivers'],
            up_mask=node_adj['up_mask'], static=node_static,
        )
        node_features = self.node_encoder_norm(node_features)
        if node_adj['x_mask'] is not None:
            node_features = jnp.where(node_adj['x_mask'][:, None], node_features, 0.0)

        edge_features = self.edge_encoder(
            boundary_x=node_features,
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj['boundary_mask'],
            x_mask=edge_adj['x_mask'], static=edge_static,
        )
        edge_features = self.edge_encoder_norm(edge_features)
        if edge_adj['x_mask'] is not None:
            edge_features = jnp.where(edge_adj['x_mask'][:, None], edge_features, 0.0)

        normed = self.edge_down_block_init.pre_norm(edge_features)
        ed_up = self.edge_down_init(
            x=normed, x_mask=edge_adj['x_mask'],
            node_features=self.edge_down_node_norm_init(node_features),
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'), static=edge_static,
        )
        edge_features = self.edge_down_block_init.residual(edge_features, ed_up, edge_adj['x_mask'])

        normed = self.edge_ffn_block_init.pre_norm(edge_features)
        edge_features = self.edge_ffn_block_init.residual(
            edge_features, self.edge_ffn_init(normed, edge_adj['x_mask']), edge_adj['x_mask'])

        normed_nodes = self.node_coboundary_block_init.pre_norm(node_features)
        nc_up = self.node_coboundary_init(
            x=normed_nodes, x_mask=node_adj['x_mask'],
            coboundary_x=self.node_coboundary_edge_norm_init(edge_features),
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj['coboundary_mask'],
        )
        node_features = self.node_coboundary_block_init.residual(node_features, nc_up, node_adj['x_mask'])

        node_features = self.chemical_reminder_init(
            x=node_features, chem_embeddings=chem_embeddings, x_mask=node_adj['x_mask'])

        layer_features = [node_features]
        for layer in self.layers:
            node_features, edge_features = layer(
                node_features, edge_features, chem_embeddings,
                node_adj, edge_adj, node_static, edge_static,
            )
            layer_features.append(node_features)

        output = self.readout.read_and_combine(layer_features, node_adj['x_mask'])
        return {
            'node_features': node_features, 'edge_features': edge_features,
            'scalars': output['scalars'], 'vectors': output['vectors'],
            'tensors': output['tensors'], 'x_mask': node_adj['x_mask'],
        }



# ============================================================================
# EGNN_v2 layer
# ============================================================================

class EquivariantGNNLayer_v2(nnx.Module):
    """Enhanced EquivariantGNNLayer with two additions:

    - **Step 3**: EdgeDown + EdgeGeometryReminder → DualChannelUpdate on edges.
    - **Step 5**: NodeCoboundary + GeometryReminder2 → DualChannelUpdate on nodes.
    """

    def __init__(
        self,
        node_irreps: str,
        edge_irreps: str,
        embedding_dim: int,
        hidden_dim: int = 64,
        hidden_l1_channels: int = 4,
        hidden_l2_channels: int = 4,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        cutoff: float = 8.0,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps

        # === 1. NodeUp + GeometryReminder1 → DualChannelUpdate ===
        self.node_up_messages = EquivariantNodeUpMessages(
            irreps_in=node_irreps, hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels, hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim, geo_basis_dim=geo_basis_dim,
            cutoff=cutoff, edge_irreps=edge_irreps, rngs=rngs,
        )
        self.geo_reinjection = GeometryReminder(
            irreps_in=node_irreps, hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels, hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim, geo_basis_dim=geo_basis_dim,
            cutoff=cutoff, edge_irreps=edge_irreps, rngs=rngs,
        )
        self.node_up_dual = EquivariantDualChannelUpdate(
            irreps=node_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        self.node_up_edge_norm = EquivariantNorm(irreps=edge_irreps, rngs=rngs)

        # === 2. EdgeBoundary ===
        self.edge_boundary_messages = EquivariantEdgeBoundaryMessages(
            node_irreps=node_irreps, edge_irreps=edge_irreps,
            hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim, geo_basis_dim=geo_basis_dim,
            cutoff=cutoff, rngs=rngs,
        )
        self.edge_boundary_res = EquivariantGatedResidual(
            irreps=edge_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        self.edge_boundary_node_norm = EquivariantNorm(irreps=node_irreps, rngs=rngs)

        # === 3. EdgeDown + EdgeGeometryReminder → DualChannelEdgeUpdate ===
        self.edge_down_messages = EquivariantEdgeDownMessages(
            edge_irreps=edge_irreps, node_irreps=node_irreps,
            hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim, geo_basis_dim=geo_basis_dim,
            rngs=rngs,
        )
        self.edge_geo_reminder = EdgeGeometryReminder(
            edge_irreps=edge_irreps, node_irreps=node_irreps,
            hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim, geo_basis_dim=geo_basis_dim,
            rngs=rngs,
        )
        self.edge_down_dual = EquivariantDualChannelUpdate(
            irreps=edge_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        self.edge_down_node_norm = EquivariantNorm(irreps=node_irreps, rngs=rngs)

        # === 4. EdgeFFN ===
        self.edge_self_mixing = EquivariantFFN(
            irreps=edge_irreps, hidden_dim=hidden_dim, rngs=rngs)
        self.edge_mix_res = EquivariantGatedResidual(
            irreps=edge_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)

        # === 5. NodeCoboundary + GeometryReminder2 → DualChannelNodeUpdate ===
        self.node_coboundary_messages = EquivariantNodeCoboundaryMessages(
            node_irreps=node_irreps, edge_irreps=edge_irreps,
            hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels, rngs=rngs,
        )
        self.geo_reinjection_2 = GeometryReminder(
            irreps_in=node_irreps, hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels, hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim, geo_basis_dim=geo_basis_dim,
            cutoff=cutoff, edge_irreps=edge_irreps, rngs=rngs,
        )
        self.node_coboundary_dual = EquivariantDualChannelUpdate(
            irreps=node_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        self.node_coboundary_edge_norm = EquivariantNorm(irreps=edge_irreps, rngs=rngs)

        # === 6. ChemicalReminder ===
        self.chemical_reminder = ChemicalReminder(
            irreps=node_irreps, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, rngs=rngs,
        )

    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_features: jnp.ndarray,
        chem_embeddings: jnp.ndarray,
        node_adj: Dict,
        edge_adj: Dict,
        node_static: Optional[Dict] = None,
        edge_static: Optional[Dict] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # === 1.1 NodeUp ===
        normed_nodes = self.node_up_dual.pre_norm(node_features, node_adj['x_mask'])
        normed_edges = self.node_up_edge_norm(edge_features, edge_adj['x_mask'])

        node_up_out = self.node_up_messages(
            x=normed_nodes, x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'], up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'), static=node_static,
            edge_features=normed_edges, up_intermediaries=node_adj.get('up_intermediaries'),
        )

        # === 1.2 GeometryReminder1 ===
        geo_out = self.geo_reinjection(
            x=normed_nodes, x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'], up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'), static=node_static,
            edge_features=normed_edges, up_intermediaries=node_adj.get('up_intermediaries'),
        )
        node_features = self.node_up_dual.combine(
            node_features, node_up_out['x'], geo_out['x'], node_adj['x_mask'])

        # === 2. EdgeBoundary ===
        normed_edges = self.edge_boundary_res.pre_norm(edge_features, edge_adj['x_mask'])
        eb_out = self.edge_boundary_messages(
            x=normed_edges, x_mask=edge_adj['x_mask'],
            boundary_x=self.edge_boundary_node_norm(node_features, node_adj['x_mask']),
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj.get('boundary_mask'), static=edge_static,
        )
        edge_features = self.edge_boundary_res.residual(
            edge_features, eb_out['x'], edge_adj['x_mask'])

        # === 3.1 EdgeDown ===
        normed_edges_for_down = self.edge_down_dual.pre_norm(edge_features, edge_adj['x_mask'])
        normed_nodes_for_down = self.edge_down_node_norm(node_features, node_adj['x_mask'])

        ed_out = self.edge_down_messages(
            x=normed_edges_for_down, x_mask=edge_adj['x_mask'],
            node_features=normed_nodes_for_down,
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'), static=edge_static,
        )

        # === 3.2 EdgeGeometryReminder ===
        egr_out = self.edge_geo_reminder(
            x=normed_edges_for_down, x_mask=edge_adj['x_mask'],
            node_features=normed_nodes_for_down,
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static, node_static=node_static,
        )
        edge_features = self.edge_down_dual.combine(
            edge_features, ed_out['x'], egr_out['x'], edge_adj['x_mask'])

        # === 4. EdgeFFN ===
        normed_edges = self.edge_mix_res.pre_norm(edge_features, edge_adj['x_mask'])
        emix = self.edge_self_mixing(x=normed_edges, x_mask=edge_adj['x_mask'])
        edge_features = self.edge_mix_res.residual(
            edge_features, emix['x'], edge_adj['x_mask'])

        # === 5.1 NodeCoboundary ===
        normed_nodes_for_cob = self.node_coboundary_dual.pre_norm(node_features, node_adj['x_mask'])
        normed_edges_for_cob = self.node_coboundary_edge_norm(edge_features, edge_adj['x_mask'])

        cob_out = self.node_coboundary_messages(
            x=normed_nodes_for_cob, x_mask=node_adj['x_mask'],
            coboundary_x=normed_edges_for_cob,
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj.get('coboundary_mask'),
        )

        # === 5.2 GeometryReminder2 ===
        geo_out_2 = self.geo_reinjection_2(
            x=normed_nodes_for_cob, x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'], up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'), static=node_static,
            edge_features=normed_edges_for_cob,
            up_intermediaries=node_adj.get('up_intermediaries'),
        )
        node_features = self.node_coboundary_dual.combine(
            node_features, cob_out['x'], geo_out_2['x'], node_adj['x_mask'])

        # === 6. ChemicalReminder ===
        node_features = self.chemical_reminder(
            x=node_features, chem_embeddings=chem_embeddings, x_mask=node_adj['x_mask'])

        return node_features, edge_features


# ============================================================================
# EGNN_v2
# ============================================================================

class EGNN_v2(nnx.Module):
    """Equivariant GNN with GeometryReminder at NodeCoboundary and
    EdgeGeometryReminder at EdgeDown.

    Layer 0 is the standard equivariant encoding, identical to EquivariantGNN.
    Layers 1+ use :class:`EquivariantGNNLayer_v2`.
    """

    def __init__(
        self,
        num_species: int,
        num_node_scalars: int = 8,
        num_node_vectors: int = 4,
        num_node_tensors: int = 2,
        num_edge_scalars: int = 8,
        num_edge_vectors: int = 4,
        num_edge_tensors: int = 4,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        hidden_l1_channels: int = 8,
        hidden_l2_channels: int = 8,
        num_layers: int = 3,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        cutoff: float = 8.0,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs

        self.node_irreps = f"{num_node_scalars}x0e + {num_node_vectors}x1o + {num_node_tensors}x2e"
        self.edge_irreps = f"{num_edge_scalars}x0e + {num_edge_vectors}x1o + {num_edge_tensors}x2e"
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # === Chemical Embedding ===
        self.chemical_embedding = ChemicalEmbedding(
            num_species=num_species, embedding_dim=embedding_dim, rngs=rngs)

        # === Layer 0: Encoders ===
        self.node_encoder = EquivariantNodeEncoder(
            num_scalar_out=num_node_scalars, num_vector_out=num_node_vectors,
            num_tensor_out=num_node_tensors, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim, cutoff=cutoff, rngs=rngs,
        )
        self.node_encoder_norm = EquivariantNorm(irreps=self.node_irreps, rngs=rngs)

        self.edge_encoder = EquivariantEdgeEncoder(
            node_irreps_in=self.node_irreps,
            num_scalar_out=num_edge_scalars, num_l1_out=num_edge_vectors,
            num_l2_out=num_edge_tensors, hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim, cutoff=cutoff, rngs=rngs,
        )
        self.edge_encoder_norm = EquivariantNorm(irreps=self.edge_irreps, rngs=rngs)

        # Layer 0 tail: EdgeDown + EdgeGeoReminder → DualChannelUpdate → EdgeFFN → NodeCob → ChemReminder
        self.edge_down_init = EquivariantEdgeDownMessages(
            edge_irreps=self.edge_irreps, node_irreps=self.node_irreps,
            hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim, rngs=rngs,
        )
        self.edge_geo_reminder_init = EdgeGeometryReminder(
            edge_irreps=self.edge_irreps, node_irreps=self.node_irreps,
            hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim, rngs=rngs,
        )
        self.edge_down_dual_init = EquivariantDualChannelUpdate(
            irreps=self.edge_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        self.edge_down_node_norm_init = EquivariantNorm(
            irreps=self.node_irreps, rngs=rngs)

        self.edge_self_mixing_init = EquivariantFFN(
            irreps=self.edge_irreps, hidden_dim=hidden_dim, rngs=rngs)
        self.edge_mix_res_init = EquivariantGatedResidual(
            irreps=self.edge_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)

        self.node_coboundary_init = EquivariantNodeCoboundaryMessages(
            node_irreps=self.node_irreps, edge_irreps=self.edge_irreps,
            hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels, rngs=rngs,
        )
        self.node_coboundary_res_init = EquivariantGatedResidual(
            irreps=self.node_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        self.node_coboundary_edge_norm_init = EquivariantNorm(
            irreps=self.edge_irreps, rngs=rngs)

        self.chemical_reminder_init = ChemicalReminder(
            irreps=self.node_irreps, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, rngs=rngs,
        )

        # === Layers 1+ ===
        self.layers = [
            EquivariantGNNLayer_v2(
                node_irreps=self.node_irreps, edge_irreps=self.edge_irreps,
                embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                hidden_l1_channels=hidden_l1_channels,
                hidden_l2_channels=hidden_l2_channels,
                geometric_filter_dim=geometric_filter_dim,
                geo_basis_dim=geo_basis_dim, cutoff=cutoff, rngs=rngs,
            )
            for _ in range(num_layers - 1)
        ]

        # === Per-Layer Readout ===
        self.readout = EquivariantPerLayerReadout(
            node_irreps=self.node_irreps, hidden_dim=hidden_dim,
            num_layers=num_layers, rngs=rngs,
        )

    def __call__(self, complex_batch: ComplexBatch) -> Dict[str, jnp.ndarray]:
        node_batch = complex_batch.cochain_batches[0]
        edge_batch = complex_batch.cochain_batches[1]

        node_adj = {
            'up_senders': node_batch.up_senders,
            'up_receivers': node_batch.up_receivers,
            'up_mask': node_batch.up_mask,
            'up_intermediaries': node_batch.up_intermediaries,
            'x_mask': node_batch.x_mask,
            'coboundary_senders': node_batch.coboundary_senders,
            'coboundary_receivers': node_batch.coboundary_receivers,
            'coboundary_mask': node_batch.coboundary_mask,
        }
        edge_adj = {
            'boundary_senders': edge_batch.boundary_senders,
            'boundary_receivers': edge_batch.boundary_receivers,
            'boundary_mask': edge_batch.boundary_mask,
            'down_senders': edge_batch.down_senders,
            'down_receivers': edge_batch.down_receivers,
            'down_intermediaries': edge_batch.down_intermediaries,
            'down_mask': edge_batch.down_mask,
            'x_mask': edge_batch.x_mask,
        }
        node_static = node_batch.static
        edge_static = edge_batch.static
        species_indices = node_static['Z']

        chem_embeddings = self.chemical_embedding(species_indices)

        # === Layer 0 ===
        node_features = self.node_encoder(
            chem_embeddings=chem_embeddings, x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'], up_receivers=node_adj['up_receivers'],
            up_mask=node_adj['up_mask'], static=node_static,
        )
        node_features = self.node_encoder_norm(x=node_features, x_mask=node_adj['x_mask'])

        edge_out = self.edge_encoder(
            boundary_x=node_features,
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj['boundary_mask'],
            x_mask=edge_adj['x_mask'], static=edge_static,
        )
        edge_features = self.edge_encoder_norm(x=edge_out['x'], x_mask=edge_adj['x_mask'])

        # EdgeDown + EdgeGeoReminder (layer 0)
        normed_edges = self.edge_down_dual_init.pre_norm(edge_features, edge_adj['x_mask'])
        normed_nodes_for_down = self.edge_down_node_norm_init(node_features, node_adj['x_mask'])

        ed_out = self.edge_down_init(
            x=normed_edges, x_mask=edge_adj['x_mask'],
            node_features=normed_nodes_for_down,
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'), static=edge_static,
        )
        egr_out = self.edge_geo_reminder_init(
            x=normed_edges, x_mask=edge_adj['x_mask'],
            node_features=normed_nodes_for_down,
            down_senders=edge_adj['down_senders'],
            down_receivers=edge_adj['down_receivers'],
            down_intermediaries=edge_adj['down_intermediaries'],
            down_mask=edge_adj.get('down_mask'),
            static=edge_static, node_static=node_static,
        )
        edge_features = self.edge_down_dual_init.combine(
            edge_features, ed_out['x'], egr_out['x'], edge_adj['x_mask'])

        # EdgeFFN (layer 0)
        normed_edges = self.edge_mix_res_init.pre_norm(edge_features, edge_adj['x_mask'])
        emix = self.edge_self_mixing_init(x=normed_edges, x_mask=edge_adj['x_mask'])
        edge_features = self.edge_mix_res_init.residual(
            edge_features, emix['x'], edge_adj['x_mask'])

        # NodeCoboundary (layer 0)
        normed_nodes = self.node_coboundary_res_init.pre_norm(node_features, node_adj['x_mask'])
        normed_edges_cob = self.node_coboundary_edge_norm_init(edge_features, edge_adj['x_mask'])
        cob_out = self.node_coboundary_init(
            x=normed_nodes, x_mask=node_adj['x_mask'],
            coboundary_x=normed_edges_cob,
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj['coboundary_mask'],
        )
        node_features = self.node_coboundary_res_init.residual(
            node_features, cob_out['x'], node_adj['x_mask'])

        # ChemReminder (layer 0)
        node_features = self.chemical_reminder_init(
            x=node_features, chem_embeddings=chem_embeddings, x_mask=node_adj['x_mask'])

        node_features_per_layer = [node_features]

        # === Layers 1+ ===
        for layer in self.layers:
            node_features, edge_features = layer(
                node_features, edge_features, chem_embeddings,
                node_adj, edge_adj, node_static, edge_static,
            )
            node_features_per_layer.append(node_features)

        # === Per-Layer Readout ===
        head_output = self.readout(
            node_features_per_layer=node_features_per_layer,
            x_mask=node_adj['x_mask'],
        )
        return {
            'node_features': node_features, 'edge_features': edge_features,
            'scalars': head_output['scalars'], 'vectors': head_output['vectors'],
            'tensors': head_output['tensors'], 'x_mask': node_adj['x_mask'],
        }


# ============================================================================
# EGNX: Flexible equivariant GNN for ablation studies
# ============================================================================

class EquivariantGNNLayer_Flex(nnx.Module):
    """Flexible equivariant GNN layer for ablation studies (EGNX).

    Step 1 (NodeUp) ALWAYS uses a GeometryReminder + DualChannelUpdate to
    ensure geometric primitives (r̂, T) are always injected at the primary
    node-update step.

    Flags:
        use_node_geo_twice: Also add GeometryReminder2 + DualChannelUpdate
            at NodeCoboundary (step 5). When False, NodeCob uses a plain
            EquivariantGatedResidual.
        use_edge_geo_reminder: EdgeDown uses EdgeGeometryReminder +
            DualChannelUpdate (step 3). When False, plain gated residual.
        use_tensor_products: NodeUp and NodeCoboundary use within-sender
            tensor product mixing (L=2×L=1→L=1, L=1×L=1→L=2).

    Also includes a NodeFFN (EquivariantFFN) after NodeUp to mix within-node
    irreps before sending to EdgeBoundary.
    """

    def __init__(
        self,
        node_irreps: str,
        edge_irreps: str,
        embedding_dim: int,
        hidden_dim: int = 64,
        hidden_l1_channels: int = 4,
        hidden_l2_channels: int = 4,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        cutoff: float = 5.25,
        use_node_geo_twice: bool = False,
        use_edge_geo_reminder: bool = False,
        use_tensor_products: bool = False,
        hidden_tp_dim: int = 8,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps
        self.use_node_geo_twice = use_node_geo_twice
        self.use_edge_geo_reminder = use_edge_geo_reminder

        # === 1. NodeUp + GeometryReminder1 → DualChannelUpdate (always) ===
        self.node_up_messages = EquivariantNodeUpMessages(
            irreps_in=node_irreps, hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim, cutoff=cutoff,
            edge_irreps=edge_irreps,
            use_tensor_products=use_tensor_products,
            hidden_tp_dim=hidden_tp_dim, rngs=rngs,
        )
        self.geo_reinjection = GeometryReminder(
            irreps_in=node_irreps, hidden_dim=hidden_dim,
            hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim, cutoff=cutoff,
            edge_irreps=edge_irreps, rngs=rngs,
        )
        self.node_up_dual = EquivariantDualChannelUpdate(
            irreps=node_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        self.node_up_edge_norm = EquivariantNorm(irreps=edge_irreps, rngs=rngs)

        # Node FFN: within-node irrep mixing after NodeUp, before EdgeBoundary
        self.node_ffn = EquivariantFFN(
            irreps=node_irreps, hidden_dim=hidden_dim, rngs=rngs)
        self.node_ffn_block = EquivariantGatedResidual(
            irreps=node_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)

        # === 2. EdgeBoundary ===
        self.edge_boundary_messages = EquivariantEdgeBoundaryMessages(
            node_irreps=node_irreps, edge_irreps=edge_irreps,
            hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim, cutoff=cutoff, rngs=rngs,
        )
        self.edge_boundary_res = EquivariantGatedResidual(
            irreps=edge_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        self.edge_boundary_node_norm = EquivariantNorm(
            irreps=node_irreps, rngs=rngs)

        # === 3. EdgeDown (+ optional EdgeGeometryReminder → DualChannel) ===
        self.edge_down_messages = EquivariantEdgeDownMessages(
            edge_irreps=edge_irreps, node_irreps=node_irreps,
            hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim, rngs=rngs,
        )
        if use_edge_geo_reminder:
            self.edge_geo_reminder = EdgeGeometryReminder(
                edge_irreps=edge_irreps, node_irreps=node_irreps,
                hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
                hidden_l2_channels=hidden_l2_channels,
                geometric_filter_dim=geometric_filter_dim,
                geo_basis_dim=geo_basis_dim, rngs=rngs,
            )
            self.edge_down_block = EquivariantDualChannelUpdate(
                irreps=edge_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        else:
            self.edge_down_block = EquivariantGatedResidual(
                irreps=edge_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        self.edge_down_node_norm = EquivariantNorm(irreps=node_irreps, rngs=rngs)

        # === 4. EdgeFFN ===
        self.edge_self_mixing = EquivariantFFN(
            irreps=edge_irreps, hidden_dim=hidden_dim, rngs=rngs)
        self.edge_mix_res = EquivariantGatedResidual(
            irreps=edge_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)

        # === 5. NodeCoboundary (+ optional GeometryReminder2 → DualChannel) ===
        self.node_coboundary_messages = EquivariantNodeCoboundaryMessages(
            node_irreps=node_irreps, edge_irreps=edge_irreps,
            hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            use_tensor_products=use_tensor_products,
            hidden_tp_dim=hidden_tp_dim, rngs=rngs,
        )
        if use_node_geo_twice:
            self.geo_reinjection_2 = GeometryReminder(
                irreps_in=node_irreps, hidden_dim=hidden_dim,
                hidden_l1_channels=hidden_l1_channels,
                hidden_l2_channels=hidden_l2_channels,
                geometric_filter_dim=geometric_filter_dim,
                geo_basis_dim=geo_basis_dim, cutoff=cutoff,
                edge_irreps=edge_irreps, rngs=rngs,
            )
            self.node_coboundary_block = EquivariantDualChannelUpdate(
                irreps=node_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        else:
            self.node_coboundary_block = EquivariantGatedResidual(
                irreps=node_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        self.node_coboundary_edge_norm = EquivariantNorm(
            irreps=edge_irreps, rngs=rngs)

        # === 6. ChemicalReminder ===
        self.chemical_reminder = ChemicalReminder(
            irreps=node_irreps, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, rngs=rngs,
        )

    def __call__(
        self,
        node_features: jnp.ndarray,
        edge_features: jnp.ndarray,
        chem_embeddings: jnp.ndarray,
        node_adj: Dict,
        edge_adj: Dict,
        node_static: Optional[Dict] = None,
        edge_static: Optional[Dict] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # === 1. NodeUp + GeometryReminder1 → DualChannelUpdate ===
        normed_nodes = self.node_up_dual.pre_norm(node_features, node_adj['x_mask'])
        normed_edges = self.node_up_edge_norm(edge_features, edge_adj['x_mask'])
        node_up_out = self.node_up_messages(
            x=normed_nodes, x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'), static=node_static,
            edge_features=normed_edges,
            up_intermediaries=node_adj.get('up_intermediaries'),
        )
        geo_out = self.geo_reinjection(
            x=normed_nodes, x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj.get('up_mask'), static=node_static,
            edge_features=normed_edges,
            up_intermediaries=node_adj.get('up_intermediaries'),
        )
        node_features = self.node_up_dual.combine(
            node_features, node_up_out['x'], geo_out['x'], node_adj['x_mask'])

        # Node FFN: within-node irrep mixing before EdgeBoundary
        normed_nodes = self.node_ffn_block.pre_norm(node_features)
        node_ffn_out = self.node_ffn(x=normed_nodes, x_mask=node_adj['x_mask'])
        node_features = self.node_ffn_block.residual(
            node_features, node_ffn_out['x'], node_adj['x_mask'])

        # === 2. EdgeBoundary ===
        normed_edges = self.edge_boundary_res.pre_norm(
            edge_features, edge_adj['x_mask'])
        eb_out = self.edge_boundary_messages(
            x=normed_edges, x_mask=edge_adj['x_mask'],
            boundary_x=self.edge_boundary_node_norm(
                node_features, node_adj['x_mask']),
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj.get('boundary_mask'),
            static=edge_static,
        )
        edge_features = self.edge_boundary_res.residual(
            edge_features, eb_out['x'], edge_adj['x_mask'])

        # === 3. EdgeDown (+ optional EdgeGeometryReminder) ===
        if self.use_edge_geo_reminder:
            normed_edges_for_down = self.edge_down_block.pre_norm(
                edge_features, edge_adj['x_mask'])
            normed_nodes_for_down = self.edge_down_node_norm(
                node_features, node_adj['x_mask'])
            ed_out = self.edge_down_messages(
                x=normed_edges_for_down, x_mask=edge_adj['x_mask'],
                node_features=normed_nodes_for_down,
                down_senders=edge_adj['down_senders'],
                down_receivers=edge_adj['down_receivers'],
                down_intermediaries=edge_adj['down_intermediaries'],
                down_mask=edge_adj.get('down_mask'), static=edge_static,
            )
            egr_out = self.edge_geo_reminder(
                x=normed_edges_for_down, x_mask=edge_adj['x_mask'],
                node_features=normed_nodes_for_down,
                down_senders=edge_adj['down_senders'],
                down_receivers=edge_adj['down_receivers'],
                down_intermediaries=edge_adj['down_intermediaries'],
                down_mask=edge_adj.get('down_mask'),
                static=edge_static, node_static=node_static,
            )
            edge_features = self.edge_down_block.combine(
                edge_features, ed_out['x'], egr_out['x'], edge_adj['x_mask'])
        else:
            normed_edges = self.edge_down_block.pre_norm(edge_features)
            normed_nodes_for_down = self.edge_down_node_norm(
                node_features, node_adj['x_mask'])
            ed_out = self.edge_down_messages(
                x=normed_edges, x_mask=edge_adj['x_mask'],
                node_features=normed_nodes_for_down,
                down_senders=edge_adj['down_senders'],
                down_receivers=edge_adj['down_receivers'],
                down_intermediaries=edge_adj['down_intermediaries'],
                down_mask=edge_adj.get('down_mask'), static=edge_static,
            )
            edge_features = self.edge_down_block.residual(
                edge_features, ed_out['x'], edge_adj['x_mask'])

        # === 4. EdgeFFN ===
        normed_edges = self.edge_mix_res.pre_norm(edge_features, edge_adj['x_mask'])
        emix = self.edge_self_mixing(x=normed_edges, x_mask=edge_adj['x_mask'])
        edge_features = self.edge_mix_res.residual(
            edge_features, emix['x'], edge_adj['x_mask'])

        # === 5. NodeCoboundary (+ optional GeometryReminder2) ===
        if self.use_node_geo_twice:
            normed_nodes_for_cob = self.node_coboundary_block.pre_norm(
                node_features, node_adj['x_mask'])
            normed_edges_for_cob = self.node_coboundary_edge_norm(
                edge_features, edge_adj['x_mask'])
            cob_out = self.node_coboundary_messages(
                x=normed_nodes_for_cob, x_mask=node_adj['x_mask'],
                coboundary_x=normed_edges_for_cob,
                coboundary_senders=node_adj['coboundary_senders'],
                coboundary_receivers=node_adj['coboundary_receivers'],
                coboundary_mask=node_adj.get('coboundary_mask'),
            )
            geo_out_2 = self.geo_reinjection_2(
                x=normed_nodes_for_cob, x_mask=node_adj['x_mask'],
                up_senders=node_adj['up_senders'],
                up_receivers=node_adj['up_receivers'],
                up_mask=node_adj.get('up_mask'), static=node_static,
                edge_features=normed_edges_for_cob,
                up_intermediaries=node_adj.get('up_intermediaries'),
            )
            node_features = self.node_coboundary_block.combine(
                node_features, cob_out['x'], geo_out_2['x'], node_adj['x_mask'])
        else:
            normed_nodes = self.node_coboundary_block.pre_norm(node_features)
            normed_edges_for_cob = self.node_coboundary_edge_norm(
                edge_features, edge_adj['x_mask'])
            cob_out = self.node_coboundary_messages(
                x=normed_nodes, x_mask=node_adj['x_mask'],
                coboundary_x=normed_edges_for_cob,
                coboundary_senders=node_adj['coboundary_senders'],
                coboundary_receivers=node_adj['coboundary_receivers'],
                coboundary_mask=node_adj.get('coboundary_mask'),
            )
            node_features = self.node_coboundary_block.residual(
                node_features, cob_out['x'], node_adj['x_mask'])

        # === 6. ChemicalReminder ===
        node_features = self.chemical_reminder(
            x=node_features, chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'])

        return node_features, edge_features


class EquivariantGNN_Flex(nnx.Module):
    """Flexible equivariant GNN (EGNX).

    Layer 0 tail: EdgeDown (+ optional EdgeGeoReminder) → EdgeFFN →
                  NodeCoboundary (with TP if enabled) → ChemReminder.
    Layers 1+: EquivariantGNNLayer_Flex with the configured flags.

    Flags:
        use_node_geo_twice: Apply GeometryReminder at both NodeUp (always)
            and NodeCoboundary (optional, this flag).
        use_edge_geo_reminder: EdgeDown uses EdgeGeometryReminder + DualChannel.
        use_tensor_products: Within-sender TP mixing in NodeUp and NodeCob.
    """

    def __init__(
        self,
        num_species: int,
        num_node_scalars: int = 8,
        num_node_vectors: int = 4,
        num_node_tensors: int = 2,
        num_edge_scalars: int = 8,
        num_edge_vectors: int = 4,
        num_edge_tensors: int = 4,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        hidden_l1_channels: int = 8,
        hidden_l2_channels: int = 8,
        num_layers: int = 3,
        geometric_filter_dim: int = 32,
        geo_basis_dim: int = 16,
        cutoff: float = 5.25,
        use_node_geo_twice: bool = False,
        use_edge_geo_reminder: bool = False,
        use_tensor_products: bool = False,
        hidden_tp_dim: int = 8,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs

        self.node_irreps = (
            f"{num_node_scalars}x0e + {num_node_vectors}x1o + {num_node_tensors}x2e"
        )
        self.edge_irreps = (
            f"{num_edge_scalars}x0e + {num_edge_vectors}x1o + {num_edge_tensors}x2e"
        )
        self.num_layers = num_layers
        self.use_edge_geo_reminder = use_edge_geo_reminder

        # === Chemical Embedding ===
        from qtnet.jax_models.scalar_layers import ChemicalEmbedding
        self.chemical_embedding = ChemicalEmbedding(
            num_species=num_species, embedding_dim=embedding_dim, rngs=rngs)

        # === Layer 0: Encoders ===
        self.node_encoder = EquivariantNodeEncoder(
            num_scalar_out=num_node_scalars,
            num_vector_out=num_node_vectors,
            num_tensor_out=num_node_tensors,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim, cutoff=cutoff, rngs=rngs,
        )
        self.node_encoder_norm = EquivariantNorm(
            irreps=self.node_irreps, rngs=rngs)

        self.edge_encoder = EquivariantEdgeEncoder(
            node_irreps_in=self.node_irreps,
            num_scalar_out=num_edge_scalars,
            num_l1_out=num_edge_vectors,
            num_l2_out=num_edge_tensors,
            hidden_dim=hidden_dim,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim, cutoff=cutoff, rngs=rngs,
        )
        self.edge_encoder_norm = EquivariantNorm(
            irreps=self.edge_irreps, rngs=rngs)

        # Layer 0 tail: EdgeDown (+ optional EdgeGeoReminder) → EdgeFFN →
        #               NodeCoboundary (with TP if enabled) → ChemReminder
        self.edge_down_init = EquivariantEdgeDownMessages(
            edge_irreps=self.edge_irreps, node_irreps=self.node_irreps,
            hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            geometric_filter_dim=geometric_filter_dim,
            geo_basis_dim=geo_basis_dim, rngs=rngs,
        )
        if use_edge_geo_reminder:
            self.edge_geo_reminder_init = EdgeGeometryReminder(
                edge_irreps=self.edge_irreps, node_irreps=self.node_irreps,
                hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
                hidden_l2_channels=hidden_l2_channels,
                geometric_filter_dim=geometric_filter_dim,
                geo_basis_dim=geo_basis_dim, rngs=rngs,
            )
            self.edge_down_block_init = EquivariantDualChannelUpdate(
                irreps=self.edge_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        else:
            self.edge_down_block_init = EquivariantGatedResidual(
                irreps=self.edge_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        self.edge_down_node_norm_init = EquivariantNorm(
            irreps=self.node_irreps, rngs=rngs)

        self.edge_self_mixing_init = EquivariantFFN(
            irreps=self.edge_irreps, hidden_dim=hidden_dim, rngs=rngs)
        self.edge_mix_res_init = EquivariantGatedResidual(
            irreps=self.edge_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)

        self.node_coboundary_init = EquivariantNodeCoboundaryMessages(
            node_irreps=self.node_irreps, edge_irreps=self.edge_irreps,
            hidden_dim=hidden_dim, hidden_l1_channels=hidden_l1_channels,
            hidden_l2_channels=hidden_l2_channels,
            use_tensor_products=use_tensor_products,
            hidden_tp_dim=hidden_tp_dim, rngs=rngs,
        )
        self.node_coboundary_res_init = EquivariantGatedResidual(
            irreps=self.node_irreps, gate_hidden_dim=hidden_dim, rngs=rngs)
        self.node_coboundary_edge_norm_init = EquivariantNorm(
            irreps=self.edge_irreps, rngs=rngs)

        self.chemical_reminder_init = ChemicalReminder(
            irreps=self.node_irreps, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, rngs=rngs,
        )

        # === Layers 1+ ===
        self.layers = [
            EquivariantGNNLayer_Flex(
                node_irreps=self.node_irreps,
                edge_irreps=self.edge_irreps,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                hidden_l1_channels=hidden_l1_channels,
                hidden_l2_channels=hidden_l2_channels,
                geometric_filter_dim=geometric_filter_dim,
                geo_basis_dim=geo_basis_dim,
                cutoff=cutoff,
                use_node_geo_twice=use_node_geo_twice,
                use_edge_geo_reminder=use_edge_geo_reminder,
                use_tensor_products=use_tensor_products,
                hidden_tp_dim=hidden_tp_dim,
                rngs=rngs,
            )
            for _ in range(num_layers - 1)
        ]

        # === Per-Layer Readout ===
        self.readout = EquivariantPerLayerReadout(
            node_irreps=self.node_irreps, hidden_dim=hidden_dim,
            num_layers=num_layers, rngs=rngs,
        )

    def __call__(self, complex_batch: ComplexBatch) -> Dict[str, jnp.ndarray]:
        node_batch = complex_batch.cochain_batches[0]
        edge_batch = complex_batch.cochain_batches[1]

        node_adj = {
            'up_senders': node_batch.up_senders,
            'up_receivers': node_batch.up_receivers,
            'up_mask': node_batch.up_mask,
            'up_intermediaries': node_batch.up_intermediaries,
            'x_mask': node_batch.x_mask,
            'coboundary_senders': node_batch.coboundary_senders,
            'coboundary_receivers': node_batch.coboundary_receivers,
            'coboundary_mask': node_batch.coboundary_mask,
        }
        edge_adj = {
            'boundary_senders': edge_batch.boundary_senders,
            'boundary_receivers': edge_batch.boundary_receivers,
            'boundary_mask': edge_batch.boundary_mask,
            'down_senders': edge_batch.down_senders,
            'down_receivers': edge_batch.down_receivers,
            'down_intermediaries': edge_batch.down_intermediaries,
            'down_mask': edge_batch.down_mask,
            'x_mask': edge_batch.x_mask,
        }
        node_static = node_batch.static
        edge_static = edge_batch.static
        species_indices = node_static['Z']

        chem_embeddings = self.chemical_embedding(species_indices)

        # === Layer 0 ===
        node_features = self.node_encoder(
            chem_embeddings=chem_embeddings, x_mask=node_adj['x_mask'],
            up_senders=node_adj['up_senders'],
            up_receivers=node_adj['up_receivers'],
            up_mask=node_adj['up_mask'], static=node_static,
        )
        node_features = self.node_encoder_norm(
            x=node_features, x_mask=node_adj['x_mask'])

        edge_out = self.edge_encoder(
            boundary_x=node_features,
            boundary_senders=edge_adj['boundary_senders'],
            boundary_receivers=edge_adj['boundary_receivers'],
            boundary_mask=edge_adj['boundary_mask'],
            x_mask=edge_adj['x_mask'], static=edge_static,
        )
        edge_features = self.edge_encoder_norm(
            x=edge_out['x'], x_mask=edge_adj['x_mask'])

        # EdgeDown (+ optional EdgeGeoReminder) at layer 0
        normed_nodes_for_down = self.edge_down_node_norm_init(
            node_features, node_adj['x_mask'])
        if self.use_edge_geo_reminder:
            normed_edges = self.edge_down_block_init.pre_norm(
                edge_features, edge_adj['x_mask'])
            ed_out = self.edge_down_init(
                x=normed_edges, x_mask=edge_adj['x_mask'],
                node_features=normed_nodes_for_down,
                down_senders=edge_adj['down_senders'],
                down_receivers=edge_adj['down_receivers'],
                down_intermediaries=edge_adj['down_intermediaries'],
                down_mask=edge_adj.get('down_mask'), static=edge_static,
            )
            egr_out = self.edge_geo_reminder_init(
                x=normed_edges, x_mask=edge_adj['x_mask'],
                node_features=normed_nodes_for_down,
                down_senders=edge_adj['down_senders'],
                down_receivers=edge_adj['down_receivers'],
                down_intermediaries=edge_adj['down_intermediaries'],
                down_mask=edge_adj.get('down_mask'),
                static=edge_static, node_static=node_static,
            )
            edge_features = self.edge_down_block_init.combine(
                edge_features, ed_out['x'], egr_out['x'], edge_adj['x_mask'])
        else:
            normed_edges = self.edge_down_block_init.pre_norm(edge_features)
            ed_out = self.edge_down_init(
                x=normed_edges, x_mask=edge_adj['x_mask'],
                node_features=normed_nodes_for_down,
                down_senders=edge_adj['down_senders'],
                down_receivers=edge_adj['down_receivers'],
                down_intermediaries=edge_adj['down_intermediaries'],
                down_mask=edge_adj.get('down_mask'), static=edge_static,
            )
            edge_features = self.edge_down_block_init.residual(
                edge_features, ed_out['x'], edge_adj['x_mask'])

        # EdgeFFN at layer 0
        normed_edges = self.edge_mix_res_init.pre_norm(
            edge_features, edge_adj['x_mask'])
        emix = self.edge_self_mixing_init(x=normed_edges, x_mask=edge_adj['x_mask'])
        edge_features = self.edge_mix_res_init.residual(
            edge_features, emix['x'], edge_adj['x_mask'])

        # NodeCoboundary at layer 0
        normed_nodes = self.node_coboundary_res_init.pre_norm(
            node_features, node_adj['x_mask'])
        normed_edges_cob = self.node_coboundary_edge_norm_init(
            edge_features, edge_adj['x_mask'])
        cob_out = self.node_coboundary_init(
            x=normed_nodes, x_mask=node_adj['x_mask'],
            coboundary_x=normed_edges_cob,
            coboundary_senders=node_adj['coboundary_senders'],
            coboundary_receivers=node_adj['coboundary_receivers'],
            coboundary_mask=node_adj['coboundary_mask'],
        )
        node_features = self.node_coboundary_res_init.residual(
            node_features, cob_out['x'], node_adj['x_mask'])

        # ChemReminder at layer 0
        node_features = self.chemical_reminder_init(
            x=node_features, chem_embeddings=chem_embeddings,
            x_mask=node_adj['x_mask'])

        node_features_per_layer = [node_features]

        # === Layers 1+ ===
        for layer in self.layers:
            node_features, edge_features = layer(
                node_features, edge_features, chem_embeddings,
                node_adj, edge_adj, node_static, edge_static,
            )
            node_features_per_layer.append(node_features)

        # === Per-Layer Readout ===
        head_output = self.readout(
            node_features_per_layer=node_features_per_layer,
            x_mask=node_adj['x_mask'],
        )
        return {
            'node_features': node_features, 'edge_features': edge_features,
            'scalars': head_output['scalars'], 'vectors': head_output['vectors'],
            'tensors': head_output['tensors'], 'x_mask': node_adj['x_mask'],
        }
