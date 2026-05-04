"""
Base cochain layer with tensor products. Provides utilities for feature
extraction and constructors for linear, gate, MLP, and norm layers.
"""
from typing import Optional, Any, Dict, List

import flax.nnx as nnx
import jax.numpy as jnp
import jax

from qtnet.jax_models.layer_utils import (
    IrrepsInfo,
    build_irreps_info,
    EPS,
    extract_by_indices,
    compute_l1_norms,
    compute_l1_cosine_similarity,
    compute_l2_norms,
    compute_tensor_alignment,
    compute_radial_basis_bessel,
    compute_smooth_cutoff,
    compute_legendre_basis,
    apply_channel_wise_linear_l1,
    apply_channel_wise_linear_l2,
    l2_to_matrix,
    contract_l2_l1,
    outer_l1_l1_to_l2,
)


class BaseCochainTensorProduct(nnx.Module):
    """
    Minimal base class for equivariant cochain layers with tensor products.
    
    FEATURE STORAGE CONVENTION (e3nn standard):
    ------------------------------------------
    Features are stored as FLAT arrays where different irreps are concatenated.
    
    Example for irreps = "4x0e + 2x1o" (4 scalars + 2 vectors):
        - 4 scalar instances: 4 values (indices 0-3)
        - 2 vector instances: 2×3 = 6 values (indices 4-9)
        - Total shape: (batch, 10) as a flat array
        
        Memory layout: [s0, s1, s2, s3, v0_x, v0_y, v0_z, v1_x, v1_y, v1_z]
                        ^-- scalars --^  ^--- vector 0 ---^  ^--- vector 1 ---^
    
    MULTIPLICITY (channels):
    -----------------------
    The "4x" in "4x0e" means 4 independent scalar channels (like 4 features).
    The "2x" in "2x1o" means 2 independent vector channels (each a 3D vector).
    
    GATING:
    ------
    For equivariance, each irrep INSTANCE is gated by a SINGLE scalar:
    - "4x0e + 2x1o" needs 6 gates total (4 for scalars + 2 for vectors)
    - Gate i multiplies ALL components of irrep instance i
    - For vectors: gate[k] * [vx, vy, vz] → same scalar multiplies all 3 components
    
    This class provides:
    - Feature extraction utilities (scalars, vectors, norms)
    - Constructors: build_linear, build_gate, build_mlp, build_norm
    - Aggregation helper
    
    Args:
        irreps_in (str): Input irreps for x (e.g., "4x0e + 2x1o")
        boundary_irreps (str, optional): Irreps for boundary_x. Defaults to irreps_in.
        coboundary_irreps (str, optional): Irreps for coboundary_x. Defaults to irreps_in.
        aggr (str): Default aggregation type ('add', 'mean', 'max'). Default: 'add'
        rngs: Flax random number generator.
    """
    
    def __init__(
        self,
        irreps_in: str,
        boundary_irreps: Optional[str] = None,
        coboundary_irreps: Optional[str] = None,
        aggr: str = "add",
        rngs: nnx.Rngs = None
    ):
        super().__init__()
        
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.rngs = rngs
        self.aggr = aggr
        
        # Import e3nn
        try:
            import e3nn_jax as e3nn
            self.e3nn = e3nn
        except ImportError:
            raise ImportError("e3nn_jax is required. Install with: pip install e3nn-jax")
        
        # Store irreps
        self.irreps_in = e3nn.Irreps(irreps_in)
        self.boundary_irreps = e3nn.Irreps(boundary_irreps) if boundary_irreps else self.irreps_in
        self.coboundary_irreps = e3nn.Irreps(coboundary_irreps) if coboundary_irreps else self.irreps_in
        
        # Precompute irreps info for feature extraction
        self._irreps_info = {
            'x': self._build_irreps_info(self.irreps_in),
            'boundary_x': self._build_irreps_info(self.boundary_irreps),
            'coboundary_x': self._build_irreps_info(self.coboundary_irreps),
        }
    
    # ==================== Irreps Information ====================
    
    def _build_irreps_info(self, irreps, name: str = None) -> Dict[str, Any]:
        """
        Precompute useful info about an irreps specification.
        
        Structure:
        - 'irreps': the original irreps object
        - 'total_dim': total feature dimension
        - 'num_instances': total number of irrep instances (for gating)
        - 'gate_indices': maps each feature dim to its gate index
        - 'instance_dims': dimension of each irrep instance
        - 'by_l': dict keyed by L value, each containing:
            - 'num_instances': multiplicity for this L
            - 'indices': flat indices for all components
            - 'instance_slices': list of (start, end) for each instance
            - 'dim_per_instance': 2*l + 1
        
        Example for "4x0e + 2x1o + 1x2e":
            info['by_l'][0] = {
                'num_instances': 4,
                'indices': [0, 1, 2, 3],
                'instance_slices': [(0,1), (1,2), (2,3), (3,4)],
                'dim_per_instance': 1
            }
            info['by_l'][1] = {
                'num_instances': 2,
                'indices': [4, 5, 6, 7, 8, 9],
                'instance_slices': [(4,7), (7,10)],
                'dim_per_instance': 3
            }
            info['by_l'][2] = {
                'num_instances': 1,
                'indices': [10, 11, 12, 13, 14],
                'instance_slices': [(10,15)],
                'dim_per_instance': 5
            }
        """
        info = {
            'irreps': irreps,
            'total_dim': 0,
            'num_instances': 0,
            'gate_indices': [],
            'instance_dims': [],
            'by_l': {},  # Dict keyed by L value
        }
        
        idx = 0
        gate_counter = 0
        
        for mul, ir in irreps:
            l, p = ir.l, ir.p
            dim = ir.dim  # 2*l + 1 components per instance
            
            # Initialize entry for this L if needed
            if l not in info['by_l']:
                info['by_l'][l] = {
                    'num_instances': 0,
                    'indices': [],
                    'instance_slices': [],
                    'dim_per_instance': dim,
                }
            
            for m in range(mul):
                start = idx
                end = idx + dim
                
                # Track indices and slices for this L
                info['by_l'][l]['num_instances'] += 1
                info['by_l'][l]['indices'].extend(range(start, end))
                info['by_l'][l]['instance_slices'].append((start, end))
                
                # Gate indices: all components of this instance share same gate
                info['gate_indices'].extend([gate_counter] * dim)
                info['instance_dims'].append(dim)
                gate_counter += 1
                idx = end
        
        info['total_dim'] = idx
        info['num_instances'] = gate_counter
        info['gate_indices'] = tuple(info['gate_indices'])
        info['instance_dims'] = tuple(info['instance_dims'])
        
        # Convert indices lists to tuples (not arrays, to avoid nnx.state issues)
        for l in info['by_l']:
            info['by_l'][l]['indices'] = tuple(info['by_l'][l]['indices'])
            info['by_l'][l]['instance_slices'] = tuple(info['by_l'][l]['instance_slices'])
        
        # If name is provided, wrap in a dict with name as key
        if name is not None:
            return {name: info}
        return info
    
    def describe_irreps(self, space: str = 'x') -> str:
        """
        Get a human-readable description of an irreps specification.
        
        Example:
            >>> layer.describe_irreps('x')
            "4x0e + 2x1o: 10 dims, 6 instances (L0: 4, L1: 2)"
        """
        info = self._irreps_info[space]
        l_counts = ", ".join(f"L{l}: {info['by_l'][l]['num_instances']}" 
                            for l in sorted(info['by_l'].keys()))
        return (f"{info['irreps']}: {info['total_dim']} dims, "
                f"{info['num_instances']} instances ({l_counts})")
    
    # ==================== IrrepsInfo Registration ====================
    
    def register_space(self, name: str, irreps) -> IrrepsInfo:
        """Register an irreps space and return its precomputed metadata.
        
        Consolidates channel counting, index building, and gate index
        construction into a single call. The result is stored internally
        and also returned for direct use.
        
        Args:
            name: Name for this space (e.g., 'x', 'node', 'edge', 'bag')
            irreps: Irreps specification (string or e3nn.Irreps object)
            
        Returns:
            IrrepsInfo with precomputed channel counts, indices, and gate mapping
        """
        irreps_obj = self.e3nn.Irreps(irreps) if isinstance(irreps, str) else irreps
        info = build_irreps_info(irreps_obj)
        if not hasattr(self, '_spaces'):
            self._spaces = {}
        self._spaces[name] = info
        return info
    
    def get_space(self, name: str) -> IrrepsInfo:
        """Retrieve precomputed IrrepsInfo for a registered space."""
        return self._spaces[name]
    
    
    def apply_gates(self, features: jnp.ndarray, gates: jnp.ndarray, space: str = 'x') -> jnp.ndarray:
        """
        Apply gates to features using precomputed indices.
        
        Each irrep instance (scalar, vector, tensor) is multiplied by one gate scalar.
        All components of a vector/tensor share the same gate value.
        
        Args:
            features: Shape (batch, total_dim) - features to gate
            gates: Shape (batch, num_instances) - one gate per irrep instance
            space: Which space's irreps to use for indexing
            
        Returns:
            Gated features, shape (batch, total_dim)
        """
        gate_indices = self._irreps_info[space]['gate_indices']
        expanded_gates = gates[:, gate_indices]
        return features * expanded_gates
    
    # ==================== Feature Extraction ====================
    
    def extract_by_l(self, features: jnp.ndarray, l: int, space: str = 'x') -> jnp.ndarray:
        """
        Extract all features of a specific L value as flat array.
        
        For "4x0e + 2x1o" with shape (batch, 10):
            extract_by_l(x, 0) → (batch, 4) - the 4 scalar values
            extract_by_l(x, 1) → (batch, 6) - the 2 vectors flattened
            
        Args:
            features: Shape (batch, total_dim)
            l: Angular momentum to extract
            space: Which space's irreps to use
            
        Returns:
            Flat array of all instances of that L, shape (batch, num_instances * (2*l+1))
        """
        info = self._irreps_info[space]
        if l not in info['by_l']:
            return jnp.zeros((features.shape[0], 0))
        
        indices = jnp.array(info['by_l'][l]['indices'])
        return features[:, indices]
    
    def extract_scalars(self, features: jnp.ndarray, space: str = 'x') -> jnp.ndarray:
        """Extract scalar (L=0) features. Convenience wrapper for extract_by_l(features, 0)."""
        return self.extract_by_l(features, 0, space)
    
    def extract_vectors(self, features: jnp.ndarray, space: str = 'x') -> jnp.ndarray:
        """Extract vector (L=1) features as flat array. Convenience wrapper for extract_by_l(features, 1)."""
        return self.extract_by_l(features, 1, space)
    
    def extract_by_l_3d(self, features: jnp.ndarray, l: int, space: str = 'x') -> jnp.ndarray:
        """
        Extract features of a specific L value as 3D array with instances separated.
        
        For "4x0e + 2x1o" with shape (batch, 10):
            extract_by_l_3d(x, 0) → (batch, 4, 1) - 4 scalars, each with 1 component
            extract_by_l_3d(x, 1) → (batch, 2, 3) - 2 vectors, each with 3 components
            
        Args:
            features: Shape (batch, total_dim)
            l: Angular momentum to extract
            space: Which space's irreps to use
            
        Returns:
            Shape (batch, num_instances, 2*l+1)
        """
        info = self._irreps_info[space]
        if l not in info['by_l']:
            return jnp.zeros((features.shape[0], 0, 2*l+1))
        
        flat = self.extract_by_l(features, l, space)
        num_instances = info['by_l'][l]['num_instances']
        dim_per_instance = 2*l + 1
        return flat.reshape(features.shape[0], num_instances, dim_per_instance)
    
    def extract_vectors_3d(self, features: jnp.ndarray, space: str = 'x') -> jnp.ndarray:
        """Extract vector (L=1) features as 3D array. Convenience wrapper for extract_by_l_3d(features, 1)."""
        return self.extract_by_l_3d(features, 1, space)
    
    # ==================== Rotation-Invariant Scalar Computations ====================
    
    def extract_norms(
        self,
        features: jnp.ndarray,
        space: str = 'x',
        l_values: Optional[List[int]] = None,
        eps: float = 1e-8
    ) -> jnp.ndarray:
        """
        Extract norms of equivariant features (rotation-invariant).
        
        For L=0: |s| (absolute value)
        For L=1: ||v|| = sqrt(vx² + vy² + vz²)
        For L=2+: ||T|| = sqrt(sum of squared components)
        
        Args:
            features: Shape (batch, total_dim)
            space: Which space's irreps to use
            l_values: If specified, only compute norms for these L values.
                      E.g., l_values=[1,2] excludes scalars.
            eps: Numerical stability
            
        Returns:
            Norms shape (batch, num_selected_instances)
            
        Example for "4x0e + 2x1o":
            extract_norms(x) → (batch, 6) - all norms
            extract_norms(x, l_values=[1]) → (batch, 2) - only vector norms
        """
        info = self._irreps_info[space]
        irreps_obj = info['irreps']
        
        norms = []
        idx = 0
        
        for mul, ir in irreps_obj:
            l, dim = ir.l, ir.dim
            include = (l_values is None) or (l in l_values)
            
            for m in range(mul):
                if include:
                    components = features[:, idx:idx + dim]
                    if l == 0:
                        norm = jnp.abs(components)
                    else:
                        norm = jnp.sqrt(jnp.sum(components ** 2, axis=-1, keepdims=True) + eps)
                    norms.append(norm)
                idx += dim
        
        return jnp.concatenate(norms, axis=-1) if norms else jnp.zeros((features.shape[0], 0))
    
    def scalar_product(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        irreps,
        l_values: Optional[List[int]] = None,
    ) -> jnp.ndarray:
        """
        Compute scalar products between corresponding irrep instances of two feature sets.
        
        For each irrep instance, computes the inner product:
        - L=0: s1 * s2 (just multiplication)
        - L=1: <v1, v2> = v1·v2 = v1x*v2x + v1y*v2y + v1z*v2z
        - L=2+: <T1, T2> = sum(T1_ij * T2_ij) (Frobenius inner product)
        
        This is rotation-invariant: <R(v1), R(v2)> = <v1, v2>
        
        Args:
            x1: First feature set, shape (batch, total_dim)
            x2: Second feature set, shape (batch, total_dim) - MUST have same irreps as x1
            irreps: The irreps specification (string or Irreps object)
            l_values: If specified, only compute products for these L values.
                      E.g., l_values=[1] for only vector dot products.
                      
        Returns:
            Scalar products, shape (batch, num_selected_instances)
            
        Example:
            # Dot products between sender and receiver vectors
            # For "4x0e + 2x1o": returns (batch, 6) if l_values=None
            #                   returns (batch, 2) if l_values=[1]
            dots = self.scalar_product(sender, receiver, self.irreps_in, l_values=[1])
        """
        irreps_obj = self.e3nn.Irreps(irreps) if isinstance(irreps, str) else irreps
        
        products = []
        idx = 0
        
        for mul, ir in irreps_obj:
            l, dim = ir.l, ir.dim
            include = (l_values is None) or (l in l_values)
            
            for m in range(mul):
                if include:
                    comp1 = x1[:, idx:idx + dim]
                    comp2 = x2[:, idx:idx + dim]
                    # Inner product: sum over components
                    prod = jnp.sum(comp1 * comp2, axis=-1, keepdims=True)
                    products.append(prod)
                idx += dim
        
        return jnp.concatenate(products, axis=-1) if products else jnp.zeros((x1.shape[0], 0))
    
    
    
    # ==================== Irreps Utilities ====================
    
    def concatenate_irreps(self, irreps1, irreps2):
        """Get irreps for concatenating two irreps (adds multiplicities per L,p)."""
        mul_dict = {}
        for mul, ir in irreps1:
            key = (ir.l, ir.p)
            mul_dict[key] = mul_dict.get(key, 0) + mul
        for mul, ir in irreps2:
            key = (ir.l, ir.p)
            mul_dict[key] = mul_dict.get(key, 0) + mul
        
        irreps_list = [f"{mul}x{l}{'e' if p == 1 else 'o'}" for (l, p), mul in sorted(mul_dict.items())]
        return self.e3nn.Irreps(" + ".join(irreps_list))
    
    def concatenate_features(self, x1, x2, irreps1, irreps2):
        """Concatenate features properly interleaved by L-type."""
        # Extract by irrep type from x1
        parts1 = []
        idx1 = 0
        for mul, ir in irreps1:
            dim = mul * ir.dim
            parts1.append((ir.l, ir.p, x1[:, idx1:idx1+dim]))
            idx1 += dim
        
        # Extract by irrep type from x2
        parts2 = []
        idx2 = 0
        for mul, ir in irreps2:
            dim = mul * ir.dim
            parts2.append((ir.l, ir.p, x2[:, idx2:idx2+dim]))
            idx2 += dim
        
        # Combine by (l, p)
        combined = {}
        for l, p, features in parts1 + parts2:
            key = (l, p)
            if key not in combined:
                combined[key] = []
            combined[key].append(features)
        
        result_parts = [jnp.concatenate(combined[key], axis=-1) for key in sorted(combined.keys())]
        return jnp.concatenate(result_parts, axis=-1)
    
    # ==================== Constructors ====================
    
    def build_linear(self, irreps_in: str, irreps_out: str) -> nnx.Module:
        """
        Create an equivariant linear layer.
        
        Example:
            self.linear = self.build_linear("4x0e + 2x1o", "8x0e + 4x1o")
            out = self.linear(x)
        """
        import jax.random as jrandom
        
        irreps_in_obj = self.e3nn.Irreps(irreps_in) if isinstance(irreps_in, str) else irreps_in
        irreps_out_obj = self.e3nn.Irreps(irreps_out) if isinstance(irreps_out, str) else irreps_out
        
        tp = self.e3nn.FunctionalLinear(irreps_in_obj, irreps_out_obj)
        key = self.rngs()
        weights = nnx.Param(jrandom.normal(key, (tp.num_weights,)))
        e3nn_ref = self.e3nn
        
        class EquivariantLinear(nnx.Module):
            def __init__(self):
                self.tp = tp
                self.weights = weights
                self.irreps_in = irreps_in_obj
                self.irreps_out = irreps_out_obj
                self.e3nn = e3nn_ref
            
            def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
                def apply_single(x_single):
                    x_irreps = self.e3nn.IrrepsArray(self.irreps_in, x_single)
                    return self.tp(self.weights.value, x_irreps).array
                return jax.vmap(apply_single)(x)
        
        return EquivariantLinear()
    
    def build_scalar_gate(
        self,
        num_scalars_in: int,
        irreps_out: str,
        hidden_dim: Optional[int] = None,
        hidden_multiplier: int = 2,
        min_hidden: int = 16,
        activation: str = "sigmoid",
    ) -> nnx.Module:
        
        irreps_out_obj = self.e3nn.Irreps(irreps_out) if isinstance(irreps_out, str) else irreps_out
        
        # One gate per irrep INSTANCE
        num_gates = sum(mul for mul, ir in irreps_out_obj)
        
        if hidden_dim is None:
            hidden_dim = max(num_gates * hidden_multiplier, min_hidden)
        
        activation_map = {"sigmoid": nnx.sigmoid, "tanh": nnx.tanh, "relu": nnx.relu, "silu": nnx.silu}
        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}")
        act_fn = activation_map[activation]
        
        gate_mlp = nnx.Sequential(
            nnx.Linear(num_scalars_in, hidden_dim, rngs=self.rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, num_gates, rngs=self.rngs),
            act_fn
        )
        
        # Precompute gate_indices
        gate_indices = []
        gate_counter = 0
        for mul, ir in irreps_out_obj:
            for m in range(mul):
                gate_indices.extend([gate_counter] * ir.dim)
                gate_counter += 1
        gate_indices = jnp.array(gate_indices)
        
        class ScalarGateModule(nnx.Module):
            def __init__(self):
                self.gate_mlp = gate_mlp
                self.irreps_out = irreps_out_obj
                self.gate_indices = gate_indices
                self.num_scalars_in = num_scalars_in
                self.num_gates = num_gates
            
            def __call__(self, scalars: jnp.ndarray) -> jnp.ndarray:
                """
                Produce gate values from scalar invariants.
                
                Args:
                    scalars: Precomputed scalar invariants, shape (batch, num_scalars_in)
                    
                Returns:
                    Gate values, shape (batch, num_instances)
                """
                return self.gate_mlp(scalars)
            
            def apply_gates(self, features: jnp.ndarray, gates: jnp.ndarray) -> jnp.ndarray:
                """
                Apply gates to features via Hadamard product.
                
                Each irrep instance is multiplied by its corresponding gate scalar.
                For vectors/tensors, all components are multiplied by the SAME gate.
                """
                expanded_gates = gates[:, self.gate_indices]
                return features * expanded_gates
        
        return ScalarGateModule()
    
    def build_norm(
        self,
        irreps: str,
        eps: float = 1e-5,
    ) -> nnx.Module:
        """
        Create equivariant layer normalization.
        
        For scalar (L=0) features: Standard LayerNorm (subtract mean, divide by std)
            with learnable gamma and beta parameters.
        
        For L>0 features: E3Norm(x) = w * x / (eps + n)
            where n = sqrt(mean over nodes of squared norms per channel)
            and w is a learnable scale parameter per multiplicity.
        
        Args:
            irreps: Irreps specification (e.g., "8x0e + 4x1o")
            eps: Small constant for numerical stability
            
        Example:
            self.norm = self.build_norm("8x0e + 4x1o")
            normalized = self.norm(features)  # features shape: (num_nodes, total_dim)
        """
        irreps_obj = self.e3nn.Irreps(irreps) if isinstance(irreps, str) else irreps
        
        # Separate scalar and non-scalar irreps
        scalar_mul = sum(mul for mul, ir in irreps_obj if ir.l == 0)
        nonscalar_mul = sum(mul for mul, ir in irreps_obj if ir.l > 0)
        
        # Learnable parameters for scalars (standard LayerNorm)
        scalar_gamma = nnx.Param(jnp.ones(scalar_mul)) if scalar_mul > 0 else None
        scalar_beta = nnx.Param(jnp.zeros(scalar_mul)) if scalar_mul > 0 else None
        
        # Learnable scale for non-scalars (one w per multiplicity)
        nonscalar_w = nnx.Param(jnp.ones(nonscalar_mul)) if nonscalar_mul > 0 else None
        
        # Precompute indices and instance info
        scalar_indices = []
        nonscalar_instance_info = []  # List of (start_idx, end_idx, w_idx)
        
        idx = 0
        w_idx = 0
        for mul, ir in irreps_obj:
            dim = ir.dim
            if ir.l == 0:
                for m in range(mul):
                    scalar_indices.append(idx)
                    idx += 1
            else:
                for m in range(mul):
                    start = idx
                    end = idx + dim
                    nonscalar_instance_info.append((start, end, w_idx))
                    w_idx += 1
                    idx += dim
        
        scalar_indices = jnp.array(scalar_indices) if scalar_indices else None
        total_dim = idx
        
        class EquivariantNorm(nnx.Module):
            def __init__(self):
                self.irreps = irreps_obj
                self.eps = eps
                self.scalar_gamma = scalar_gamma
                self.scalar_beta = scalar_beta
                self.nonscalar_w = nonscalar_w
                self.scalar_indices = scalar_indices
                self.nonscalar_instance_info = nonscalar_instance_info
                self.total_dim = total_dim
            
            def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
                """
                Apply equivariant normalization.
                
                Args:
                    x: Features of shape (num_nodes, total_dim)
                    
                Returns:
                    Normalized features of shape (num_nodes, total_dim)
                """
                output = x.copy()
                
                # Handle scalars with standard LayerNorm
                if self.scalar_indices is not None and len(self.scalar_indices) > 0:
                    scalars = x[:, self.scalar_indices]  # (num_nodes, num_scalars)
                    # LayerNorm: normalize across feature dimension for each node
                    mean = jnp.mean(scalars, axis=-1, keepdims=True)
                    var = jnp.var(scalars, axis=-1, keepdims=True)
                    scalars_norm = (scalars - mean) / jnp.sqrt(var + self.eps)
                    # Apply learnable affine transform
                    scalars_norm = self.scalar_gamma.value[None, :] * scalars_norm + self.scalar_beta.value[None, :]
                    output = output.at[:, self.scalar_indices].set(scalars_norm)
                
                # Handle non-scalars with E3Norm
                # E3Norm(x) = w * x / (eps + n)
                # where n_c = sqrt(mean over nodes of squared norms for channel c)
                # n and w are arrays of shape (num_nonscalar_channels,)
                if len(self.nonscalar_instance_info) > 0:
                    # Compute squared norms for all channels: (num_nodes, num_channels)
                    norms_sq_per_channel = []
                    for start, end, w_idx in self.nonscalar_instance_info:
                        instance = x[:, start:end]  # (num_nodes, 2l+1)
                        norm_sq = jnp.sum(instance ** 2, axis=-1, keepdims=True)  # (num_nodes, 1)
                        norms_sq_per_channel.append(norm_sq)
                    
                    # Stack to (num_nodes, num_channels)
                    norms_sq = jnp.concatenate(norms_sq_per_channel, axis=-1)  # (num_nodes, num_channels)
                    
                    # n_c = sqrt(mean over nodes of squared norms) -> (num_channels,)
                    n = jnp.sqrt(jnp.mean(norms_sq, axis=0) + self.eps)  # (num_channels,)
                    
                    # Apply normalization: w * x / (eps + n) for each channel
                    # w and n are both (num_channels,), need to expand to match each component
                    for i, (start, end, w_idx) in enumerate(self.nonscalar_instance_info):
                        instance = x[:, start:end]  # (num_nodes, 2l+1)
                        w_c = self.nonscalar_w.value[w_idx]  # scalar for this channel
                        n_c = n[i]  # scalar for this channel
                        normalized = w_c * instance / (self.eps + n_c)
                        output = output.at[:, start:end].set(normalized)
                
                return output
        
        return EquivariantNorm()
    
    # ==================== Aggregation ====================
    
    def aggregate(
        self,
        messages: jnp.ndarray,
        index: jnp.ndarray,
        num_segments: int,
        aggr_type: Optional[str] = None,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Aggregate messages to target indices.
        
        Args:
            messages: Message features (num_messages, feature_dim)
            index: Target indices for each message (num_messages,)
            num_segments: Number of target elements
            aggr_type: Aggregation type ('add', 'mean', 'max'). Uses self.aggr if None.
            mask: Optional mask for valid messages (num_messages,)
        """
        aggr_type = aggr_type or self.aggr
        
        if mask is not None:
            if aggr_type in ("add", "mean"):
                messages = jnp.where(mask[:, None], messages, 0.0)
            elif aggr_type == "max":
                messages = jnp.where(mask[:, None], messages, -jnp.inf)
        
        if aggr_type == "add":
            return jax.ops.segment_sum(messages, index, num_segments)
        
        elif aggr_type == "mean":
            if mask is not None:
                counts = jax.ops.segment_sum(mask.astype(jnp.float32), index, num_segments)
            else:
                counts = jax.ops.segment_sum(
                    jnp.ones(messages.shape[0], dtype=messages.dtype), index, num_segments
                )
            sums = jax.ops.segment_sum(messages, index, num_segments)
            return jnp.where(counts[:, None] > 0, sums / counts[:, None], 0.0)
        
        elif aggr_type == "max":
            return jax.ops.segment_max(messages, index, num_segments)
        
        else:
            raise ValueError(f"Unknown aggregation type: {aggr_type}")
    
    # ==================== Forward Pass ====================
    
    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: jnp.ndarray,
        boundary_x: Optional[jnp.ndarray] = None,
        boundary_x_mask: Optional[jnp.ndarray] = None,
        coboundary_x: Optional[jnp.ndarray] = None,
        coboundary_x_mask: Optional[jnp.ndarray] = None,
        # Same-dimension adjacencies
        up_senders: Optional[jnp.ndarray] = None,
        up_receivers: Optional[jnp.ndarray] = None,
        up_intermediaries: Optional[jnp.ndarray] = None,
        up_mask: Optional[jnp.ndarray] = None,
        down_senders: Optional[jnp.ndarray] = None,
        down_receivers: Optional[jnp.ndarray] = None,
        down_intermediaries: Optional[jnp.ndarray] = None,
        down_mask: Optional[jnp.ndarray] = None,
        # Cross-dimension adjacencies
        coboundary_senders: Optional[jnp.ndarray] = None,
        coboundary_receivers: Optional[jnp.ndarray] = None,
        coboundary_mask: Optional[jnp.ndarray] = None,
        boundary_senders: Optional[jnp.ndarray] = None,
        boundary_receivers: Optional[jnp.ndarray] = None,
        boundary_mask: Optional[jnp.ndarray] = None,
        **kwargs
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass. Override this method in subclasses.
        
        Adjacency conventions:
        - up_adj: cells i,j in same dim connected via shared coboundary k
            up_senders[e] -> up_receivers[e] via up_intermediaries[e]
        - down_adj: cells i,j in same dim connected via shared boundary k
            down_senders[e] -> down_receivers[e] via down_intermediaries[e]
        - boundary_adj: cell i receives from boundary cell j
            boundary_senders[e] (lower dim) -> boundary_receivers[e] (current dim)
        - coboundary_adj: cell i receives from coboundary cell j
            coboundary_senders[e] (higher dim) -> coboundary_receivers[e] (current dim)
        
        Returns:
            Dict with updated features. Keys can include 'x', 'boundary_x', 'coboundary_x'.
        """
        raise NotImplementedError("Subclasses must implement __call__")


class EquivariantMessageLayer(BaseCochainTensorProduct):
    """Unified equivariant message passing layer.

    Implements the common pattern shared by all message layers:

        1. Gather sender/receiver features at adjacency indices
        2. Compute rotation-invariant features (scalars, norms, cross-alignments)
        3. Feature gate = MLP(invariants) → sigmoid
        4. Geometric gate = MLP(basis(geo_quantity)) → sigmoid  (optional)
        5. Map sender features to receiver dimensions (channel-wise linear)
        6. message = expand(feature_gate * geo_gate) * mapped_sender
        7. Aggregate messages to receivers

    Invariant inputs to the feature gate MLP:
        - Sender and receiver L=0 scalars
        - Sender and receiver L=1 norms (if present)
        - Compressed sender/receiver L=2 norms and alignment (if present)
        - Intermediary scalars, L=1 norms, L=2 norms (if provided)

    Args:
        sender_irreps: Irreps of the sending cochain.
        receiver_irreps: Irreps of the receiving cochain (determines output).
        hidden_dim: Hidden dimension for the feature gate MLP.
        hidden_l1_channels: Channels for L=1 alignment compression.
        hidden_l2_channels: Channels for L=2 alignment compression.
        geo_basis_type: ``'rbf_bessel'``, ``'legendre'``, or ``None``.
        geo_basis_dim: Number of basis functions for the geometric gate.
        geo_filter_dim: Hidden dimension for the geometric gate MLP.
        cutoff: Distance cutoff (only used when ``geo_basis_type='rbf_bessel'``).
        intermediary_irreps: Optional irreps of an intermediary cell whose
            invariants are appended to the feature gate input.
        aggr: Aggregation type (``'add'``, ``'mean'``, ``'max'``).
        rngs: Flax random number generator.
    """

    def __init__(
        self,
        sender_irreps: str,
        receiver_irreps: str,
        hidden_dim: int = 32,
        hidden_l1_channels: int = 4,
        hidden_l2_channels: int = 4,
        geo_basis_type: Optional[str] = None,
        geo_basis_dim: int = 16,
        geo_filter_dim: int = 32,
        cutoff: float = 5.0,
        intermediary_irreps: Optional[str] = None,
        aggr: str = "add",
        use_tensor_products: bool = False,
        hidden_tp_dim: int = 8,
        rngs: nnx.Rngs = None,
    ):
        super().__init__(irreps_in=receiver_irreps, aggr=aggr, rngs=rngs)

        self.hidden_dim = hidden_dim
        self.hidden_l1_channels = hidden_l1_channels
        self.hidden_l2_channels = hidden_l2_channels
        self.geo_basis_type = geo_basis_type
        self.geo_basis_dim = geo_basis_dim
        self.cutoff = cutoff
        self.has_intermediary = intermediary_irreps is not None

        # Register sender / receiver spaces
        sender_obj = self.e3nn.Irreps(sender_irreps)
        receiver_obj = self.e3nn.Irreps(receiver_irreps)
        self.sender_info = self.register_space('sender', sender_obj)
        self.receiver_info = self.register_space('receiver', receiver_obj)

        if self.has_intermediary:
            inter_obj = self.e3nn.Irreps(intermediary_irreps)
            self.inter_info = self.register_space('intermediary', inter_obj)

        # --- W1: map sender features to receiver dimensions ----------------
        self.W1_l0 = None
        if self.sender_info.num_l0 > 0 and self.receiver_info.num_l0 > 0:
            if self.sender_info.num_l0 != self.receiver_info.num_l0:
                self.W1_l0 = nnx.Linear(
                    self.sender_info.num_l0, self.receiver_info.num_l0, rngs=self.rngs)

        self.W1_l1 = None
        if self.sender_info.num_l1 > 0 and self.receiver_info.num_l1 > 0:
            if self.sender_info.num_l1 != self.receiver_info.num_l1:
                self.W1_l1 = nnx.Linear(
                    self.sender_info.num_l1, self.receiver_info.num_l1, rngs=self.rngs)

        self.W1_l2 = None
        if self.sender_info.num_l2 > 0 and self.receiver_info.num_l2 > 0:
            if self.sender_info.num_l2 != self.receiver_info.num_l2:
                self.W1_l2 = nnx.Linear(
                    self.sender_info.num_l2, self.receiver_info.num_l2, rngs=self.rngs)

        # --- L=1 compression for cross-alignment --------------------------
        self.U_l1 = None
        self.V_l1 = None
        if self.sender_info.num_l1 > 0 and self.receiver_info.num_l1 > 0:
            self.U_l1 = nnx.Linear(
                self.sender_info.num_l1, hidden_l1_channels, rngs=self.rngs)
            self.V_l1 = nnx.Linear(
                self.receiver_info.num_l1, hidden_l1_channels, rngs=self.rngs)

        # --- L=2 compression for cross-alignment --------------------------
        self.U_l2 = None
        self.V_l2 = None
        if self.sender_info.num_l2 > 0 and self.receiver_info.num_l2 > 0:
            self.U_l2 = nnx.Linear(
                self.sender_info.num_l2, hidden_l2_channels, rngs=self.rngs)
            self.V_l2 = nnx.Linear(
                self.receiver_info.num_l2, hidden_l2_channels, rngs=self.rngs)

        # --- Feature gate MLP ---------------------------------------------
        feat_dim = self._feature_gate_dim()
        self.feature_gate_mlp = nnx.Sequential(
            nnx.Linear(feat_dim, hidden_dim, rngs=self.rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=self.rngs),
            nnx.silu,
            nnx.Linear(hidden_dim, self.receiver_info.num_instances, rngs=self.rngs),
            nnx.sigmoid,
        )

        # --- Geometric gate MLP (optional) ---------------------------------
        self.geo_gate_mlp = None
        if geo_basis_type is not None:
            self.geo_gate_mlp = nnx.Sequential(
                nnx.Linear(geo_basis_dim, geo_filter_dim, rngs=self.rngs),
                nnx.silu,
                nnx.Linear(
                    geo_filter_dim, self.receiver_info.num_instances, rngs=self.rngs),
                nnx.sigmoid,
            )

        # --- Tensor product layers (optional) ---------------------------------
        # Within-sender mixing: L=2(s) × L=1(s) → new L=1 message;
        #                       L=1(s) × L=1(s) → new L=2 message.
        # When enabled, these REPLACE the raw L>0 mapped features (W1_l1/W1_l2).
        # Follows U/V convention: V projects sender, W projects output.
        self.use_tensor_products = use_tensor_products
        self.hidden_tp_dim = hidden_tp_dim

        # L=2(sender) × L=1(sender) → new L=1 message
        self.tp_l2l1_enabled = (
            use_tensor_products
            and self.sender_info.num_l2 > 0
            and self.sender_info.num_l1 > 0
            and self.receiver_info.num_l1 > 0
        )
        if self.tp_l2l1_enabled:
            # Project sender L=2 and L=1 to shared hidden_tp_dim, then output
            self.TP_V_l2 = nnx.Linear(
                self.sender_info.num_l2, hidden_tp_dim, rngs=self.rngs)
            self.TP_V_l1 = nnx.Linear(
                self.sender_info.num_l1, hidden_tp_dim, rngs=self.rngs)
            self.TP_W_l2l1 = nnx.Linear(
                hidden_tp_dim, self.receiver_info.num_l1, rngs=self.rngs)

        # L=1(sender) × L=1(sender) → new L=2 message
        self.tp_l1l1_enabled = (
            use_tensor_products
            and self.sender_info.num_l1 > 0
            and self.receiver_info.num_l2 > 0
        )
        if self.tp_l1l1_enabled:
            # Two independent projections to hidden_tp_dim, then outer product
            self.TP_V_l1_u = nnx.Linear(
                self.sender_info.num_l1, hidden_tp_dim, rngs=self.rngs)
            self.TP_V_l1_v = nnx.Linear(
                self.sender_info.num_l1, hidden_tp_dim, rngs=self.rngs)
            self.TP_W_l1l1 = nnx.Linear(
                hidden_tp_dim, self.receiver_info.num_l2, rngs=self.rngs)

        self.output_irreps = receiver_obj
        self.output_dim = receiver_obj.dim

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _feature_gate_dim(self) -> int:
        dim = self.sender_info.num_l0 + self.receiver_info.num_l0
        if self.sender_info.num_l1 > 0 and self.receiver_info.num_l1 > 0:
            dim += 3 * self.hidden_l1_channels  # U norms, V norms, cosine sim
        else:
            if self.sender_info.num_l1 > 0:
                dim += self.sender_info.num_l1
            if self.receiver_info.num_l1 > 0:
                dim += self.receiver_info.num_l1
        if self.sender_info.num_l2 > 0 and self.receiver_info.num_l2 > 0:
            dim += 3 * self.hidden_l2_channels  # U norms, V norms, alignment
        if self.has_intermediary:
            dim += self.inter_info.num_l0
            if self.inter_info.num_l1 > 0:
                dim += self.inter_info.num_l1
            if self.inter_info.num_l2 > 0:
                dim += self.inter_info.num_l2
        return dim

    def _compute_invariants(
        self,
        s: jnp.ndarray,
        r: jnp.ndarray,
        inter: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Build the invariant vector fed to the feature gate MLP."""
        parts: List[jnp.ndarray] = []

        # Sender / receiver scalars
        parts.append(extract_by_indices(s, self.sender_info.l0_indices))
        parts.append(extract_by_indices(r, self.receiver_info.l0_indices))

        # L=1 compressed norms + cross-alignment (when both sides have L=1)
        if self.sender_info.num_l1 > 0 and self.receiver_info.num_l1 > 0:
            s_l1 = extract_by_indices(s, self.sender_info.l1_indices)
            r_l1 = extract_by_indices(r, self.receiver_info.l1_indices)
            U_s = apply_channel_wise_linear_l1(
                self.U_l1, s_l1, self.sender_info.num_l1)
            V_r = apply_channel_wise_linear_l1(
                self.V_l1, r_l1, self.receiver_info.num_l1)
            parts.append(compute_l1_norms(U_s, self.hidden_l1_channels))
            parts.append(compute_l1_norms(V_r, self.hidden_l1_channels))
            parts.append(
                compute_l1_cosine_similarity(U_s, V_r, self.hidden_l1_channels))
        else:
            # Fallback: raw norms when only one side has L=1
            if self.sender_info.num_l1 > 0:
                s_l1 = extract_by_indices(s, self.sender_info.l1_indices)
                parts.append(compute_l1_norms(s_l1, self.sender_info.num_l1))
            if self.receiver_info.num_l1 > 0:
                r_l1 = extract_by_indices(r, self.receiver_info.l1_indices)
                parts.append(compute_l1_norms(r_l1, self.receiver_info.num_l1))

        # L=2 compressed norms + cross-alignment
        if self.sender_info.num_l2 > 0 and self.receiver_info.num_l2 > 0:
            s_l2 = extract_by_indices(s, self.sender_info.l2_indices)
            r_l2 = extract_by_indices(r, self.receiver_info.l2_indices)
            U_s = apply_channel_wise_linear_l2(
                self.U_l2, s_l2, self.sender_info.num_l2)
            V_r = apply_channel_wise_linear_l2(
                self.V_l2, r_l2, self.receiver_info.num_l2)
            parts.append(compute_l2_norms(U_s, self.hidden_l2_channels))
            parts.append(compute_l2_norms(V_r, self.hidden_l2_channels))
            parts.append(
                compute_tensor_alignment(U_s, V_r, self.hidden_l2_channels))

        # Intermediary invariants
        if inter is not None and self.has_intermediary:
            parts.append(extract_by_indices(inter, self.inter_info.l0_indices))
            if self.inter_info.num_l1 > 0:
                i_l1 = extract_by_indices(inter, self.inter_info.l1_indices)
                parts.append(compute_l1_norms(i_l1, self.inter_info.num_l1))
            if self.inter_info.num_l2 > 0:
                i_l2 = extract_by_indices(inter, self.inter_info.l2_indices)
                parts.append(compute_l2_norms(i_l2, self.inter_info.num_l2))

        return jnp.concatenate(parts, axis=-1)

    def _compute_geo_basis(self, geo_quantity: jnp.ndarray) -> jnp.ndarray:
        """Expand the geometric quantity into a basis."""
        if self.geo_basis_type == 'rbf_bessel':
            rbf = compute_radial_basis_bessel(
                geo_quantity, self.geo_basis_dim, self.cutoff)
            cutoff_env = compute_smooth_cutoff(geo_quantity, self.cutoff)
            return rbf * cutoff_env[:, None]
        elif self.geo_basis_type == 'legendre':
            return compute_legendre_basis(geo_quantity, self.geo_basis_dim)
        raise ValueError(f"Unknown geo_basis_type: {self.geo_basis_type}")

    def _map_gate_aggregate(
        self,
        sender_gathered: jnp.ndarray,
        gate: jnp.ndarray,
        senders: Optional[jnp.ndarray], #Used in GeometryReminder
        receivers: jnp.ndarray,
        num_receivers: int,
        mask: Optional[jnp.ndarray],
        static: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """Map sender to receiver dims, apply gate, aggregate."""
        num_msg = sender_gathered.shape[0]
        mapped = jnp.zeros((num_msg, self.receiver_info.total_dim))

        # L=0
        if self.sender_info.num_l0 > 0 and self.receiver_info.num_l0 > 0:
            s_l0 = extract_by_indices(sender_gathered, self.sender_info.l0_indices)
            if self.W1_l0 is not None:
                s_l0 = self.W1_l0(s_l0)
            mapped = mapped.at[:, self.receiver_info.l0_indices].set(s_l0)

        # L=1 (channel-wise, or TP-mixed when enabled)
        if self.receiver_info.num_l1 > 0:
            if self.tp_l2l1_enabled:
                # Replace raw L=1 with: contract(W_l2(s_l2), W_l1(s_l1)) → new L=1
                s_l2_raw = extract_by_indices(sender_gathered, self.sender_info.l2_indices)
                s_l1_raw = extract_by_indices(sender_gathered, self.sender_info.l1_indices)
                s_l2_proj = apply_channel_wise_linear_l2(
                    self.TP_V_l2, s_l2_raw, self.sender_info.num_l2)
                s_l1_proj = apply_channel_wise_linear_l1(
                    self.TP_V_l1, s_l1_raw, self.sender_info.num_l1)
                # Reshape to (num_msg, hidden_tp_dim, 5/3), contract, flatten
                num_msg = sender_gathered.shape[0]
                T_proj = s_l2_proj.reshape(num_msg, self.hidden_tp_dim, 5)
                v_proj = s_l1_proj.reshape(num_msg, self.hidden_tp_dim, 3)
                tp_l1 = contract_l2_l1(T_proj, v_proj)
                tp_l1 = apply_channel_wise_linear_l1(
                    self.TP_W_l2l1, tp_l1.reshape(num_msg, self.hidden_tp_dim * 3),
                    self.hidden_tp_dim)
                mapped = mapped.at[:, self.receiver_info.l1_indices].set(tp_l1)
            elif self.sender_info.num_l1 > 0:
                s_l1 = extract_by_indices(sender_gathered, self.sender_info.l1_indices)
                if self.W1_l1 is not None:
                    s_l1 = apply_channel_wise_linear_l1(
                        self.W1_l1, s_l1, self.sender_info.num_l1)
                mapped = mapped.at[:, self.receiver_info.l1_indices].set(s_l1)

        # L=2 (channel-wise, or TP-mixed when enabled)
        if self.receiver_info.num_l2 > 0:
            if self.tp_l1l1_enabled:
                # Replace raw L=2 with: outer(W_u(s_l1), W_v(s_l1)) → new L=2
                num_msg = sender_gathered.shape[0]
                s_l1_raw = extract_by_indices(sender_gathered, self.sender_info.l1_indices)
                u_proj = apply_channel_wise_linear_l1(
                    self.TP_V_l1_u, s_l1_raw, self.sender_info.num_l1)
                v_proj2 = apply_channel_wise_linear_l1(
                    self.TP_V_l1_v, s_l1_raw, self.sender_info.num_l1)
                u_r = u_proj.reshape(num_msg, self.hidden_tp_dim, 3)
                v_r = v_proj2.reshape(num_msg, self.hidden_tp_dim, 3)
                tp_l2 = outer_l1_l1_to_l2(u_r, v_r)  # (num_msg, hidden_tp_dim, 5)
                tp_l2 = apply_channel_wise_linear_l2(
                    self.TP_W_l1l1, tp_l2.reshape(num_msg, self.hidden_tp_dim * 5),
                    self.hidden_tp_dim)
                mapped = mapped.at[:, self.receiver_info.l2_indices].set(tp_l2)
            elif self.sender_info.num_l2 > 0:
                s_l2 = extract_by_indices(sender_gathered, self.sender_info.l2_indices)
                if self.W1_l2 is not None:
                    s_l2 = apply_channel_wise_linear_l2(
                        self.W1_l2, s_l2, self.sender_info.num_l2)
                mapped = mapped.at[:, self.receiver_info.l2_indices].set(s_l2)

        # Expand gate: (num_msg, num_instances) → (num_msg, total_dim)
        expanded_gate = gate[:, self.receiver_info.gate_indices]
        messages = expanded_gate * mapped

        # Aggregate
        return self.aggregate(messages, receivers, num_receivers, mask=mask)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        sender_features: jnp.ndarray,
        receiver_features: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        num_receivers: int,
        mask: Optional[jnp.ndarray] = None,
        receiver_mask: Optional[jnp.ndarray] = None,
        geo_quantity: Optional[jnp.ndarray] = None,
        intermediary_features: Optional[jnp.ndarray] = None,
        intermediary_indices: Optional[jnp.ndarray] = None,
        static: Optional[Dict] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Compute gated messages and aggregate to receivers.

        Args:
            sender_features: All sender cell features ``(num_senders, sender_dim)``.
            receiver_features: All receiver cell features ``(num_receivers, receiver_dim)``.
            senders: Sender indices per message ``(num_messages,)``.
            receivers: Receiver indices per message ``(num_messages,)``.
            num_receivers: Total number of receiver cells.
            mask: Valid-message mask ``(num_messages,)``.
            receiver_mask: Valid-receiver mask ``(num_receivers,)``.
            geo_quantity: Per-message geometric scalar ``(num_messages,)``
                (distances for RBF, alignment scores for Legendre).
            intermediary_features: Features of intermediary cells.
            intermediary_indices: Intermediary index per message ``(num_messages,)``.

        Returns:
            ``{'x': aggregated_update}`` with shape ``(num_receivers, receiver_dim)``.
        """
        # Gather
        s = sender_features[senders]
        r = receiver_features[receivers]
        inter = None
        if intermediary_features is not None and intermediary_indices is not None:
            inter = intermediary_features[intermediary_indices]

        # Feature gate
        invariants = self._compute_invariants(s, r, inter)
        feature_gate = self.feature_gate_mlp(invariants)

        # Geometric gate (optional)
        if self.geo_basis_type is not None and geo_quantity is not None:
            geo_basis = self._compute_geo_basis(geo_quantity)
            geo_gate = self.geo_gate_mlp(geo_basis)
            combined_gate = feature_gate * geo_gate
        else:
            combined_gate = feature_gate

        # Map, gate, aggregate
        output = self._map_gate_aggregate(
            s, combined_gate, senders, receivers, num_receivers, mask, static = static)

        # Apply receiver mask
        if receiver_mask is not None:
            output = jnp.where(receiver_mask[:, None], output, 0.0)

        return {'x': output}


class BaseCochainLayer(nnx.Module):
    """
    Minimal base class for cochain message passing layers.
    
    Stores adjacency information between cells:
    - Same dimension: up_adj (via shared coboundary), down_adj (via shared boundary)
    - Neighboring dimensions: boundary_adj (to lower dim), coboundary_adj (to higher dim)
    
    Subclasses implement their own messaging and update logic by overriding __call__.
    This class provides utility methods for aggregation.
    
    Args:
        aggr (str): Default aggregation type ('add', 'mean', 'max'). Default: 'add'
    """
    
    def __init__(self, aggr: str = "add"):
        super().__init__()
        self.aggr = aggr
    
    def aggregate(
        self,
        messages: jnp.ndarray,
        index: jnp.ndarray,
        num_segments: int,
        aggr_type: Optional[str] = None,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Aggregate messages to target indices.
        
        Args:
            messages: Message features (num_messages, feature_dim)
            index: Target indices for each message (num_messages,)
            num_segments: Number of target elements
            aggr_type: Aggregation type ('add', 'mean', 'max'). Uses self.aggr if None.
            mask: Optional mask for valid messages (num_messages,)
            
        Returns:
            Aggregated features (num_segments, feature_dim)
        """
        aggr_type = aggr_type or self.aggr
        
        if mask is not None:
            if aggr_type in ("add", "mean"):
                messages = jnp.where(mask[:, None], messages, 0.0)
            elif aggr_type == "max":
                messages = jnp.where(mask[:, None], messages, -jnp.inf)
        
        if aggr_type == "add":
            return jax.ops.segment_sum(messages, index, num_segments)
        
        elif aggr_type == "mean":
            if mask is not None:
                counts = jax.ops.segment_sum(mask.astype(jnp.float32), index, num_segments)
            else:
                counts = jax.ops.segment_sum(
                    jnp.ones(messages.shape[0], dtype=messages.dtype), index, num_segments
                )
            sums = jax.ops.segment_sum(messages, index, num_segments)
            return jnp.where(counts[:, None] > 0, sums / counts[:, None], 0.0)
        
        elif aggr_type == "max":
            return jax.ops.segment_max(messages, index, num_segments)
        
        else:
            raise ValueError(f"Unknown aggregation type: {aggr_type}. Use 'add', 'mean', or 'max'.")
    
    def __call__(
        self,
        x: jnp.ndarray,
        x_mask: jnp.ndarray,
        boundary_x: Optional[jnp.ndarray] = None,
        boundary_x_mask: Optional[jnp.ndarray] = None,
        coboundary_x: Optional[jnp.ndarray] = None,
        coboundary_x_mask: Optional[jnp.ndarray] = None,
        # Same-dimension adjacencies
        up_senders: Optional[jnp.ndarray] = None,
        up_receivers: Optional[jnp.ndarray] = None,
        up_intermediaries: Optional[jnp.ndarray] = None,
        up_mask: Optional[jnp.ndarray] = None,
        down_senders: Optional[jnp.ndarray] = None,
        down_receivers: Optional[jnp.ndarray] = None,
        down_intermediaries: Optional[jnp.ndarray] = None,
        down_mask: Optional[jnp.ndarray] = None,
        # Cross-dimension adjacencies
        coboundary_senders: Optional[jnp.ndarray] = None,
        coboundary_receivers: Optional[jnp.ndarray] = None,
        coboundary_mask: Optional[jnp.ndarray] = None,
        boundary_senders: Optional[jnp.ndarray] = None,
        boundary_receivers: Optional[jnp.ndarray] = None,
        boundary_mask: Optional[jnp.ndarray] = None,
        **kwargs
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass. Override this method in subclasses.
        
        Adjacency conventions:
        - up_adj: cells i,j in same dim connected via shared coboundary k
            up_senders[e] -> up_receivers[e] via up_intermediaries[e]
        - down_adj: cells i,j in same dim connected via shared boundary k
            down_senders[e] -> down_receivers[e] via down_intermediaries[e]
        - boundary_adj: cell i receives from boundary cell j
            boundary_senders[e] (lower dim) -> boundary_receivers[e] (current dim)
        - coboundary_adj: cell i receives from coboundary cell j
            coboundary_senders[e] (higher dim) -> coboundary_receivers[e] (current dim)
        
        Returns:
            Dict with updated features. Keys can include 'x', 'boundary_x', 'coboundary_x'.
        """
        raise NotImplementedError("Subclasses must implement __call__")