from typing import Optional, List, Dict, Any, Tuple

import jax 
import flax
import jax.numpy as jnp
import numpy as np
import pickle
import time
from rdkit import Chem
from rdkit.Chem import AllChem

# Molecular-level target properties
MOLECULAR_PROPERTIES = ['mu', 'alpha', 'gap', 'r2']



@flax.struct.dataclass #Ensure immutability, built-in __init__ method
class Cochain:
    """
    Class representing a cochain on k-dim cells (i.e. vector-valued signals on k-dim cells).

    Args:
        dim: dim of the cells in the cochain
        num_cells: size of the cochain
        x: learned feature matrix, shape [num_cells, num_features]; evolves layer-by-layer
        static: dictionary of static features that remain constant through layers.
                Useful for geometric/physical information used in gating or filtering.
                Examples:
                - For nodes (dim=0): {'Z': atomic_numbers, 'pos': positions}
                - For edges (dim=1): {'r_ij': relative_vectors, 'r_ij_traceless': traceless_repr}
                - For faces (dim=2): {'normal': face_normals, 'area': face_areas}
                Each value should have shape [num_cells, ...] where first dim matches num_cells.
        num_cells_up: size of the (k+1)-dim cochain
        num_cells_down: size of the (k-1)-dim cochain
        up_senders: indices of sender cells of the cochain through coboundary cells
        up_receivers: indices of receiver cells of the cochain through coboundary cells
        up_intermediaries: indices of coboundary cells connecting up_senders/receivers
        down_senders: indices of sender cells of the cochain through boundary cells
        down_receivers: indices of receiver cells of the cochain through boundary cells
        down_intermediaries: indices of boundary cells connecting down_senders/receivers
        boundary_senders: indices of boundary cells sending messages to this cochain
        boundary_receivers: indices of cells in this cochain receiving boundary messages
        coboundary_senders: indices of coboundary cells sending messages to this cochain
        coboundary_receivers: indices of cells in this cochain receiving coboundary messages
        y: labels over cells in the cochain, shape [num_cells,]
    """

    dim:                    int
    num_cells:              int
    x:                      jnp.ndarray                     = None
    static:                 Optional[Dict[str, jnp.ndarray]] = None
    num_cells_up:           Optional[int]                   = None
    num_cells_down:         Optional[int]                   = None
    up_senders:             jnp.ndarray                     = None
    up_receivers:           jnp.ndarray                     = None
    up_intermediaries:      jnp.ndarray                     = None
    down_senders:           jnp.ndarray                     = None
    down_receivers:         jnp.ndarray                     = None
    down_intermediaries:    jnp.ndarray                     = None
    boundary_senders:       jnp.ndarray                     = None
    boundary_receivers:     jnp.ndarray                     = None
    coboundary_senders:     jnp.ndarray                     = None
    coboundary_receivers:   jnp.ndarray                     = None
    y:                      jnp.ndarray                     = None

    #Method for sanity checks after creation
    def __post_init__(self):

        # Helper function to check if a value is integer-like
        def is_integer_like(val):
            return isinstance(val, (int, jnp.integer)) or (hasattr(val, '__index__') and hasattr(val, '__int__'))
        
        if self.dim == 0:
            # dim-0 cochains may have down connectivity (e.g. via bags of bonds)
            if self.down_senders is not None or self.down_receivers is not None:
                assert is_integer_like(self.num_cells_down) and self.num_cells_down >= 0, \
                    "dim-0 cochain with down connectivity requires valid num_cells_down"
                if self.down_intermediaries is not None and self.down_intermediaries.size > 0:
                    assert jnp.max(self.down_intermediaries) <= self.num_cells_down - 1

        else:

            # Needed checks to ensure proper offsets when batching
            assert is_integer_like(self.num_cells_down) and self.num_cells_down >= 0

            if self.num_cells > 0:

                # Only check bounds if down_intermediaries is not empty
                if self.down_intermediaries is not None:
                    if self.down_intermediaries.size > 0:
                        assert jnp.max(self.down_intermediaries) <= self.num_cells_down-1

            else:

                # For empty cochains, connectivity arrays must be None or empty arrays
                for arr_name in ['down_senders', 'down_receivers', 'down_intermediaries', 
                               'up_senders', 'up_receivers', 'up_intermediaries',
                               'boundary_senders', 'boundary_receivers',
                               'coboundary_senders', 'coboundary_receivers']:
                    
                    arr = getattr(self, arr_name)
                    if arr is not None:
                        assert isinstance(arr, (np.ndarray, jnp.ndarray)), f"{arr_name} must be None or ndarray"
                        # Empty cochains can have empty arrays
                        assert arr.size == 0, f"Empty cochain {arr_name} must be None or empty array"

        
        
        assert is_integer_like(self.num_cells) and self.num_cells >= 0, "num_cells must be a non-negative integer"

        # Check num_cells_up if present
        if self.num_cells_up is not None:
            assert is_integer_like(self.num_cells_up) and self.num_cells_up >= 0, "num_cells_up must be a non-negative integer"
        
        # Check index arrays are within valid range
        for arr_name, arr, maxval in [
            ("up_senders",      self.up_senders,        self.num_cells),
            ("up_receivers",    self.up_receivers,      self.num_cells),
            ("down_senders",    self.down_senders,      self.num_cells),
            ("down_receivers",  self.down_receivers,    self.num_cells),
        ]:
            
            if arr is not None and arr.size > 0:
                assert jnp.all((arr >= 0) & (arr < maxval)), f"{arr_name} contains out-of-bounds indices"

        # Check consistency between up_* arrays and up_intermediaries
        if self.up_senders is not None or self.up_receivers is not None:
            assert self.up_intermediaries is not None, "If up_senders or up_receivers are not None, up_intermediaries must also not be None"
        
        # Check consistency between down_* arrays and down_intermediaries
        if self.down_senders is not None or self.down_receivers is not None:
            assert self.down_intermediaries is not None, "If down_senders or down_receivers are not None, down_intermediaries must also not be None"

        # Check adjacency triples have matching lengths
        for label, s, r, i in [
            ("up",   self.up_senders,   self.up_receivers,   self.up_intermediaries),
            ("down", self.down_senders, self.down_receivers, self.down_intermediaries),
        ]:
            arrs = [a for a in (s, r, i) if a is not None]
            if arrs:
                lens = [a.shape[0] for a in arrs]
                assert len(set(lens)) == 1, (
                    f"{label}_senders/receivers/intermediaries length mismatch: {lens}"
                )

        # Check boundary/coboundary pairs have matching lengths
        for label, s, r in [
            ("boundary",   self.boundary_senders,   self.boundary_receivers),
            ("coboundary", self.coboundary_senders, self.coboundary_receivers),
        ]:
            if s is not None and r is not None:
                assert s.shape[0] == r.shape[0], (
                    f"{label}_senders/receivers length mismatch: "
                    f"{s.shape[0]} != {r.shape[0]}"
                )

        if self.up_intermediaries is not None and self.up_intermediaries.size > 0 and self.num_cells_up is not None:
            assert jnp.all((self.up_intermediaries >= 0) & (self.up_intermediaries < self.num_cells_up)), "up_intermediaries out of bounds"
        
        if self.down_intermediaries is not None and self.down_intermediaries.size > 0 and self.num_cells_down is not None:
            assert jnp.all((self.down_intermediaries >= 0) & (self.down_intermediaries < self.num_cells_down)), "down_intermediaries out of bounds"

        # Check consistency for new boundary/coboundary message arrays
        if self.boundary_senders is not None or self.boundary_receivers is not None:
            assert self.boundary_senders is not None and self.boundary_receivers is not None, "boundary_senders and boundary_receivers must both be None or both not None"
            if self.boundary_senders.size > 0:
                assert self.num_cells_down is not None and self.num_cells_down > 0, "boundary messages require num_cells_down > 0"
                assert jnp.all((self.boundary_senders >= 0) & (self.boundary_senders < self.num_cells_down)), "boundary_senders out of bounds"
                assert jnp.all((self.boundary_receivers >= 0) & (self.boundary_receivers < self.num_cells)), "boundary_receivers out of bounds"

        if self.coboundary_senders is not None or self.coboundary_receivers is not None:
            assert self.coboundary_senders is not None and self.coboundary_receivers is not None, "coboundary_senders and coboundary_receivers must both be None or both not None"
            if self.coboundary_senders.size > 0:
                assert self.num_cells_up is not None and self.num_cells_up > 0, "coboundary messages require num_cells_up > 0"
                assert jnp.all((self.coboundary_senders >= 0) & (self.coboundary_senders < self.num_cells_up)), "coboundary_senders out of bounds"
                assert jnp.all((self.coboundary_receivers >= 0) & (self.coboundary_receivers < self.num_cells)), "coboundary_receivers out of bounds"

        # Check feature dimensions
        if self.x is not None:
            assert self.x.shape[0] == self.num_cells, "x.shape[0] != num_cells"
        if self.y is not None:
            assert self.y.shape[0] == self.num_cells, "y.shape[0] != num_cells"
        
        # Check static features dimensions
        if self.static is not None:
            assert isinstance(self.static, dict), "static must be a dictionary"
            for key, val in self.static.items():
                assert isinstance(val, (np.ndarray, jnp.ndarray)), f"static['{key}'] must be an ndarray"
                assert val.shape[0] == self.num_cells, f"static['{key}'].shape[0] ({val.shape[0]}) != num_cells ({self.num_cells})"
    
    # ==================== Static Feature Access Helpers ====================
    
    def get_static(self, key: str, default: Any = None) -> Optional[jnp.ndarray]:
        """
        Safely get a static feature by key.
        
        Args:
            key: Name of the static feature (e.g., 'Z', 'pos', 'r_ij')
            default: Value to return if key is not found
            
        Returns:
            The static feature array, or default if not found
        """
        if self.static is None:
            return default
        return self.static.get(key, default)
    
    def has_static(self, key: str) -> bool:
        """Check if a static feature exists."""
        return self.static is not None and key in self.static

    @staticmethod
    def empty_cochain(dim, num_cells_down=None, num_cells_up=None, use_empty_arrays=False):
        """Create an empty cochain with 0 cells.
        
        Args:
            dim: Dimension of the cochain
            num_cells_down: Number of cells in the (k-1)-dimension
            num_cells_up: Number of cells in the (k+1)-dimension  
            use_empty_arrays: If True, use empty jnp.arrays instead of None for connectivity
        """
        if use_empty_arrays:
            # Use empty arrays instead of None
            empty_int_array = jnp.array([], dtype=jnp.int32)
            return Cochain(
                dim                     = dim,
                num_cells               = 0,
                x                       = None,  # Features always None for empty cochain
                num_cells_up            = num_cells_up,
                num_cells_down          = num_cells_down,
                up_senders              = empty_int_array,
                up_receivers            = empty_int_array,
                up_intermediaries       = empty_int_array,
                down_senders            = empty_int_array,
                down_receivers          = empty_int_array,
                down_intermediaries     = empty_int_array,
                boundary_senders        = empty_int_array,
                boundary_receivers      = empty_int_array,
                coboundary_senders      = empty_int_array,
                coboundary_receivers    = empty_int_array,
                y                       = None
            )
        else:
            # Use None (default behavior)
            return Cochain(
                dim                     = dim,
                num_cells               = 0,
                x                       = None,
                num_cells_up            = num_cells_up,
                num_cells_down          = num_cells_down,
                up_senders              = None,
                up_receivers            = None,
                up_intermediaries       = None,
                down_senders            = None,
                down_receivers          = None,
                down_intermediaries     = None,
                boundary_senders        = None,
                boundary_receivers      = None,
                coboundary_senders      = None,
                coboundary_receivers    = None,
                y                       = None
            )


@flax.struct.dataclass
class Complex:
    """Class representing a cochain complex or an attributed cellular complex.

    Args:
        cochains: A list of cochains forming the cochain complex
        y: A tensor of shape (1,) containing a label for the complex for complex-level tasks.
        dimension: The dimension of the complex.
    """
    cochains:   List[Cochain]
    dimension:  Optional[int]
    y:          jnp.ndarray = None

    def __post_init__(self):

        if len(self.cochains) == 0:
            raise ValueError('At least one cochain is required.')

        # Check all cochains are Cochain instances
        for c in self.cochains:
            assert isinstance(c, Cochain), "All elements of cochains must be Cochain instances"

        # Check y shape if present
        if self.y is not None:
            assert self.y.shape[0] == 1 or len(self.y.shape) == 1, "Complex label y should be shape (1,) or 1D"

        if self.dimension is None:
            self.dimension = len(self.cochains) - 1

        elif len(self.cochains) < self.dimension + 1:
            raise ValueError(f'Not enough cochains passed, '
                            f'expected {self.dimension + 1}, received {len(self.cochains)}')

        for dim in range(self.dimension+1):

            cochain = self.cochains[dim]
            assert cochain.dim == dim
            
            if dim < self.dimension:

                upper_cochain = self.cochains[dim + 1]
                num_cells_up = upper_cochain.num_cells
                assert num_cells_up is not None
                
                if cochain.num_cells_up is not None:
                    assert cochain.num_cells_up == num_cells_up, f"Cochain {dim}: num_cells_up mismatch {cochain.num_cells_up} != {num_cells_up}"
                # If num_cells_up is None, that's also valid (no connectivity to upper dimension)
            
            if dim > 0:
                lower_cochain = self.cochains[dim - 1]
                num_cells_down = lower_cochain.num_cells
                assert num_cells_down is not None
                
                if cochain.num_cells_down is not None:
                    assert cochain.num_cells_down == num_cells_down, f"Cochain {dim}: num_cells_down mismatch {cochain.num_cells_down} != {num_cells_down}"
                # If num_cells_down is None, that's also valid (no connectivity to lower dimension)

        # ------------------------------------------------------------------
        # Cross-cochain boundary / coboundary duality
        # ------------------------------------------------------------------
        # If cochain A has coboundary pointing into cochain B (A.num_cells_up
        # == B.num_cells) and B has boundary pointing into A (B.num_cells_down
        # == A.num_cells), the incidence pairs must agree as sorted multisets.
        for i in range(len(self.cochains)):
            ci = self.cochains[i]
            # skip if cochain i has no coboundary
            if ci.coboundary_senders is None or ci.coboundary_senders.size == 0:
                continue
            # find the cochain j that ci's coboundary points to
            for j in range(len(self.cochains)):
                if i == j:
                    continue
                cj = self.cochains[j]
                if ci.num_cells_up != cj.num_cells:
                    continue
                if cj.num_cells_down is None or cj.num_cells_down != ci.num_cells:
                    continue
                if cj.boundary_senders is None or cj.boundary_senders.size == 0:
                    continue
                # ci coboundary: (ci_cell, cj_cell) pairs
                cb_pairs = sorted(zip(
                    np.asarray(ci.coboundary_receivers).tolist(),
                    np.asarray(ci.coboundary_senders).tolist(),
                ))
                # cj boundary: (ci_cell, cj_cell) pairs
                bd_pairs = sorted(zip(
                    np.asarray(cj.boundary_senders).tolist(),
                    np.asarray(cj.boundary_receivers).tolist(),
                ))
                assert cb_pairs == bd_pairs, (
                    f"Boundary/coboundary duality violated between "
                    f"dim-{ci.dim} coboundary and dim-{cj.dim} boundary: "
                    f"{len(cb_pairs)} vs {len(bd_pairs)} pairs"
                )


    def extract_features_and_adjacencies(self):

        cochain_features = []
        cochain_static = []
        cochain_adjacencies = {}

        for d, cb in enumerate(self.cochains):
            cochain_features.append(cb.x)
            cochain_static.append(cb.static)
            cochain_adjacencies[str(d)] = {
                'up_senders':           cb.up_senders,          #same cochain as receivers
                'up_receivers':         cb.up_receivers,        #same cochain as senders
                'down_senders':         cb.down_senders,        #same cochain as receivers
                'down_receivers':       cb.down_receivers,      #same cochain as senders
                'down_intermediaries':  cb.down_intermediaries, #belong to the lower cochain
                'up_intermediaries':    cb.up_intermediaries,   #belong to the upper cochain
                'boundary_senders':     cb.boundary_senders,    #belong to the upper cochain
                'boundary_receivers':   cb.boundary_receivers,  #belong to the lower cochain
                'coboundary_senders':   cb.coboundary_senders,  #belong to the lower cochain
                'coboundary_receivers': cb.coboundary_receivers,#belong to the upper cochain
            }

        return cochain_features, cochain_static, cochain_adjacencies



@flax.struct.dataclass
class CochainBatch:
    """Class representing a batch of cochains of the same dimension."""
    
    dim:                    int
    num_cells:              jnp.ndarray   
    x:                      Optional[jnp.ndarray] = None    # shape: (total_num_cells, num_features)
    static:                 Optional[Dict[str, jnp.ndarray]] = None  # Batched static features
    owner_cochains:         Optional[jnp.ndarray] = None    # shape: (total_num_cells,), onwer_cochain[i] = index of cochain containing cell i
    num_cells_up:           Optional[jnp.ndarray] = None    
    num_cells_down:         Optional[jnp.ndarray] = None    
    up_senders:             Optional[jnp.ndarray] = None    
    up_receivers:           Optional[jnp.ndarray] = None    
    up_intermediaries:      Optional[jnp.ndarray] = None    
    down_senders:           Optional[jnp.ndarray] = None    
    down_receivers:         Optional[jnp.ndarray] = None    
    down_intermediaries:    Optional[jnp.ndarray] = None    
    boundary_senders:       Optional[jnp.ndarray] = None    
    boundary_receivers:     Optional[jnp.ndarray] = None    
    coboundary_senders:     Optional[jnp.ndarray] = None    
    coboundary_receivers:   Optional[jnp.ndarray] = None    
    y:                      Optional[jnp.ndarray] = None    # shape: (total_num_cells,)

    # Mask attributes for handling padding
    x_mask:                 Optional[jnp.ndarray] = None    # shape: (total_num_cells,) - True for real cells
    up_mask:                Optional[jnp.ndarray] = None    # shape: (length of up_intermediaries,) - True for real indices  
    down_mask:              Optional[jnp.ndarray] = None    # shape: (length of down_intermediaries,) - True for real indices
    boundary_mask:          Optional[jnp.ndarray] = None    # shape: (length of boundary_senders,) - True for real indices  
    coboundary_mask:        Optional[jnp.ndarray] = None    # shape: (length of coboundary_senders,) - True for real indices
    
    def __post_init__(self):
        # Note: These checks are skipped during JAX tracing to allow pytree operations
        # The assertions would fail during JIT because traced values don't support comparisons
        pass  # Validation disabled for JIT compatibility
    
    # ==================== Static Feature Access Helpers ====================
    
    def get_static(self, key: str, default: Any = None) -> Optional[jnp.ndarray]:
        """
        Safely get a static feature by key.
        
        Args:
            key: Name of the static feature (e.g., 'Z', 'pos', 'r_ij')
            default: Value to return if key is not found
            
        Returns:
            The static feature array, or default if not found
        """
        if self.static is None:
            return default
        return self.static.get(key, default)
    
    def has_static(self, key: str) -> bool:
        """Check if a static feature exists."""
        return self.static is not None and key in self.static


    @staticmethod
    def from_cochain_list(cochains: List['Cochain']) -> 'CochainBatch':
        """Batch a list of Cochain objects into a single CochainBatch.

        Uses numpy internally for all concatenation to avoid per-shape
        XLA compilation overhead, converting to jnp only for the final result.
        """
        dim = cochains[0].dim

        # --- numpy metadata arrays (avoids jnp dispatch overhead) ---
        _num_cells     = np.array([c.num_cells for c in cochains], dtype=np.int32)
        _num_cells_up  = np.array([c.num_cells_up  if c.num_cells_up  is not None else 0 for c in cochains], dtype=np.int32)
        _num_cells_down= np.array([c.num_cells_down if c.num_cells_down is not None else 0 for c in cochains], dtype=np.int32)

        _cell_offsets      = np.concatenate([[0], np.cumsum(_num_cells)[:-1]])
        _cell_up_offsets   = np.concatenate([[0], np.cumsum(_num_cells_up)[:-1]])
        _cell_down_offsets = np.concatenate([[0], np.cumsum(_num_cells_down)[:-1]])

        # --- Concatenate x and y with numpy ---
        x_list = [np.asarray(c.x) for c in cochains if c.x is not None]
        _x = np.concatenate(x_list, axis=0) if x_list else None

        y_list = [np.asarray(c.y) for c in cochains if c.y is not None]
        _y = np.concatenate(y_list, axis=0) if y_list else None

        # --- Concatenate static features with numpy ---
        _static = None
        if any(c.static is not None for c in cochains):
            ref_cochain = next(c for c in cochains if c.static is not None)
            _static = {}
            for key in ref_cochain.static.keys():
                arrays_to_concat = []
                for c in cochains:
                    if c.static is not None and key in c.static:
                        arrays_to_concat.append(np.asarray(c.static[key]))
                    elif c.num_cells > 0:
                        ref_arr = np.asarray(ref_cochain.static[key])
                        shape = (c.num_cells,) + ref_arr.shape[1:]
                        arrays_to_concat.append(np.zeros(shape, dtype=ref_arr.dtype))
                if arrays_to_concat:
                    _static[key] = np.concatenate(arrays_to_concat, axis=0)

        # --- Owner cochain array with numpy ---
        _owner = np.concatenate([
            np.full(c.num_cells, i, dtype=np.int32) for i, c in enumerate(cochains)
        ], axis=0)

        # --- Helper: shift and concatenate index arrays with numpy ---
        def shift_and_concat(arrs, offsets):
            shifted = []
            for arr, offset in zip(arrs, offsets):
                if arr is not None:
                    shifted.append(np.asarray(arr) + int(offset))
            return np.concatenate(shifted, axis=0) if shifted else None

        _up_s  = shift_and_concat([c.up_senders for c in cochains], _cell_offsets)
        _up_r  = shift_and_concat([c.up_receivers for c in cochains], _cell_offsets)
        _up_i  = shift_and_concat([c.up_intermediaries for c in cochains], _cell_up_offsets)

        _dn_s  = shift_and_concat([c.down_senders for c in cochains], _cell_offsets)
        _dn_r  = shift_and_concat([c.down_receivers for c in cochains], _cell_offsets)
        _dn_i  = shift_and_concat([c.down_intermediaries for c in cochains], _cell_down_offsets)

        _bd_s  = shift_and_concat([c.boundary_senders for c in cochains], _cell_down_offsets)
        _bd_r  = shift_and_concat([c.boundary_receivers for c in cochains], _cell_offsets)

        _cb_s  = shift_and_concat([c.coboundary_senders for c in cochains], _cell_up_offsets)
        _cb_r  = shift_and_concat([c.coboundary_receivers for c in cochains], _cell_offsets)

        # --- Mask lengths ---
        total_cells = int(np.sum(_num_cells))
        up_len  = _up_s.shape[0] if _up_s is not None else 0
        dn_len  = _dn_s.shape[0] if _dn_s is not None else 0
        bd_len  = _bd_s.shape[0] if _bd_s is not None else 0
        cb_len  = _cb_s.shape[0] if _cb_s is not None else 0

        # --- Convert all numpy results to jnp for the final CochainBatch ---
        _j = lambda a: jnp.asarray(a) if a is not None else None
        _jstatic = {k: jnp.asarray(v) for k, v in _static.items()} if _static is not None else None

        return CochainBatch(
            dim=dim,
            num_cells=jnp.asarray(_num_cells),
            x=_j(_x),
            static=_jstatic,
            owner_cochains=jnp.asarray(_owner),
            num_cells_up=jnp.asarray(_num_cells_up),
            num_cells_down=jnp.asarray(_num_cells_down),
            up_senders=_j(_up_s),
            up_receivers=_j(_up_r),
            up_intermediaries=_j(_up_i),
            down_senders=_j(_dn_s),
            down_receivers=_j(_dn_r),
            down_intermediaries=_j(_dn_i),
            boundary_senders=_j(_bd_s),
            boundary_receivers=_j(_bd_r),
            coboundary_senders=_j(_cb_s),
            coboundary_receivers=_j(_cb_r),
            y=_j(_y),
            x_mask=jnp.ones(total_cells, dtype=bool) if _x is not None else jnp.array([], dtype=bool),
            up_mask=jnp.ones(up_len, dtype=bool) if up_len > 0 else jnp.array([], dtype=bool),
            down_mask=jnp.ones(dn_len, dtype=bool) if dn_len > 0 else jnp.array([], dtype=bool),
            boundary_mask=jnp.ones(bd_len, dtype=bool) if bd_len > 0 else jnp.array([], dtype=bool),
            coboundary_mask=jnp.ones(cb_len, dtype=bool) if cb_len > 0 else jnp.array([], dtype=bool),
        )



@flax.struct.dataclass
class ComplexBatch:

    cochain_batches:    List[CochainBatch]  # One per dimension
    dimension:          int
    num_complexes:      int                 # Number of complexes in the batch (for segment_sum)
    x:                  Optional[jnp.ndarray] = None      # shape: (batch_size, d_cx) — e.g. SMILES-derived features
    y:                  Optional[jnp.ndarray] = None      # shape: (batch_size, ...)


    def __post_init__(self):
        # Note: Validation disabled for JIT compatibility
        # The assertions would fail during JIT because types change during tracing
        pass

    @staticmethod
    def from_complex_list(complexes: List['Complex']) -> 'ComplexBatch':
        if len(complexes) == 0:
            raise ValueError("No complexes provided for batching.")

        max_dim = max(c.dimension for c in complexes) #Infer dimension from complexes

        cochain_batches = []

        for dim in range(max_dim + 1):
            # Collect cochains for this dimension across all complexes
            cochains_dim = []
            for comp in complexes:
                if dim < len(comp.cochains):
                    cochains_dim.append(comp.cochains[dim])
                else:
                    # Create empty cochain with proper connectivity information
                    num_cells_down = comp.cochains[dim - 1].num_cells if dim > 0 else None
                    num_cells_up = comp.cochains[dim + 1].num_cells if dim + 1 < len(comp.cochains) else None
                    cochains_dim.append(Cochain.empty_cochain(dim, num_cells_down, num_cells_up))

            cochain_batches.append(CochainBatch.from_cochain_list(cochains_dim))

        y = None
        if all(comp.y is not None for comp in complexes):
            y = jnp.stack([comp.y for comp in complexes], axis=0)

        return ComplexBatch(
            cochain_batches = cochain_batches,
            dimension       = max_dim,
            num_complexes   = len(complexes),
            y               = y
        )


    def extract_features_and_adjacencies(self):
        cochain_features = []
        cochain_static = []
        cochain_adjacencies = {}
        for d, cb in enumerate(self.cochain_batches):
            cochain_features.append(cb.x)
            cochain_static.append(cb.static)
            adjacencies = {
                'up_senders':           cb.up_senders,
                'up_receivers':         cb.up_receivers,
                'up_intermediaries':    cb.up_intermediaries,
                'down_senders':         cb.down_senders,
                'down_receivers':       cb.down_receivers,
                'down_intermediaries':  cb.down_intermediaries,
                'boundary_senders':     cb.boundary_senders,
                'boundary_receivers':   cb.boundary_receivers,
                'coboundary_senders':   cb.coboundary_senders,
                'coboundary_receivers': cb.coboundary_receivers,
                'x_mask':               cb.x_mask,
                'up_mask':              cb.up_mask,
                'down_mask':            cb.down_mask,
                'boundary_mask':        cb.boundary_mask,
                'coboundary_mask':      cb.coboundary_mask,
                'owner_cochains':       cb.owner_cochains,
            }
            
            cochain_adjacencies[str(d)] = adjacencies
        return cochain_features, cochain_static, cochain_adjacencies




# =============================================================================
# Gyration Tensor Computation
# =============================================================================

# Frobenius weights for 5-component L=2 tensor: [xy, xz, yz, (xx-yy)/2, zz]
# 
# For a traceless symmetric 3x3 tensor with 5 components:
#   ||T||_F^2 = xx^2 + yy^2 + zz^2 + 2*xy^2 + 2*xz^2 + 2*yz^2
#
# With our representation: aniso = (xx-yy)/2, and tracelessness xx+yy+zz=0:
#   xx = aniso - zz/2
#   yy = -aniso - zz/2
#
# Substituting:
#   ||T||_F^2 = (aniso - zz/2)^2 + (-aniso - zz/2)^2 + zz^2 + 2*xy^2 + 2*xz^2 + 2*yz^2
#             = 2*aniso^2 + zz^2/2 + zz^2 + 2*xy^2 + 2*xz^2 + 2*yz^2  
#             = 2*xy^2 + 2*xz^2 + 2*yz^2 + 2*aniso^2 + 1.5*zz^2
#
# Therefore the correct weights are [2, 2, 2, 2, 1.5]
FROBENIUS_WEIGHTS_L2 = jnp.array([2.0, 2.0, 2.0, 2.0, 1.5])

# Small epsilon for numerical stability in divisions/sqrt
EPS = 1e-8


def compute_frobenius_norm(
    tensor: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute Frobenius norm of an L=2 tensor encoded in 5 components.

    The representation uses the ordering
    ``[xy, xz, yz, (xx-yy)/2, zz]`` and the fixed Frobenius weights
    ``[2,2,2,2,1.5]``.  The helper previously accepted a ``weighted`` flag
    which is now removed since we always apply the correct weights.

    Args:
        tensor: Array of shape ``(..., 5)`` storing the L=2 components.

    Returns:
        A jnp.ndarray of shape ``(...)`` containing the rotation‑invariant norm.
    """
    weighted_sq = FROBENIUS_WEIGHTS_L2 * tensor ** 2
    return jnp.sqrt(jnp.sum(weighted_sq, axis=-1) + EPS)
    


@jax.jit

def compute_gyration_tensor(r: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the gyration tensor (traceless symmetric outer product) from relative position.
    
    For two particles, the gyration tensor is proportional to the traceless part of r ⊗ r:
        G_ab = r_a * r_b - (1/3)|r|^2 * δ_ab
    
    This is symmetric (G_ab = G_ba) and traceless (G_xx + G_yy + G_zz = 0).
    It transforms as an L=2 irreducible representation under rotations.
    Importantly, G(-r) = G(r), so it's orientation-invariant.
    
    We store the 5 independent components in Cartesian form:
    - G_xy (off-diagonal)
    - G_xz (off-diagonal)
    - G_yz (off-diagonal)
    - (G_xx - G_yy) / 2 (diagonal anisotropy)
    - G_zz (axial component, determines trace since G_xx + G_yy = -G_zz)
    
    Args:
        r: Relative displacement vectors, shape ``(...,3)``.

    Returns:
        Tuple ``(G, G_norm)`` with ``G`` of shape ``(...,5)`` and
        ``G_norm`` of shape ``(...)``.
    """
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r_sq = jnp.sum(r**2, axis=-1)

    # Diagonal components (traceless)
    G_xx = x**2 - r_sq / 3
    G_yy = y**2 - r_sq / 3
    G_zz = z**2 - r_sq / 3

    # Off-diagonal components
    G_xy = x * y
    G_xz = x * z
    G_yz = y * z

    # 5 independent components
    comp_xy = G_xy
    comp_xz = G_xz
    comp_yz = G_yz
    comp_aniso = (G_xx - G_yy) / 2  # x-y anisotropy
    comp_axial = G_zz  # axial component

    G = jnp.stack([comp_xy, comp_xz, comp_yz, comp_aniso, comp_axial], axis=-1)
    G_norm = compute_frobenius_norm(G)
    return G, G_norm



@jax.jit
def compute_ring_gyration_tensor(positions: jnp.ndarray, ring_atoms: List[int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute gyration tensor for a ring from atom positions using JAX operations.

    The ring gyration tensor is simply the sum of individual gyration tensors
    computed from displacements of ring atoms relative to the ring centroid.
    We leverage `compute_gyration_tensor` for the per-atom calculation,
    ensuring the same ordering and Frobenius weighting.

    Args:
        positions: Array of shape (n_atoms, 3) with all atom positions
        ring_atoms: List of atom indices forming the ring

    Returns:
        Tuple of (G_total, G_norm) where:
        - G_total: JAX array of shape (5,) [xy, xz, yz, aniso, zz]
        - G_norm: scalar JAX value (Frobenius norm)
    """
    ring_positions = positions[jnp.array(ring_atoms)]
    centroid = jnp.mean(ring_positions, axis=0)
    r_from_centroid = ring_positions - centroid

    # compute per-displacement tensors and sum
    per_atom, _ = compute_gyration_tensor(r_from_centroid)  # shape (k,5)
    G_total = jnp.sum(per_atom, axis=0)

    # use helper for norm
    G_norm = compute_frobenius_norm(G_total)

    return G_total, G_norm




def row_to_complex(
    row,
    element_to_idx: Dict[str, int],
    cutoff: float = 5.25, #minimum distance at which all N atoms have at least 2 edges
    max_neighbors: Optional[int] = 5, #otherwise explodes for C,H,O
    fully_connected: bool = False,
    include_atom_features: bool = False, #for molecular property prediction with cutoff
    max_dim: int = 2,
) -> Complex:
    """
    Convert a molecule row to a Complex using distance-based connectivity.

    Edges are atom pairs within a distance *cutoff*.  When *max_neighbors*
    is set, each atom keeps at most that many closest neighbours (the union
    of per-atom neighbour lists determines the final edge set).  When
    *fully_connected* is ``True``, all N*(N-1)/2 atom pairs become edges
    and the dim-2 (bags-of-bonds) cochain is left empty; *cutoff* and
    *max_neighbors* are ignored.

    The dim-2 cells are **bags of bonds**: each cell groups all edges
    connecting the same element-pair type (e.g. all C–C edges, all C–H
    edges).  Omitted when *fully_connected* is ``True``.

    Connectivity rules:
    * An edge's coboundary cell is the bag of bonds it belongs to.
    * A bag's boundary cells are all edges of that type.
    * Two edges are upper-adjacent when they belong to the same bag
      (intermediary = bag index).
    * Bags of bonds have no down adjacency or upper adjacency.
    * A bag's coboundary cells are all atoms whose element appears in
      the bond type (e.g. bag C–H → all C and H atoms).
    * Equivalently, bags are boundary senders for the node cochain.
    * Two nodes are **down-adjacent** when they share a bag in their
      boundary but are **not** upper-adjacent; the intermediary is
      the shared bag index.

    Args:
        row: Pandas Series with ``atom``, ``position_x/y/z`` columns.
        element_to_idx: Dict mapping element symbols to integer indices.
        cutoff: Distance cutoff (Bohr) for edge creation.  Ignored when
            *fully_connected* is ``True``.
        max_neighbors: If set, each atom keeps at most this many closest
            neighbours.  The final edge set is the union across all atoms.
            Ignored when *fully_connected* is ``True``.
        fully_connected: If ``True``, connect every atom pair as an edge
            and leave the dim-2 cochain empty.

    Returns:
        Complex with nodes (dim 0), edges (dim 1), and bags-of-bonds (dim 2).
    """
    # ---- atoms ----
    try:
        atom_symbols = row['atom']
    except KeyError:
        atom_symbols = row['elements'] 
    if hasattr(atom_symbols, 'tolist'):
        atom_symbols = atom_symbols.tolist()
    n_atoms = len(atom_symbols)

    pos_np = np.column_stack([
        np.asarray(row['position_x'], dtype=np.float32),
        np.asarray(row['position_y'], dtype=np.float32),
        np.asarray(row['position_z'], dtype=np.float32),
    ])
    positions = jnp.asarray(pos_np)

    species_indices = jnp.array(
        [element_to_idx.get(sym, len(element_to_idx)) for sym in atom_symbols],
        dtype=jnp.int32,
    )

    # ---- edges ----
    i_upper, j_upper = np.triu_indices(n_atoms, k=1)
    if fully_connected:
        edge_i = i_upper.tolist()
        edge_j = j_upper.tolist()
    else:
        diff = pos_np[:, None, :] - pos_np[None, :, :]   # (n, n, 3)
        dists = np.linalg.norm(diff, axis=-1)              # (n, n)
        within = dists[i_upper, j_upper] <= cutoff
        edge_i = i_upper[within].tolist()
        edge_j = j_upper[within].tolist()
    edges = list(zip(edge_i, edge_j))

    if not fully_connected and max_neighbors is not None and edges:
        # For each atom keep only the max_neighbors closest; final set is the union
        allowed_edge_indices = set()
        for atom_idx in range(n_atoms):
            atom_edges = []
            for e_idx, (a1, a2) in enumerate(edges):
                if a1 == atom_idx or a2 == atom_idx:
                    atom_edges.append((dists[a1, a2], e_idx))
            atom_edges.sort()  # sort by distance
            for _, e_idx in atom_edges[:max_neighbors]:
                allowed_edge_indices.add(e_idx)
        edges = [edges[i] for i in sorted(allowed_edge_indices)]

    n_edges = len(edges)
    edge_i = [e[0] for e in edges]
    edge_j = [e[1] for e in edges]

    # ---- bond types & bags of bonds ----
    if fully_connected or max_dim < 2:
        n_bags = 0
        unique_bond_types: List = []
        bag_edge_lists: Dict[int, List[int]] = {}
    else:
        edge_bond_types = [
            tuple(sorted([atom_symbols[a1], atom_symbols[a2]]))
            for a1, a2 in edges
        ]
        unique_bond_types = sorted(set(edge_bond_types))
        bond_type_to_bag = {bt: idx for idx, bt in enumerate(unique_bond_types)}
        n_bags = len(unique_bond_types)

        bag_edge_lists = {idx: [] for idx in range(n_bags)}
        for edge_idx, bt in enumerate(edge_bond_types):
            bag_edge_lists[bond_type_to_bag[bt]].append(edge_idx)

    # ---- edge gyration tensors and interatomic distances ----
    if n_edges > 0:
        r_ij_np = pos_np[edge_j] - pos_np[edge_i]         # (n_edges, 3)
        edge_G, edge_G_norm = compute_gyration_tensor(jnp.asarray(r_ij_np))
        edge_distance = jnp.asarray(
            np.linalg.norm(r_ij_np, axis=-1), dtype=jnp.float32
        )  # (n_edges,)
    else:
        edge_G = jnp.zeros((0, 5), dtype=jnp.float32)
        edge_G_norm = jnp.zeros((0,), dtype=jnp.float32)
        edge_distance = jnp.zeros((0,), dtype=jnp.float32)

    # ---- node connectivity (dim 0) ----
    node_up_s, node_up_r, node_up_i = [], [], []
    node_cb_s, node_cb_r = [], []
    for edge_idx, (a1, a2) in enumerate(edges):
        node_up_s.extend([a1, a2])
        node_up_r.extend([a2, a1])
        node_up_i.extend([edge_idx, edge_idx])
        node_cb_s.extend([edge_idx, edge_idx])
        node_cb_r.extend([a1, a2])

    # ---- edge down adjacency (shared node) ----
    node_to_edges: Dict[int, List[int]] = {i: [] for i in range(n_atoms)}
    for edge_idx, (a1, a2) in enumerate(edges):
        node_to_edges[a1].append(edge_idx)
        node_to_edges[a2].append(edge_idx)

    edge_dn_s, edge_dn_r, edge_dn_i = [], [], []
    for node, elist in node_to_edges.items():
        for i, e1 in enumerate(elist):
            for e2 in elist[i + 1:]:
                edge_dn_s.extend([e1, e2])
                edge_dn_r.extend([e2, e1])
                edge_dn_i.extend([node, node])

    # ---- edge boundary ----
    edge_bd_s, edge_bd_r = [], []
    for edge_idx, (a1, a2) in enumerate(edges):
        edge_bd_s.extend([a1, a2])
        edge_bd_r.extend([edge_idx, edge_idx])

    # ---- edge up adjacency (same bag of bonds) & edge coboundary ----
    edge_up_s, edge_up_r, edge_up_i = [], [], []
    edge_cb_s, edge_cb_r = [], []
    for bag_idx in range(n_bags):
        el = bag_edge_lists[bag_idx]
        for e_idx in el:
            edge_cb_s.append(bag_idx)
            edge_cb_r.append(e_idx)
        for i, e1 in enumerate(el):
            for e2 in el[i + 1:]:
                edge_up_s.extend([e1, e2])
                edge_up_r.extend([e2, e1])
                edge_up_i.extend([bag_idx, bag_idx])

    # ---- bag-of-bonds boundary ----
    bag_bd_s, bag_bd_r = [], []
    for bag_idx in range(n_bags):
        for e_idx in bag_edge_lists[bag_idx]:
            bag_bd_s.append(e_idx)
            bag_bd_r.append(bag_idx)

    # ---- bag-of-bonds gyration tensors and sum of edge distances ----
    bag_G_list, bag_G_norm_list, bag_dist_sum_list = [], [], []
    for bag_idx in range(n_bags):
        eidxs = jnp.array(bag_edge_lists[bag_idx])
        bag_tensor = jnp.sum(edge_G[eidxs], axis=0)
        bag_G_list.append(bag_tensor)
        bag_G_norm_list.append(compute_frobenius_norm(bag_tensor))
        bag_dist_sum_list.append(jnp.sum(edge_distance[eidxs]))

    if n_bags > 0:
        bag_G_arr = jnp.stack(bag_G_list)
        bag_G_norm_arr = jnp.array(bag_G_norm_list, dtype=jnp.float32)
        bag_dist_sum_arr = jnp.array(bag_dist_sum_list, dtype=jnp.float32)
    else:
        bag_G_arr = jnp.zeros((0, 5), dtype=jnp.float32)
        bag_G_norm_arr = jnp.zeros((0,), dtype=jnp.float32)
        bag_dist_sum_arr = jnp.zeros((0,), dtype=jnp.float32)

    # ---- bag coboundary → nodes & node boundary ← bags ----
    # For bag type (X, Y), coboundary = all atoms of element X or Y
    element_to_atoms: Dict[str, List[int]] = {}
    for atom_idx, sym in enumerate(atom_symbols):
        element_to_atoms.setdefault(sym, []).append(atom_idx)

    bag_cb_s, bag_cb_r = [], []   # senders=node idx, receivers=bag idx
    node_bd_s, node_bd_r = [], []  # senders=bag idx, receivers=node idx

    # Also build node→bags mapping for down adjacency
    node_to_bags: Dict[int, List[int]] = {i: [] for i in range(n_atoms)}

    for bag_idx, bt in enumerate(unique_bond_types):
        coboundary_atoms = set()
        for elem in bt:
            coboundary_atoms.update(element_to_atoms.get(elem, []))
        for atom_idx in sorted(coboundary_atoms):
            bag_cb_s.append(atom_idx)
            bag_cb_r.append(bag_idx)
            node_bd_s.append(bag_idx)
            node_bd_r.append(atom_idx)
            node_to_bags[atom_idx].append(bag_idx)

    # ---- node down adjacency (shared bag, not upper-adjacent) ----
    upper_adjacent_set = set()
    for a1, a2 in edges:
        upper_adjacent_set.add((min(a1, a2), max(a1, a2)))

    node_dn_s, node_dn_r, node_dn_i = [], [], []
    node_bag_sets = {i: set(node_to_bags[i]) for i in range(n_atoms)}
    for n1 in range(n_atoms):
        for n2 in range(n1 + 1, n_atoms):
            if (n1, n2) in upper_adjacent_set:
                continue
            shared_bags = node_bag_sets[n1] & node_bag_sets[n2]
            for bag_idx in shared_bags:
                node_dn_s.extend([n1, n2])
                node_dn_r.extend([n2, n1])
                node_dn_i.extend([bag_idx, bag_idx])

    # ---- helper for int32 arrays ----
    node_static: Dict[str, np.ndarray] = {'Z': species_indices, 'pos': positions}
    if include_atom_features:
        def _col(name):
            vals = row.get(name, None)
            if vals is None:
                return np.zeros(n_atoms, dtype=np.float32)
            return np.array(list(vals), dtype=np.float32)

        node_static['N']       = _col('N')
        node_static['LI']      = _col('LI')
        node_static['Mu_X']    = _col('Mu_X')
        node_static['Mu_Y']    = _col('Mu_Y')
        node_static['Mu_Z']    = _col('Mu_Z')
        node_static['Q_XY']    = _col('Q_XY')
        node_static['Q_XZ']    = _col('Q_XZ')
        node_static['Q_YZ']    = _col('Q_YZ')
        node_static['Q_aniso'] = _col('Q_aniso')
        node_static['Q_ZZ']    = _col('Q_ZZ')

    # ---- build cochains ----
    _a = lambda lst: jnp.asarray(lst, dtype=jnp.int32) if lst else jnp.array([], dtype=jnp.int32)
    node_cochain = Cochain(
        dim=0, num_cells=n_atoms,
        x=jnp.zeros((n_atoms, 1), dtype=jnp.float32),
        static=node_static,
        num_cells_up=n_edges, num_cells_down=n_bags,
        up_senders=_a(node_up_s), up_receivers=_a(node_up_r),
        up_intermediaries=_a(node_up_i),
        down_senders=_a(node_dn_s), down_receivers=_a(node_dn_r),
        down_intermediaries=_a(node_dn_i),
        boundary_senders=_a(node_bd_s), boundary_receivers=_a(node_bd_r),
        coboundary_senders=_a(node_cb_s), coboundary_receivers=_a(node_cb_r),
        y=None,
    )

    # Store which atoms form each edge for downstream geometry layers
    if n_edges > 0:
        edge_atoms = jnp.asarray(
            np.stack([edge_i, edge_j], axis=1), dtype=jnp.int32
        )  # (n_edges, 2)
    else:
        edge_atoms = jnp.zeros((0, 2), dtype=jnp.int32)

    edge_cochain = Cochain(
        dim=1, num_cells=n_edges,
        x=jnp.zeros((n_edges, 1), dtype=jnp.float32),
        static={'G': edge_G, 'G_norm': edge_G_norm, 'distance': edge_distance,
                'atoms': edge_atoms},
        num_cells_up=n_bags, num_cells_down=n_atoms,
        up_senders=_a(edge_up_s), up_receivers=_a(edge_up_r),
        up_intermediaries=_a(edge_up_i),
        down_senders=_a(edge_dn_s), down_receivers=_a(edge_dn_r),
        down_intermediaries=_a(edge_dn_i),
        boundary_senders=_a(edge_bd_s), boundary_receivers=_a(edge_bd_r),
        coboundary_senders=_a(edge_cb_s), coboundary_receivers=_a(edge_cb_r),
        y=None,
    )

    bag_cochain = Cochain(
        dim=2, num_cells=n_bags,
        x=jnp.zeros((n_bags, 1), dtype=jnp.float32),
        static={'G': bag_G_arr, 'G_norm': bag_G_norm_arr, 'distance_sum': bag_dist_sum_arr},
        num_cells_up=n_atoms, num_cells_down=n_edges,
        up_senders=None, up_receivers=None, up_intermediaries=None,
        down_senders=None, down_receivers=None, down_intermediaries=None,
        boundary_senders=_a(bag_bd_s), boundary_receivers=_a(bag_bd_r),
        coboundary_senders=_a(bag_cb_s), coboundary_receivers=_a(bag_cb_r),
        y=None,
    )

    if max_dim >= 2:
        return Complex(cochains=[node_cochain, edge_cochain, bag_cochain], dimension=2, y=None)
    else:
        return Complex(cochains=[node_cochain, edge_cochain], dimension=1, y=None)



def precompute_complexes(
    df,
    element_to_idx: Dict[str, int],
    cutoff: float = 3.5,
    max_neighbors: Optional[int] = None,
    fully_connected: bool = False,
    max_dim: int = 2,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Precompute Complex representations for every row in a DataFrame.

    Complexes are keyed by the DataFrame's ``.index`` (i.e. ``loc`` labels),
    not by positional index (``iloc``), so they remain valid after
    train / val splitting across CV folds.

    Because geometry (positions, bonds, gyration tensors) is
    fold-independent, this function should be called **once** on the
    full dataset *before* any regularization.  Fold-specific targets
    can later be attached with :func:`attach_targets_to_complexes`.

    Args:
        df: DataFrame with molecular data (must contain columns expected
            by :func:`row_to_complex`: ``atom``,
            ``position_x``, ``position_y``, ``position_z``).
        element_to_idx: Dict mapping element symbols to integer indices.
        cutoff: Distance cutoff (Bohr) for edge creation.  Ignored when
            *fully_connected* is ``True``.
        max_neighbors: If set, each atom keeps at most this many closest
            neighbours (passed through to :func:`row_to_complex`).  Ignored
            when *fully_connected* is ``True``.
        fully_connected: If ``True``, connect every atom pair as an edge
            and leave the dim-2 cochain empty (passed through to
            :func:`row_to_complex`).
        output_path: If provided, persist the dict to this ``.pkl`` file.
        verbose: Print progress every 500 molecules.

    Returns:
        Dict mapping ``df.index`` values → ``Complex`` objects.
    """
    complexes = {}
    n_rows = len(df)

    for i, (idx, row) in enumerate(df.iterrows()):
        if verbose and (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{n_rows} molecules")
        complexes[idx] = row_to_complex(row, element_to_idx, cutoff=cutoff,
                                        max_neighbors=max_neighbors,
                                        fully_connected=fully_connected,
                                        max_dim=max_dim)

    if verbose:
        print(f"  Precomputed {len(complexes)} complexes")

    if output_path is not None:
        with open(output_path, 'wb') as f:
            pickle.dump(complexes, f)
        if verbose:
            print(f"  Saved to {output_path}")

    return complexes


def attach_targets_to_complexes(
    complexes: Dict,
    df,
    target_columns: List[str],
    verbose: bool = False,
) -> List[Complex]:
    """
    Attach per-atom targets from a (regularized) DataFrame to precomputed Complexes.

    For every row in *df* the corresponding ``Complex`` is looked up by
    ``df.index`` (``loc``).  The values in *target_columns* — which must be
    per-atom list / array columns — are stacked into a 2-D array of shape
    ``(num_atoms, len(target_columns))`` and stored in the ``y`` attribute
    of the **node cochain** (dimension 0).

    Args:
        complexes: Dict mapping ``df.index`` → ``Complex`` (from
            :func:`precompute_complexes`).
        df: DataFrame (typically after regularization) whose rows
            correspond to entries in *complexes*.
        target_columns: Column names whose per-atom values become the
            target array.
        verbose: If True, print timing information.

    Returns:
        List of ``Complex`` objects (in ``df`` row order) with ``y``
        set on the node cochain.  ``y.shape == (num_atoms,
        len(target_columns))``.
    """
    t0 = time.perf_counter()
    result = []

    for idx, row in df.iterrows():
        if idx not in complexes:
            raise KeyError(f"No precomputed Complex found for index {idx}")

        comp = complexes[idx]
        n_atoms = comp.cochains[0].num_cells

        # Build per-atom target array with numpy, convert to jnp once
        target_arrays = []
        for col in target_columns:
            vals = np.asarray(row[col], dtype=np.float32)
            if vals.shape[0] != n_atoms:
                raise ValueError(
                    f"Row {idx}: column '{col}' has {vals.shape[0]} values "
                    f"but complex has {n_atoms} atoms"
                )
            target_arrays.append(vals)

        # (num_atoms, num_targets) — single jnp conversion
        y = jnp.asarray(np.column_stack(target_arrays))

        # Replace y in node cochain (dim=0) — Cochain is a flax struct dataclass
        new_node_cochain = comp.cochains[0].replace(y=y)
        new_cochains = [new_node_cochain] + list(comp.cochains[1:])
        new_complex = Complex(
            cochains=new_cochains,
            dimension=comp.dimension,
            y=comp.y,
        )
        result.append(new_complex)

    if verbose:
        print(f"  attach_targets_to_complexes: {time.perf_counter() - t0:.2f}s "
              f"for {len(result)} complexes")

    return result



#TODO: add support to read targets for molecular properties
def prepare_padded_batches(
    complexes: Dict,
    df,
    target_columns: List[str],
    batch_size: int = 32,
    verbose: bool = True,
    as_numpy: bool = False,
) -> List[ComplexBatch]:
    """
    Combined attach-targets → batch → pad pipeline via direct pre-allocation.

    Instead of building intermediate ``Complex``, ``CochainBatch``, and
    ``ComplexBatch`` objects, this function:

    1. **Scans sizes** — one lightweight pass (pure Python ints, no array ops)
       to determine per-batch / per-dimension totals and global maxima.
    2. **Pre-allocates padded arrays** — one ``np.zeros`` / ``np.full`` per
       field per dimension, already at the final padded size (``max + 1`` OOB
       row for features, ``max`` for index arrays filled with OOB sentinel).
    3. **Fills in-place** — iterates over complexes once, copies data with
       simple slice assignments (``arr[off:off+n] = src``).  Each dimension is
       processed independently; the only cross-dimension information is the
       OOB fill value for intermediary / boundary / coboundary index arrays,
       which comes from step 1.
    4. **Converts to jnp once** per CochainBatch at the end.

    This eliminates all intermediate concatenation, all temporary Python lists
    of arrays, and all per-operation JAX dispatch overhead during construction.

    Args:
        complexes: Dict mapping ``df.index`` → ``Complex``
            (from :func:`precompute_complexes`).
        df: DataFrame (typically after regularisation).
        target_columns: Per-atom target column names.
        batch_size: Number of complexes per batch.
        verbose: Print per-stage timing.

    Returns:
        List of padded ``ComplexBatch`` objects ready for training.
    """
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 0 — gather ordered (Complex, target_numpy) pairs
    # ------------------------------------------------------------------
    ordered: List[Tuple[Complex, np.ndarray]] = []
    for idx, row in df.iterrows():
        if idx not in complexes:
            raise KeyError(f"No precomputed Complex for index {idx}")
        comp = complexes[idx]
        n_atoms = comp.cochains[0].num_cells

        cols: List[np.ndarray] = []
        for col in target_columns:
            v = np.asarray(row[col], dtype=np.float32)
            if v.shape[0] != n_atoms:
                raise ValueError(
                    f"Row {idx}: column '{col}' has {v.shape[0]} values "
                    f"but complex has {n_atoms} atoms"
                )
            cols.append(v)
        target_arr = np.column_stack(cols) if cols else np.empty((n_atoms, 0), dtype=np.float32)
        ordered.append((comp, target_arr))

    n_total  = len(ordered)
    num_dims = ordered[0][0].dimension + 1              # typically 3

    t1 = time.perf_counter()
    if verbose:
        print(f"  Gather targets: {t1 - t0:.2f}s ({n_total} complexes)")

    # ------------------------------------------------------------------
    # Step 1 — batch ranges & per-dimension max sizes  (pure Python ints)
    # ------------------------------------------------------------------
    batch_ranges = [
        (i, min(i + batch_size, n_total))
        for i in range(0, n_total, batch_size)
    ]

    # Per-batch, per-dim totals  (lightweight integer scan)
    max_sizes: Dict[int, Dict[str, int]] = {}
    for d in range(num_dims):
        max_c = max_u = max_d = max_b = max_cb = 0
        for start, end in batch_ranges:
            tc = tu = td = tb = tcb = 0
            for j in range(start, end):
                c = ordered[j][0].cochains[d]
                tc += c.num_cells
                if c.up_senders is not None:
                    tu += c.up_senders.shape[0]
                if c.down_senders is not None:
                    td += c.down_senders.shape[0]
                if c.boundary_senders is not None:
                    tb += c.boundary_senders.shape[0]
                if c.coboundary_senders is not None:
                    tcb += c.coboundary_senders.shape[0]
            max_c  = max(max_c, tc)
            max_u  = max(max_u, tu)
            max_d  = max(max_d, td)
            max_b  = max(max_b, tb)
            max_cb = max(max_cb, tcb)
        max_sizes[d] = {
            'cells': int(max_c), 'up': int(max_u), 'down': int(max_d),
            'boundary': int(max_b), 'coboundary': int(max_cb),
        }

    # Feature dim per dim (from first complex)
    feat_dims: Dict[int, Optional[int]] = {}
    for d in range(num_dims):
        ref_x = ordered[0][0].cochains[d].x
        feat_dims[d] = int(ref_x.shape[1]) if ref_x is not None else None

    # Static keys / trailing shapes per dim (from first complex)
    static_info: Dict[int, Optional[Dict[str, Tuple]]] = {}
    for d in range(num_dims):
        ref_s = ordered[0][0].cochains[d].static
        if ref_s is not None:
            static_info[d] = {
                k: (tuple(int(s) for s in np.asarray(v).shape[1:]),
                    np.asarray(v).dtype)
                for k, v in ref_s.items()
            }
        else:
            static_info[d] = None

    num_targets = len(target_columns)

    t2 = time.perf_counter()
    if verbose:
        print(f"  Size scan: {t2 - t1:.2f}s "
              f"({len(batch_ranges)} batches, {num_dims} dims)")

    # ------------------------------------------------------------------
    # Step 2 — pre-allocate & fill  (one pass per batch × dimension)
    # ------------------------------------------------------------------
    if as_numpy:
        _j     = lambda a: a
        _jdict = lambda d: d if d else None
        _arr   = np.array
    else:
        _j     = lambda a: jnp.asarray(a) if a is not None else None
        _jdict = lambda d: {k: jnp.asarray(v) for k, v in d.items()} if d else None
        _arr   = jnp.array

    padded_batches: List[ComplexBatch] = []

    for start, end in batch_ranges:
        bs_actual = end - start
        cochain_batches: List[CochainBatch] = []

        # Each dimension is independent (cross-dim info limited to OOB fill
        # values, which we already know from max_sizes).
        for d in range(num_dims):
            mc  = max_sizes[d]['cells']
            mu  = max_sizes[d]['up']
            md_ = max_sizes[d]['down']
            mb  = max_sizes[d]['boundary']
            mcb = max_sizes[d]['coboundary']
            fd  = feat_dims[d]

            # OOB fill values for cross-dimension index arrays
            oob_up   = max_sizes[d + 1]['cells'] if d + 1 < num_dims else mc
            oob_down = max_sizes[d - 1]['cells'] if d > 0 else mc

            rows = mc + 1                          # +1 OOB zero row

            # ---- pre-allocate feature arrays ----
            x      = np.zeros((rows, fd), dtype=np.float32) if fd is not None else None
            x_mask = np.zeros(rows, dtype=bool)    if fd is not None else np.array([], dtype=bool)
            # Padding cells (the OOB +1 slot) route to the extra complex index
            # bs_actual so they never pollute real complex features in segment_sum.
            owner  = np.full(rows, bs_actual, dtype=np.int32)

            y_arr = np.zeros((rows, num_targets), dtype=np.float32) if d == 0 else None

            if static_info[d] is not None:
                static_arrs = {
                    k: np.zeros((rows,) + tshape, dtype=dt)
                    for k, (tshape, dt) in static_info[d].items()
                }
            else:
                static_arrs = None

            # ---- pre-allocate index arrays (filled with OOB sentinel) ----
            up_s  = np.full(mu,  mc,       dtype=np.int32) if mu  > 0 else None
            up_r  = np.full(mu,  mc,       dtype=np.int32) if mu  > 0 else None
            up_i  = np.full(mu,  oob_up,   dtype=np.int32) if mu  > 0 else None
            up_m  = np.zeros(mu, dtype=bool)               if mu  > 0 else np.array([], dtype=bool)

            dn_s  = np.full(md_, mc,       dtype=np.int32) if md_ > 0 else None
            dn_r  = np.full(md_, mc,       dtype=np.int32) if md_ > 0 else None
            dn_i  = np.full(md_, oob_down, dtype=np.int32) if md_ > 0 else None
            dn_m  = np.zeros(md_, dtype=bool)              if md_ > 0 else np.array([], dtype=bool)

            bd_s  = np.full(mb,  oob_down, dtype=np.int32) if mb  > 0 else None
            bd_r  = np.full(mb,  mc,       dtype=np.int32) if mb  > 0 else None
            bd_m  = np.zeros(mb, dtype=bool)               if mb  > 0 else np.array([], dtype=bool)

            co_s  = np.full(mcb, oob_up,   dtype=np.int32) if mcb > 0 else None
            co_r  = np.full(mcb, mc,       dtype=np.int32) if mcb > 0 else None
            co_m  = np.zeros(mcb, dtype=bool)              if mcb > 0 else np.array([], dtype=bool)

            # per-complex metadata (goes into CochainBatch as arrays)
            nc_per  = np.zeros(bs_actual, dtype=np.int32)
            ncu_per = np.zeros(bs_actual, dtype=np.int32)
            ncd_per = np.zeros(bs_actual, dtype=np.int32)

            # ---- fill from each complex ----
            c_off  = 0    # cell offset (current dim)
            u_off  = 0    # up-index offset
            d_off  = 0    # down-index offset
            b_off  = 0    # boundary-index offset
            cb_off = 0    # coboundary-index offset
            cu_off = 0    # cumulative cells in dim d+1 (up intermediaries)
            cd_off = 0    # cumulative cells in dim d−1 (down intermediaries, boundary senders)

            for li, gj in enumerate(range(start, end)):
                comp, targets = ordered[gj]
                ch = comp.cochains[d]
                nc = ch.num_cells

                # x
                if x is not None and ch.x is not None:
                    x[c_off:c_off + nc] = np.asarray(ch.x)
                    x_mask[c_off:c_off + nc] = True

                # owner
                owner[c_off:c_off + nc] = li

                # y (dim 0 only)
                if d == 0:
                    y_arr[c_off:c_off + nc] = targets

                # static
                if static_arrs is not None and ch.static is not None:
                    for key in static_arrs:
                        if key in ch.static:
                            static_arrs[key][c_off:c_off + nc] = np.asarray(ch.static[key])

                # up
                if ch.up_senders is not None and up_s is not None:
                    nu = int(ch.up_senders.shape[0])
                    up_s[u_off:u_off + nu] = np.asarray(ch.up_senders)        + c_off
                    up_r[u_off:u_off + nu] = np.asarray(ch.up_receivers)       + c_off
                    up_i[u_off:u_off + nu] = np.asarray(ch.up_intermediaries)  + cu_off
                    up_m[u_off:u_off + nu] = True
                    u_off += nu

                # down
                if ch.down_senders is not None and dn_s is not None:
                    nd = int(ch.down_senders.shape[0])
                    dn_s[d_off:d_off + nd] = np.asarray(ch.down_senders)        + c_off
                    dn_r[d_off:d_off + nd] = np.asarray(ch.down_receivers)       + c_off
                    dn_i[d_off:d_off + nd] = np.asarray(ch.down_intermediaries)  + cd_off
                    dn_m[d_off:d_off + nd] = True
                    d_off += nd

                # boundary
                if ch.boundary_senders is not None and bd_s is not None:
                    nb = int(ch.boundary_senders.shape[0])
                    bd_s[b_off:b_off + nb] = np.asarray(ch.boundary_senders)   + cd_off
                    bd_r[b_off:b_off + nb] = np.asarray(ch.boundary_receivers)  + c_off
                    bd_m[b_off:b_off + nb] = True
                    b_off += nb

                # coboundary
                if ch.coboundary_senders is not None and co_s is not None:
                    ncb_ = int(ch.coboundary_senders.shape[0])
                    co_s[cb_off:cb_off + ncb_] = np.asarray(ch.coboundary_senders)   + cu_off
                    co_r[cb_off:cb_off + ncb_] = np.asarray(ch.coboundary_receivers)  + c_off
                    co_m[cb_off:cb_off + ncb_] = True
                    cb_off += ncb_

                # per-complex metadata
                nc_per[li]  = nc
                ncu_per[li] = ch.num_cells_up   if ch.num_cells_up   is not None else 0
                ncd_per[li] = ch.num_cells_down if ch.num_cells_down is not None else 0

                # advance cumulative offsets
                c_off  += nc
                cu_off += (ch.num_cells_up   if ch.num_cells_up   is not None else 0)
                cd_off += (ch.num_cells_down if ch.num_cells_down is not None else 0)

            # ---- single jnp conversion → CochainBatch ----
            cochain_batches.append(CochainBatch(
                dim=d,
                num_cells=_j(nc_per),
                x=_j(x),
                static=_jdict(static_arrs),
                owner_cochains=_j(owner),
                num_cells_up=_j(ncu_per),
                num_cells_down=_j(ncd_per),
                up_senders=_j(up_s),       up_receivers=_j(up_r),
                up_intermediaries=_j(up_i),
                down_senders=_j(dn_s),     down_receivers=_j(dn_r),
                down_intermediaries=_j(dn_i),
                boundary_senders=_j(bd_s), boundary_receivers=_j(bd_r),
                coboundary_senders=_j(co_s), coboundary_receivers=_j(co_r),
                y=_j(y_arr),
                x_mask=_j(x_mask),
                up_mask=_j(up_m),
                down_mask=_j(dn_m),
                boundary_mask=_j(bd_m),
                coboundary_mask=_j(co_m),
            ))

        padded_batches.append(ComplexBatch(
            cochain_batches=cochain_batches,
            dimension=num_dims - 1,
            num_complexes=_arr([bs_actual]),
            y=None,
        ))

    t3 = time.perf_counter()
    if verbose:
        print(f"  Fill + convert: {t3 - t2:.2f}s")
        print(f"  Total prepare_padded_batches: {t3 - t0:.2f}s")

    return padded_batches


# ---------------------------------------------------------------------------
# Molecular complex builders (BCP/RCP topology, no 3D geometry)
# ---------------------------------------------------------------------------

def _parse_bcp_to_edges(a_name, bcp_connectivity):
    """Build sorted undirected edge list from per-atom BCP connectivity strings."""
    name_to_idx = {name: i for i, name in enumerate(a_name)}
    edges = set()
    for i, bcp in enumerate(bcp_connectivity):
        if bcp is None:
            continue
        s = str(bcp).strip()
        if s == '' or s == 'nan':
            continue
        for neighbour in s.split(';'):
            neighbour = neighbour.strip()
            if neighbour and neighbour in name_to_idx:
                j = name_to_idx[neighbour]
                if i != j:
                    edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def _parse_rcp_to_rings(a_name, rcp_connectivity):
    """Build list of frozenset(atom_indices) per unique ring from per-atom RCP strings."""
    name_to_idx = {name: i for i, name in enumerate(a_name)}
    rings = set()
    for rcp in rcp_connectivity:
        if rcp is None:
            continue
        s = str(rcp).strip()
        if s == '' or s == 'nan':
            continue
        for ring_str in s.split(';'):
            ring_str = ring_str.strip()
            if not ring_str:
                continue
            atom_names = [n.strip() for n in ring_str.split(',')]
            ring_idx_set = frozenset(
                name_to_idx[n] for n in atom_names if n in name_to_idx
            )
            if len(ring_idx_set) >= 3:
                rings.add(ring_idx_set)
    return list(rings)


def row_to_molecular_complex(row, element_to_idx, include_atom_features=False):
    """
    Build a Complex from a .pkl row using BCP/RCP topology (no 3D geometry).

    Always returns a dimension=2 Complex with three cochains so that all
    molecules in a batch share the same structure for padding:
      Dim-0: atoms  — species index Z; optionally N, LI, Mu, Q atomic features
      Dim-1: bonds  — BCP-derived undirected edges
      Dim-2: rings  — RCP-derived rings (empty cochain when no rings exist)

    Connectivity conventions mirror row_to_complex (bags-of-bonds → rings):
      - boundary_senders of dim-k cochain: indices of (k-1)-dim cells
      - coboundary_senders of dim-k cochain: indices of (k+1)-dim cells
      - num_cells_down = #(k-1)-dim cells  (bounds boundary_senders)
      - num_cells_up   = #(k+1)-dim cells  (bounds coboundary_senders)
    All arrays use numpy (conversion to JAX happens in prepare_padded_batches).
    """
    _a = lambda lst: np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32)

    a_name = list(row['a_name'])
    n_atoms = len(a_name)

    bcp = row.get('BCP_connectivity', [None] * n_atoms)
    if bcp is None:
        bcp = [None] * n_atoms
    bcp = list(bcp)

    rcp = row.get('RCP_connectivity', [None] * n_atoms)
    if rcp is None:
        rcp = [None] * n_atoms
    rcp = list(rcp)

    # Strip trailing digits from atom names like "C1", "H2" → element symbol
    element_names = [''.join(c for c in name if not c.isdigit()) for name in a_name]
    species = np.array([element_to_idx.get(e, 0) for e in element_names], dtype=np.int32)

    # --- rings needed up-front for cochain counts ----------------------
    rings = _parse_rcp_to_rings(a_name, rcp)
    n_rings = len(rings)

    # --- edges (dim-1) -------------------------------------------------
    edge_list = _parse_bcp_to_edges(a_name, bcp)
    n_edges = len(edge_list)
    edge_to_idx = {e: k for k, e in enumerate(edge_list)}

    # node up: atom_i ← atom_j through edge k (both directions)
    node_up_s, node_up_r, node_up_i = [], [], []
    for k, (i, j) in enumerate(edge_list):
        node_up_s.extend([i, j]); node_up_r.extend([j, i])
        node_up_i.extend([k, k])

    # node coboundary: edge_k → atom_i and atom_j  (senders=edge, receivers=atom)
    node_cb_s, node_cb_r = [], []
    for k, (i, j) in enumerate(edge_list):
        node_cb_s.extend([k, k]); node_cb_r.extend([i, j])

    # edge down: edge pairs sharing a node (intermediary = shared node)
    edge_to_node_list: Dict[int, List[int]] = {i: [] for i in range(n_atoms)}
    for k, (i, j) in enumerate(edge_list):
        edge_to_node_list[i].append(k)
        edge_to_node_list[j].append(k)

    edge_dn_s, edge_dn_r, edge_dn_i = [], [], []
    for node_idx, inc_edges in edge_to_node_list.items():
        for a in inc_edges:
            for b in inc_edges:
                if a != b:
                    edge_dn_s.append(a); edge_dn_r.append(b)
                    edge_dn_i.append(node_idx)

    # edge boundary: atom_i → edge_k  (boundary of edge = its two endpoint atoms)
    edge_bd_s, edge_bd_r = [], []
    for k, (i, j) in enumerate(edge_list):
        edge_bd_s.extend([i, j]); edge_bd_r.extend([k, k])

    # --- ring connectivity ---------------------------------------------
    # node boundary:    ring_idx → atom_idx  (rings sending to member atoms)
    # ring coboundary:  atom_idx → ring_idx  (atoms pointing to containing rings)
    # ring boundary:    edge_idx → ring_idx  (edges forming ring boundary)
    # edge coboundary:  ring_idx → edge_idx  (rings above each member edge)
    node_bd_s, node_bd_r = [], []
    ring_cb_s, ring_cb_r = [], []
    ring_bd_s, ring_bd_r = [], []
    edge_cb_s, edge_cb_r = [], []

    for r_idx, ring_set in enumerate(rings):
        ring_list = sorted(ring_set)
        for atom_idx in ring_list:
            node_bd_s.append(r_idx); node_bd_r.append(atom_idx)
            ring_cb_s.append(atom_idx); ring_cb_r.append(r_idx)
        for a in ring_list:
            for b in ring_list:
                if a < b and (a, b) in edge_to_idx:
                    k = edge_to_idx[(a, b)]
                    ring_bd_s.append(k); ring_bd_r.append(r_idx)
                    edge_cb_s.append(r_idx); edge_cb_r.append(k)

    # --- node static features ------------------------------------------
    node_static: Dict[str, np.ndarray] = {'Z': species}
    if include_atom_features:
        def _col(name):
            vals = row.get(name, None)
            if vals is None:
                return np.zeros(n_atoms, dtype=np.float32)
            return np.array(list(vals), dtype=np.float32)

        node_static['N']       = _col('N')
        node_static['LI']      = _col('LI')
        node_static['Mu_X']    = _col('Mu_X')
        node_static['Mu_Y']    = _col('Mu_Y')
        node_static['Mu_Z']    = _col('Mu_Z')
        node_static['Q_XY']    = _col('Q_XY')
        node_static['Q_XZ']    = _col('Q_XZ')
        node_static['Q_YZ']    = _col('Q_YZ')
        node_static['Q_aniso'] = _col('Q_aniso')
        node_static['Q_ZZ']    = _col('Q_ZZ')

    # --- build cochains ------------------------------------------------
    # Node cochain (dim=0):
    #   num_cells_up   = n_edges  (bounds coboundary_senders = edge indices)
    #   num_cells_down = n_rings  (bounds boundary_senders = ring indices)
    node_cochain = Cochain(
        dim=0, num_cells=n_atoms,
        x=np.zeros((n_atoms, 1), dtype=np.float32),
        static=node_static,
        num_cells_up=n_edges, num_cells_down=n_rings,
        up_senders=_a(node_up_s), up_receivers=_a(node_up_r),
        up_intermediaries=_a(node_up_i),
        down_senders=_a([]), down_receivers=_a([]),
        down_intermediaries=_a([]),
        boundary_senders=_a(node_bd_s), boundary_receivers=_a(node_bd_r),
        coboundary_senders=_a(node_cb_s), coboundary_receivers=_a(node_cb_r),
        y=None,
    )

    # Edge cochain (dim=1):
    #   num_cells_up   = n_rings  (bounds coboundary_senders = ring indices)
    #   num_cells_down = n_atoms  (bounds boundary_senders = atom indices)
    edge_cochain = Cochain(
        dim=1, num_cells=n_edges,
        x=np.zeros((n_edges, 1), dtype=np.float32),
        static={},
        num_cells_up=n_rings, num_cells_down=n_atoms,
        up_senders=_a([]), up_receivers=_a([]),
        up_intermediaries=_a([]),
        down_senders=_a(edge_dn_s), down_receivers=_a(edge_dn_r),
        down_intermediaries=_a(edge_dn_i),
        boundary_senders=_a(edge_bd_s), boundary_receivers=_a(edge_bd_r),
        coboundary_senders=_a(edge_cb_s), coboundary_receivers=_a(edge_cb_r),
        y=None,
    )

    # Ring cochain (dim=2):
    #   num_cells_up   = n_atoms  (bounds coboundary_senders = atom indices)
    #   num_cells_down = n_edges  (bounds boundary_senders = edge indices)
    ring_cochain = Cochain(
        dim=2, num_cells=n_rings,
        x=np.zeros((n_rings, 1), dtype=np.float32),
        static={},
        num_cells_up=n_atoms, num_cells_down=n_edges,
        up_senders=None, up_receivers=None, up_intermediaries=None,
        down_senders=None, down_receivers=None, down_intermediaries=None,
        boundary_senders=_a(ring_bd_s), boundary_receivers=_a(ring_bd_r),
        coboundary_senders=_a(ring_cb_s), coboundary_receivers=_a(ring_cb_r),
        y=None,
    )

    return Complex(cochains=[node_cochain, edge_cochain, ring_cochain], dimension=2, y=None)
