"""
Functions for SOAP-based dataset split. Objective is to cluster atomic environ-
ments per central atom. We keep track of the molecule where the environment is
found since each subdataset needs to contain full molecules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ase import Atoms
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import hdbscan
import matplotlib.pyplot as plt
from matplotlib import cm


# SOAP hyperparameters for organic molecules
SOAP_PARAMS = {
    'r_cut': 4.5,      # Cutoff radius in Bohr
    'n_max': 8,        # Number of radial basis functions
    'l_max': 6,        # Maximum degree of spherical harmonics
    'sigma': 0.7,      # Width of Gaussian smearing in Bohr
    'species': ['H', 'C', 'N', 'O'],  # All element types in dataset
    'periodic': False,  # Molecules are not periodic
    'sparse': False,    # Use dense arrays
}


def row_to_ase_atoms(row: pd.Series) -> Atoms:
    """
    Convert a DataFrame row (one molecule) to an ASE Atoms object.
    
    Parameters
    ----------
    row : pd.Series
        A row from the AIMEl dataset containing:
        - 'atom': array of element symbols (e.g., ['C', 'H', 'H', 'O'])
        - 'position_x', 'position_y', 'position_z': arrays of atomic coordinates
        
    Returns
    -------
    ase.Atoms
        ASE Atoms object representing the molecule with 3D coordinates
        
    Examples
    --------
    >>> df = pd.read_pickle('aimel_molecules_grouped.pkl')
    >>> ase_mol = row_to_ase_atoms(df.iloc[0])
    >>> print(f"Molecule has {len(ase_mol)} atoms")
    """
    # Extract element symbols
    symbols = row['atom']
    
    # Build positions array (N_atoms x 3)
    positions = np.column_stack([
        row['position_x'],
        row['position_y'],
        row['position_z']
    ])
    
    # Create ASE Atoms object
    atoms = Atoms(symbols=symbols, positions=positions)
    
    return atoms


def compute_soap_descriptors(atoms: Atoms, soap_params: Optional[Dict] = SOAP_PARAMS) -> np.ndarray:
    """
    Compute SOAP descriptors for each atom in an ASE Atoms object.
    
    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object representing a molecule
    soap_params : dict, optional
        Dictionary of SOAP parameters. If None, uses default SOAP_PARAMS.
        Expected keys:
        - r_cut: float, cutoff radius in Angstroms
        - n_max: int, number of radial basis functions
        - l_max: int, maximum degree of spherical harmonics
        - sigma: float, width of Gaussian smearing
        - species: list of str, element types to consider
        - periodic: bool, whether structure is periodic
        - sparse: bool, whether to use sparse representation
        
    Returns
    -------
    np.ndarray
        SOAP descriptor matrix of shape (n_atoms, n_features)
        where n_features depends on n_max, l_max, and number of species
        
    Examples
    --------
    >>> ase_mol = row_to_ase_atoms(df.iloc[0])
    >>> soap_descriptors = compute_soap_descriptors(ase_mol)
    >>> print(f"SOAP shape: {soap_descriptors.shape}")
    >>> print(f"Descriptor dimensionality: {soap_descriptors.shape[1]}")
    
    Notes
    -----
    SOAP feature dimensionality formula:
    n_features = n_species * (n_species + 1) / 2 * n_max * (l_max + 1)
    For our params: 5 species, n_max=8, l_max=6 -> 15 * 8 * 7 = 840 features
    """
    
    # Initialize SOAP descriptor
    soap = SOAP(
        species=soap_params['species'],
        r_cut=soap_params['r_cut'],
        n_max=soap_params['n_max'],
        l_max=soap_params['l_max'],
        sigma=soap_params['sigma'],
        periodic=soap_params['periodic'],
        sparse=soap_params['sparse'],
    )
    
    # Compute SOAP descriptors for all atoms
    # Returns array of shape (n_atoms, n_features)
    soap_descriptors = soap.create(atoms)
    
    return soap_descriptors


def compute_soap_for_dataset(
    df: pd.DataFrame,
    soap_params: Optional[Dict] = SOAP_PARAMS,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Compute SOAP descriptors for all molecules in a dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each row is a molecule (from aimel_molecules_grouped.pkl)
    soap_params : dict, optional
        SOAP parameters to use. If None, uses default SOAP_PARAMS.
    verbose : bool, default=True
        Whether to print progress information
        
    Returns
    -------
    df_with_soap : pd.DataFrame
        Original DataFrame with added 'soap_descriptors' column containing
        SOAP vectors for each atom (shape: n_atoms x n_features)
    atom_info : dict
        Dictionary mapping element type to list of tuples:
        (molecule_idx, atom_idx, soap_vector)
        Used for clustering atoms by element type
        
    Examples
    --------
    >>> df = pd.read_pickle('aimel_molecules_grouped.pkl')
    >>> df_soap, atom_info = compute_soap_for_dataset(df)
    >>> print(f"Added SOAP descriptors to {len(df_soap)} molecules")
    >>> print(f"Element types: {list(atom_info.keys())}")
    >>> print(f"Number of H atoms: {len(atom_info['H'])}")
    """
    
    df_with_soap = df.copy()
    soap_descriptors_list = []
    
    # Dictionary to store (mol_idx, atom_idx, soap_vector) for each element
    atom_info = {elem: [] for elem in soap_params['species']}
    
    n_molecules = len(df)
    for mol_idx, row in df.iterrows():
        if verbose and mol_idx % 1000 == 0:
            print(f"Processing molecule {mol_idx}/{n_molecules} ({100*mol_idx/n_molecules:.1f}%)")
        
        # Convert to ASE and compute SOAP
        atoms = row_to_ase_atoms(row)
        soap_desc = compute_soap_descriptors(atoms)
        soap_descriptors_list.append(soap_desc)
        
        # Store atom information by element type
        elements = row['atom']
        for atom_idx, (element, soap_vector) in enumerate(zip(elements, soap_desc)):
            atom_info[element].append((mol_idx, atom_idx, soap_vector))
    
    # Add SOAP descriptors to dataframe
    df_with_soap['soap_descriptors'] = soap_descriptors_list
    
    if verbose:
        print(f"\nSOAP computation complete!")
        print(f"Descriptor dimensionality: {soap_descriptors_list[0].shape[1]}")
        print(f"\nAtom counts by element:")
        for elem in sorted(atom_info.keys()):
            print(f"  {elem}: {len(atom_info[elem]):,} atoms")
    
    return df_with_soap, atom_info




def cluster_single_element_environments(
    atom_info: Dict[str, List[Tuple]],
    element: str,
    target_pca_components: int = 80,
    min_cluster_size: int = 100,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = 'eom',
    core_dist_n_jobs: int = -1,
    random_state: int = 42
 ) -> Tuple[np.ndarray, Dict, PCA, hdbscan.HDBSCAN]:
    """
    Cluster atomic environments for a single element type with PCA.
    
    This function performs PCA dimensionality reduction followed by HDBSCAN
    clustering for one specific element (H, C, N, or O). This allows testing
    different hyperparameters for each element independently.
    
    Parameters
    ----------
    atom_info : dict
        Dictionary mapping element type to list of tuples:
        (molecule_idx, atom_idx, soap_vector)
    element : str
        Element symbol to cluster (e.g., 'H', 'C', 'N', 'O')
    target_pca_components : int, default=80
        Target number of PCA components. Actual value will be
        min(target, n_features, n_samples).
    min_cluster_size : int, default=100
        Minimum number of atoms in a cluster (HDBSCAN parameter)
    min_samples : int or None, default=None
        Minimum samples for core points (HDBSCAN parameter)
        If None, defaults to min_cluster_size
    cluster_selection_epsilon : float, default=0.0
        Distance threshold for merging clusters (HDBSCAN parameter)
    cluster_selection_method : str, default='eom'
        HDBSCAN cluster selection method: 'eom' or 'leaf'
    core_dist_n_jobs : int, default=-1
        Number of parallel jobs for HDBSCAN (-1 = all CPUs)
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    labels : np.ndarray
        Cluster label for each atom (same order as atom_info[element]).
        -1 indicates noise points.
    stats : dict
        Statistics dictionary with keys:
        - n_atoms: total atoms
        - n_clusters: number of clusters found
        - n_noise: number of noise points
        - pca_components: number of PCA components used
        - variance_retained: % variance retained by PCA
        - cluster_sizes: dict mapping cluster_id -> count
        - atom_data: list of (mol_idx, atom_idx, soap_vec) tuples
    pca : PCA
        Fitted PCA object (can be used to transform new data)
    clusterer : hdbscan.HDBSCAN
        Fitted HDBSCAN clusterer (useful to plot condensed tree, access
        cluster probabilities, etc.)
        
    Examples
    --------
    >>> # Cluster carbon atoms with custom parameters
    >>> labels, stats, pca = cluster_single_element_environments(
    ...     atom_info, 'C',
    ...     target_pca_components=100,
    ...     min_cluster_size=50,
    ...     min_samples=25
    ... )
    >>> print(f"Found {stats['n_clusters']} clusters for C")
    """
    if element not in atom_info or len(atom_info[element]) == 0:
        raise ValueError(f"No atoms found for element '{element}'")
    
    atom_data = atom_info[element]
    X = np.array([item[2] for item in atom_data])
    n_atoms = X.shape[0]
    n_features = X.shape[1]
    
    print(f"\n{'='*70}")
    print(f"CLUSTERING {element} ATOMIC ENVIRONMENTS")
    print(f"{'='*70}")
    print(f"\nInput:")
    print(f"  Total atoms: {n_atoms:,}")
    print(f"  SOAP features: {n_features}")
    
    # PCA dimensionality reduction
    n_components = min(target_pca_components, n_features, n_atoms)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    variance_retained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"\nPCA:")
    print(f"  {n_features} -> {n_components} components")
    print(f"  Variance retained: {variance_retained:.2f}%")
    
    # L2 normalize so Euclidean distance behaves like cosine similarity
    X_pca = normalize(X_pca, norm='l2')
    print(f"  Applied L2 normalization (Euclidean → cosine similarity)")
    
    # HDBSCAN clustering
    if min_samples is None:
        min_samples_actual = min_cluster_size
    else:
        min_samples_actual = min_samples
    
    print(f"\nHDBSCAN parameters:")
    print(f"  min_cluster_size: {min_cluster_size}")
    print(f"  min_samples: {min_samples_actual}")
    print(f"  cluster_selection_epsilon: {cluster_selection_epsilon}")
    print(f"  cluster_selection_method: {cluster_selection_method}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples_actual,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='euclidean',
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=core_dist_n_jobs
    )
    labels = clusterer.fit_predict(X_pca)
    
    # Count clusters and noise
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l != -1])
    n_noise = np.sum(labels == -1)
    
    print(f"\nResults:")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise:,} ({100*n_noise/n_atoms:.2f}%)")
    
    # Store cluster sizes
    cluster_sizes = {}
    for label in unique_labels:
        cluster_sizes[label] = int(np.sum(labels == label))
    
    # Show top 10 clusters
    sorted_clusters = sorted([(k, v) for k, v in cluster_sizes.items() if k != -1], 
                            key=lambda x: -x[1])
    if sorted_clusters:
        top_10 = sorted_clusters[:10]
        print(f"  Top 10 clusters:")
        for k, v in top_10:
            print(f"    Cluster {k}: {v:,} atoms ({100*v/n_atoms:.2f}%)")
    
    # Compile statistics
    stats = {
        'element': element,
        'n_atoms': n_atoms,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'pca_components': n_components,
        'variance_retained': variance_retained,
        'cluster_sizes': cluster_sizes,
        'atom_data': atom_data  # Keep for creating dataframe mappings
    }
    
    print(f"{'='*70}\n")
    
    return labels, stats, pca, clusterer


def cluster_all_atomic_environments(
    df: pd.DataFrame,
    atom_info: Dict[str, List[Tuple]],
    precomputed_clusters: Optional[Dict[str, Tuple[np.ndarray, Dict, PCA]]] = None,
    element_params: Optional[Dict[str, Dict]] = None,
    default_pca_components: int = 80,
    default_min_cluster_size: int = 100,
    default_min_samples: Optional[int] = None,
    default_cluster_selection_epsilon: float = 0.0,
    default_cluster_selection_method: str = 'eom',
    core_dist_n_jobs: int = -1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Cluster atomic environments for all species in the dataset, with element-specific parameters.
    
    This function can either:
    1. Use pre-computed clustering results from cluster_single_element_environments (recommended
       after tuning hyperparameters), OR
    2. Compute clustering from scratch using element_params
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with molecules (must have 'atom' and 'soap_descriptors' columns)
    atom_info : dict
        Dictionary mapping element type to list of tuples:
        (molecule_idx, atom_idx, soap_vector)
    precomputed_clusters : dict or None, default=None
        Optional dictionary mapping element -> (labels, stats, pca) tuple from
        cluster_single_element_environments. If provided, uses these results
        instead of computing new clusters.
        Example: {'C': (c_labels, c_stats, c_pca), 'H': (h_labels, h_stats, h_pca)}
    element_params : dict or None, default=None
        Optional dictionary mapping element -> parameter dict.
        Only used if precomputed_clusters is None.
        Each parameter dict can contain:
        - target_pca_components: int
        - min_cluster_size: int
        - min_samples: int or None
        - cluster_selection_epsilon: float
        - cluster_selection_method: str
        Example: {'C': {'min_cluster_size': 50}, 'H': {'min_cluster_size': 200}}
    default_pca_components : int, default=80
        Default PCA components for elements not in element_params
    default_min_cluster_size : int, default=100
        Default min_cluster_size for elements not in element_params
    default_min_samples : int or None, default=None
        Default min_samples for elements not in element_params
    default_cluster_selection_epsilon : float, default=0.0
        Default cluster_selection_epsilon for elements not in element_params
    default_cluster_selection_method : str, default='eom'
        Default cluster_selection_method for elements not in element_params
    core_dist_n_jobs : int, default=-1
        Number of parallel jobs for HDBSCAN (-1 = all CPUs)
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    df_with_clusters : pd.DataFrame
        Input dataframe with added 'atom_cluster_labels' column.
        Each entry is an array of cluster labels (one per atom in molecule).
        Cluster labels are species-specific (e.g., H_0, C_5, N_2, O_1).
        
    Examples
    --------
    >>> # Option 1: Use pre-computed clusters (after hyperparameter tuning)
    >>> c_labels, c_stats, c_pca = cluster_single_element_environments(atom_info, 'C', ...)
    >>> h_labels, h_stats, h_pca = cluster_single_element_environments(atom_info, 'H', ...)
    >>> n_labels, n_stats, n_pca = cluster_single_element_environments(atom_info, 'N', ...)
    >>> o_labels, o_stats, o_pca = cluster_single_element_environments(atom_info, 'O', ...)
    >>> precomputed = {'C': (c_labels, c_stats, c_pca), 'H': (h_labels, h_stats, h_pca),
    ...                'N': (n_labels, n_stats, n_pca), 'O': (o_labels, o_stats, o_pca)}
    >>> df_clustered = cluster_all_atomic_environments(df, atom_info, precomputed_clusters=precomputed)
    
    >>> # Option 2: Compute from scratch with default parameters
    >>> df_clustered = cluster_all_atomic_environments(df, atom_info)
    
    >>> # Option 3: Compute from scratch with custom parameters per element
    >>> element_params = {
    ...     'H': {'min_cluster_size': 200, 'target_pca_components': 50},
    ...     'C': {'min_cluster_size': 50, 'min_samples': 25},
    ...     'N': {'min_cluster_size': 30},
    ...     'O': {'min_cluster_size': 40}
    ... }
    >>> df_clustered = cluster_all_atomic_environments(df, atom_info, element_params=element_params)
    """
    df_copy = df.copy()
    
    # Dictionary to store cluster labels: (mol_idx, atom_idx) -> cluster_label_str
    atom_cluster_map = {}
    
    # Statistics for summary
    all_stats = {}
    
    print(f"\n{'='*70}")
    print(f"CLUSTERING ATOMIC ENVIRONMENTS BY SPECIES")
    print(f"{'='*70}\n")
    
    # Check if using precomputed clusters
    use_precomputed = precomputed_clusters is not None
    
    if use_precomputed:
        print(f"Using pre-computed clustering results for:")
        for elem in precomputed_clusters.keys():
            print(f"  {elem}")
        print()
        
        # Process precomputed results
        for element in SOAP_PARAMS['species']:
            if element not in precomputed_clusters:
                print(f"⚠ Skipping {element}: no pre-computed results provided\n")
                continue
            
            labels, stats, pca = precomputed_clusters[element]
            all_stats[element] = stats
            
            # Map each atom to its cluster label
            atom_data = stats['atom_data']
            for idx, (mol_idx, atom_idx, _) in enumerate(atom_data):
                cluster_label = labels[idx]
                # Store as "Element_ClusterNum" (e.g., "C_5", "H_-1")
                atom_cluster_map[(mol_idx, atom_idx)] = f"{element}_{cluster_label}"
    else:
        # Compute clusters from scratch
        if element_params is not None:
            print(f"Using element-specific parameters:")
            for elem, params in element_params.items():
                print(f"  {elem}: {params}")
            print()
        
        # Process each element separately
        for element in SOAP_PARAMS['species']:
            if element not in atom_info or len(atom_info[element]) == 0:
                print(f"⚠ Skipping {element}: no atoms found\n")
                continue
            
            # Get parameters for this element
            if element_params is not None and element in element_params:
                params = element_params[element]
                pca_comp = params.get('target_pca_components', default_pca_components)
                min_clust = params.get('min_cluster_size', default_min_cluster_size)
                min_samp = params.get('min_samples', default_min_samples)
                clust_eps = params.get('cluster_selection_epsilon', default_cluster_selection_epsilon)
                clust_method = params.get('cluster_selection_method', default_cluster_selection_method)
            else:
                pca_comp = default_pca_components
                min_clust = default_min_cluster_size
                min_samp = default_min_samples
                clust_eps = default_cluster_selection_epsilon
                clust_method = default_cluster_selection_method
            
            # Cluster this element
            labels, stats, pca = cluster_single_element_environments(
                atom_info,
                element,
                target_pca_components=pca_comp,
                min_cluster_size=min_clust,
                min_samples=min_samp,
                cluster_selection_epsilon=clust_eps,
                cluster_selection_method=clust_method,
                core_dist_n_jobs=core_dist_n_jobs,
                random_state=random_state
            )
            
            all_stats[element] = stats
            
            # Map each atom to its cluster label
            atom_data = stats['atom_data']
            for idx, (mol_idx, atom_idx, _) in enumerate(atom_data):
                cluster_label = labels[idx]
                # Store as "Element_ClusterNum" (e.g., "C_5", "H_-1")
                atom_cluster_map[(mol_idx, atom_idx)] = f"{element}_{cluster_label}"
    
    # Add cluster labels to dataframe
    print(f"{'='*70}")
    print(f"Adding cluster labels to dataframe...")
    print(f"{'='*70}\n")
    
    cluster_label_arrays = []
    for mol_idx, row in df_copy.iterrows():
        atoms = row['atom']
        n_atoms_in_mol = len(atoms)
        
        # Build array of cluster labels for this molecule
        labels_array = []
        for atom_idx in range(n_atoms_in_mol):
            label = atom_cluster_map.get((mol_idx, atom_idx), 'Unknown')
            labels_array.append(label)
        
        cluster_label_arrays.append(labels_array)
    
    df_copy['atom_cluster_labels'] = cluster_label_arrays
    
    # Print summary
    print(f"Summary by element:")
    print(f"  {'Element':<8} {'Atoms':>10} {'PCA':>6} {'Variance':>9} {'Clusters':>9} {'Noise':>10}")
    print(f"  {'-'*60}")
    for elem in SOAP_PARAMS['species']:
        if elem in all_stats:
            stats = all_stats[elem]
            print(f"  {elem:<8} {stats['n_atoms']:>10,} {stats['pca_components']:>6} "
                  f"{stats['variance_retained']:>8.1f}% {stats['n_clusters']:>9} "
                  f"{stats['n_noise']:>10,} ({100*stats['n_noise']/stats['n_atoms']:>4.1f}%)")
    
    print(f"\n✓ Added 'atom_cluster_labels' column to dataframe")
    print(f"  Each entry is an array of cluster labels (one per atom)")
    print(f"  Format: 'Element_ClusterNum' (e.g., 'C_5', 'H_-1')")
    print(f"  Label -1 indicates noise (not assigned to any cluster)\n")
    
    return df_copy


def visualize_element_umap(
    stats: Dict,
    pca: PCA,
    labels: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    ax=None,
    cmap: str = 'tab20',
    s: int = 8
) -> np.ndarray:
    """
    Create a 2D UMAP visualization for a clustered element.

    Parameters
    ----------
    stats : dict
        The statistics dict returned by `cluster_single_element_environments`.
    pca : PCA
        Fitted PCA used to reduce the SOAP vectors.
    labels : np.ndarray
        Cluster labels for each atom (same order as stats['atom_data']).
    n_neighbors, min_dist, random_state : UMAP params
    ax : matplotlib.axes.Axes or None
        Axes to draw on (if None, creates a new figure)
    cmap : str
        Matplotlib colormap name
    s : int
        Marker size

    Returns
    -------
    embedding : np.ndarray
        2D UMAP embedding (n_samples, 2)
    """
    try:
        import umap
    except Exception as e:
        raise ImportError("UMAP is required for visualize_element_umap. Install via 'pip install umap-learn'.")

    X = np.array([item[2] for item in stats['atom_data']])
    X_pca = pca.transform(X)
    X_pca = normalize(X_pca, norm='l2')

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding = reducer.fit_transform(X_pca)

    import matplotlib.pyplot as _plt
    if ax is None:
        fig, ax = _plt.subplots(figsize=(6, 5))

    unique = np.unique(labels)
    cmap_obj = cm.get_cmap(cmap, len(unique))

    for i, lab in enumerate(unique):
        mask = labels == lab
        if lab == -1:
            col = '#7f7f7f'
        else:
            col = cmap_obj(i)
        ax.scatter(embedding[mask, 0], embedding[mask, 1], c=[col]*np.sum(mask), s=s, label=str(lab), alpha=0.8)

    ax.set_title(f"UMAP (n={len(labels):,})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    # Only show legend if not too many clusters
    if len(unique) <= 20:
        ax.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    else:
        ax.text(0.98, 0.98, f'{len([u for u in unique if u != -1])} clusters', 
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return embedding


def visualize_element_tsne(
    stats: Dict,
    pca: PCA,
    labels: np.ndarray,
    perplexity: Optional[float] = None,
    random_state: int = 42,
    ax=None,
    cmap: str = 'tab20',
    s: int = 8
) -> np.ndarray:
    """
    Create a 2D t-SNE visualization for a clustered element.

    Parameters
    ----------
    stats, pca, labels : same as for UMAP
    perplexity : float or None
        If None, set automatically to min(30, max(5, n_atoms//3)). Must be < n_samples.
    Returns
    -------
    embedding : np.ndarray
        2D t-SNE embedding
    """
    from sklearn.manifold import TSNE

    X = np.array([item[2] for item in stats['atom_data']])
    X_pca = pca.transform(X)
    X_pca = normalize(X_pca, norm='l2')

    n_samples = X_pca.shape[0]
    if n_samples < 5:
        raise ValueError("Not enough samples for t-SNE (need >=5)")

    if perplexity is None:
        cand = max(5, n_samples // 3)
        perplexity = float(min(30, cand))
        # ensure perplexity < n_samples
        if perplexity >= n_samples:
            perplexity = float(max(1, n_samples - 1))

    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=random_state)
    embedding = tsne.fit_transform(X_pca)

    import matplotlib.pyplot as _plt
    if ax is None:
        fig, ax = _plt.subplots(figsize=(6, 5))

    unique = np.unique(labels)
    cmap_obj = cm.get_cmap(cmap, len(unique))

    for i, lab in enumerate(unique):
        mask = labels == lab
        if lab == -1:
            col = '#7f7f7f'
        else:
            col = cmap_obj(i)
        ax.scatter(embedding[mask, 0], embedding[mask, 1], c=[col]*np.sum(mask), s=s, label=str(lab), alpha=0.8)

    ax.set_title(f"t-SNE (n={n_samples:,}, perplexity={perplexity})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    # Only show legend if not too many clusters
    if len(unique) <= 20:
        ax.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    else:
        ax.text(0.98, 0.98, f'{len([u for u in unique if u != -1])} clusters', 
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return embedding


def visualize_element_all(stats: Dict, pca: PCA, labels: np.ndarray):
    """
    Convenience wrapper to create UMAP and t-SNE figures for an element.
    Returns a dict with embeddings and axes.
    """
    import matplotlib.pyplot as _plt

    results = {}

    # UMAP
    fig1 = _plt.figure(figsize=(6, 5))
    ax1 = fig1.gca()
    emb_umap = visualize_element_umap(stats, pca, labels, ax=ax1)
    results['umap'] = {'embedding': emb_umap, 'ax': ax1}

    # t-SNE
    fig2 = _plt.figure(figsize=(6, 5))
    ax2 = fig2.gca()
    emb_tsne = visualize_element_tsne(stats, pca, labels, ax=ax2)
    results['tsne'] = {'embedding': emb_tsne, 'ax': ax2}

    return results

