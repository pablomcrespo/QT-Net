import pandas as pd
from collections import defaultdict


def find_molecules_with_cluster(df, element, label):
    """
    Find all molecules that contain a specific cluster label.
    
    Args:
        df: DataFrame with 'molecule_cluster_labels' column
        element: Element symbol (e.g., 'H', 'C', 'N', 'O')
        label: Full label string (e.g., 'H_10') or just the number (e.g., 10)
    
    Returns:
        DataFrame subset of molecules containing this cluster
    """
    # Handle both 'H_10' and 10 as input
    if isinstance(label, int):
        target_label = f"{element}_{label}"
    else:
        target_label = label
    
    # Filter molecules that have this label
    mask = df['molecule_cluster_labels'].apply(
        lambda x: element in x and target_label in x[element]
    )
    return df[mask]


def find_molecules_with_multiple_clusters(df, cluster_dict):
    """
    Find molecules containing ALL specified clusters.
    
    Args:
        df: DataFrame with 'molecule_cluster_labels' column
        cluster_dict: Dict like {'H': [10, 11], 'C': [5]} specifying required clusters
    
    Returns:
        DataFrame subset of molecules containing all specified clusters
    """
    mask = pd.Series([True] * len(df), index=df.index)
    
    for element, labels in cluster_dict.items():
        for label in labels:
            target_label = f"{element}_{label}"
            mask &= df['molecule_cluster_labels'].apply(
                lambda x: element in x and target_label in x[element]
            )
    
    return df[mask]



def analyze_cluster_cooccurrence(df, element, label):
    """
    Analyze which other clusters co-occur with a specific cluster label.
    
    Args:
        df: DataFrame with 'molecule_cluster_labels' column
        element: Element symbol (e.g., 'H', 'C')
        label: Label number (e.g., 10)
    
    Returns:
        Dictionary with co-occurrence counts for each element
    """
    target_label = f"{element}_{label}"
    
    # Get molecules containing this cluster
    molecules = find_molecules_with_cluster(df, element, label)
    
    # Count co-occurrences
    cooccurrence = defaultdict(lambda: defaultdict(int))
    
    for idx, row in molecules.iterrows():
        cluster_labels = row['molecule_cluster_labels']
        for elem, labels in cluster_labels.items():
            for lab in labels:
                cooccurrence[elem][lab] += 1
    
    return dict(cooccurrence)