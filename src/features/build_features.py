#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script creates DTW-based features and other advanced features
for the electricity price forecasting models.
"""

import os
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def create_pivot_table(df):
    """
    Create a pivot table with POCs as columns and time indices as rows.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with electricity price data

    Returns:
    --------
    pd.DataFrame
        Pivot table with POCs as columns
    """
    print("Creating pivot table...")

    # Create a time index by combining TradingDate and TradingPeriod
    df['TimeIndex'] = df['TradingDate'].dt.strftime('%Y-%m-%d') + '_' + df['TradingPeriod'].astype(str)

    # Pivot the DataFrame
    pivot_df = df.pivot(
        index='TimeIndex',
        columns='PointOfConnection',
        values='DollarsPerMegawattHour'
    )

    # Sort the pivoted DataFrame by the time index
    pivot_df.sort_index(inplace=True)

    # Handle missing values by forward filling then back filling
    pivot_df = pivot_df.fillna(method='ffill').fillna(method='bfill')

    print(f"Created pivot table with {pivot_df.shape[1]} POCs and {pivot_df.shape[0]} time points")

    return pivot_df


def compute_dtw_distances(pivot_df, window_size=24, max_pocs=None):
    """
    Compute DTW distances between all pairs of POCs.

    Parameters:
    -----------
    pivot_df : pd.DataFrame
        Pivot table with POCs as columns
    window_size : int
        Window size constraint for FastDTW algorithm
    max_pocs : int
        Maximum number of POCs to process (for memory constraints)

    Returns:
    --------
    np.ndarray
        Distance matrix with DTW distances
    list
        List of POC names
    """
    print("Computing DTW distances...")

    # Get POC names
    poc_names = list(pivot_df.columns)

    # Limit the number of POCs if specified
    if max_pocs is not None and max_pocs < len(poc_names):
        print(f"Limiting to {max_pocs} POCs due to memory constraints")
        poc_names = poc_names[:max_pocs]

    num_pocs = len(poc_names)
    distance_matrix = np.zeros((num_pocs, num_pocs))

    # Define a simple custom distance function: absolute difference between scalars
    distance_func = lambda x, y: abs(x - y)

    # Compute DTW distances
    total_comparisons = (num_pocs * (num_pocs - 1)) // 2
    comparison_count = 0

    for i in range(num_pocs):
        ts_i = pivot_df[poc_names[i]].values

        # Display progress every 10%
        progress_interval = max(1, total_comparisons // 10)

        for j in range(i, num_pocs):
            if i == j:
                distance_matrix[i, j] = 0.0
                continue

            ts_j = pivot_df[poc_names[j]].values
            distance, _ = fastdtw(ts_i, ts_j, dist=distance_func, radius=window_size)

            # DTW is symmetric
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

            comparison_count += 1
            if comparison_count % progress_interval == 0:
                progress = comparison_count / total_comparisons * 100
                print(f"Progress: {progress:.1f}% ({comparison_count}/{total_comparisons})")

    print("DTW distance computation complete")

    return distance_matrix, poc_names


def create_adjacency_matrix(distance_matrix, poc_names):
    """
    Create an adjacency matrix using a Gaussian kernel transformation.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        Distance matrix with DTW distances
    poc_names : list
        List of POC names

    Returns:
    --------
    pd.DataFrame
        Adjacency matrix as a DataFrame
    """
    print("Creating adjacency matrix...")

    # Calculate sigma as the standard deviation of the distances
    sigma = np.std(distance_matrix)

    # Apply the Gaussian (RBF) kernel transformation
    adjacency_matrix = np.exp(-(distance_matrix**2) / (2 * sigma**2))

    # Convert to DataFrame for easier interpretation
    adjacency_df = pd.DataFrame(adjacency_matrix, index=poc_names, columns=poc_names)

    print("Adjacency matrix creation complete")

    return adjacency_df


def extract_static_node_features(df, poc_list):
    """
    Extract static node features for each POC.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with electricity price data
    poc_list : list
        List of POC names

    Returns:
    --------
    pd.DataFrame
        DataFrame with static features for each POC
    """
    print("Extracting static node features...")

    features = []

    for poc in poc_list:
        poc_data = df[df['PointOfConnection'] == poc]

        if len(poc_data) == 0:
            print(f"No data found for POC: {poc}")
            continue

        # Calculate basic statistics
        mean_price = poc_data['DollarsPerMegawattHour'].mean()
        std_price = poc_data['DollarsPerMegawattHour'].std()
        min_price = poc_data['DollarsPerMegawattHour'].min()
        max_price = poc_data['DollarsPerMegawattHour'].max()

        # Calculate volatility
        volatility = std_price / mean_price if mean_price > 0 else 0

        # Get island
        island = poc_data['Island'].iloc[0] if 'Island' in poc_data.columns else "Unknown"

        # Create one-hot encoding for island
        is_north = 1 if island == "North" else 0
        is_south = 1 if island == "South" else 0

        # Calculate peak hour statistics if available
        peak_ratio = None
        if 'IsPeakHour' in poc_data.columns:
            peak_data = poc_data[poc_data['IsPeakHour'] == 1]
            offpeak_data = poc_data[poc_data['IsPeakHour'] == 0]

            if len(peak_data) > 0 and len(offpeak_data) > 0:
                peak_mean = peak_data['DollarsPerMegawattHour'].mean()
                offpeak_mean = offpeak_data['DollarsPerMegawattHour'].mean()
                peak_ratio = peak_mean / offpeak_mean if offpeak_mean > 0 else 1

        # Store features
        features.append({
            'PointOfConnection': poc,
            'MeanPrice': mean_price,
            'StdPrice': std_price,
            'MinPrice': min_price,
            'MaxPrice': max_price,
            'Volatility': volatility,
            'Island': island,
            'IsNorth': is_north,
            'IsSouth': is_south,
            'PeakRatio': peak_ratio
        })

    # Create DataFrame
    node_features_df = pd.DataFrame(features)

    # Fill missing values
    if 'PeakRatio' in node_features_df.columns:
        node_features_df['PeakRatio'].fillna(1, inplace=True)

    print(f"Extracted static features for {len(node_features_df)} POCs")

    return node_features_df


def create_adjacency_list(adjacency_df, threshold=0.7):
    """
    Create an adjacency list for the graph model.

    Parameters:
    -----------
    adjacency_df : pd.DataFrame
        Adjacency matrix as a DataFrame
    threshold : float
        Threshold for edge creation

    Returns:
    --------
    list
        List of edges (pairs of POCs)
    list
        List of edge weights
    """
    print(f"Creating adjacency list with threshold {threshold}...")

    edges = []
    weights = []

    # Convert similarity to adjacency list
    for i, poc1 in enumerate(adjacency_df.index):
        for j, poc2 in enumerate(adjacency_df.columns):
            similarity = adjacency_df.loc[poc1, poc2]

            # Add edge if similarity is above threshold
            if i != j and similarity >= threshold:
                edges.append((poc1, poc2))
                weights.append(similarity)

    print(f"Created {len(edges)} edges")

    return edges, weights


def save_features(adjacency_df, node_features_df, edges, weights,
                  adjacency_path, node_features_path, edge_list_path):
    """
    Save the created features to disk.

    Parameters:
    -----------
    adjacency_df : pd.DataFrame
        Adjacency matrix as a DataFrame
    node_features_df : pd.DataFrame
        DataFrame with static features for each POC
    edges : list
        List of edges (pairs of POCs)
    weights : list
        List of edge weights
    adjacency_path : str
        Path to save the adjacency matrix
    node_features_path : str
        Path to save the node features
    edge_list_path : str
        Path to save the edge list
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(adjacency_path), exist_ok=True)
    os.makedirs(os.path.dirname(node_features_path), exist_ok=True)
    os.makedirs(os.path.dirname(edge_list_path), exist_ok=True)

    # Save adjacency matrix
    adjacency_df.to_csv(adjacency_path)
    print(f"Adjacency matrix saved to {adjacency_path}")

    # Save node features
    node_features_df.to_csv(node_features_path, index=False)
    print(f"Node features saved to {node_features_path}")

    # Save edge list
    edge_df = pd.DataFrame({
        'source': [e[0] for e in edges],
        'target': [e[1] for e in edges],
        'weight': weights
    })
    edge_df.to_csv(edge_list_path, index=False)
    print(f"Edge list saved to {edge_list_path}")


def main(input_path, output_dir, window_size=24, similarity_threshold=0.7, max_pocs=None):
    """
    Main function to create graph features from electricity price data.

    Parameters:
    -----------
    input_path : str
        Path to the input dataset
    output_dir : str
        Directory to save the outputs
    window_size : int
        Window size constraint for FastDTW algorithm
    similarity_threshold : float
        Threshold for edge creation
    max_pocs : int
        Maximum number of POCs to process (for memory constraints)
    """
    # Load the data
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    # Convert date columns if needed
    if "TradingDate" in df.columns and not pd.api.types.is_datetime64_dtype(df["TradingDate"]):
        df["TradingDate"] = pd.to_datetime(df["TradingDate"])

    # Create pivot table
    pivot_df = create_pivot_table(df)

    # Compute DTW distances
    distance_matrix, poc_names = compute_dtw_distances(pivot_df, window_size, max_pocs)

    # Create adjacency matrix
    adjacency_df = create_adjacency_matrix(distance_matrix, poc_names)

    # Extract static node features
    node_features_df = extract_static_node_features(df, poc_names)

    # Create adjacency list
    edges, weights = create_adjacency_list(adjacency_df, similarity_threshold)

    # Define output paths
    adjacency_path = os.path.join(output_dir, "dtw_adjacency_matrix.csv")
    node_features_path = os.path.join(output_dir, "node_features.csv")
    edge_list_path = os.path.join(output_dir, "edge_list.csv")

    # Save features
    save_features(adjacency_df, node_features_df, edges, weights,
                  adjacency_path, node_features_path, edge_list_path)


if __name__ == "__main__":
    # Define paths - change these to match your project structure
    input_path = "../../data/processed/featured_data.csv"
    output_dir = "../../data/processed"

    # Define parameters
    window_size = 24  # 12 hours with half-hourly data
    similarity_threshold = 0.7
    max_pocs = None  # Set to a number (e.g., 100) if memory is limited

    main(input_path, output_dir, window_size, similarity_threshold, max_pocs)
