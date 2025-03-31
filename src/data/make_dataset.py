#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script loads the raw data from the electricity market and
creates the dataset for the project.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime


def load_raw_data(raw_data_path):
    """
    Load all CSV files from the raw data directory and combine them into a single DataFrame.

    Parameters:
    -----------
    raw_data_path : str
        Path to the directory containing raw data files

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all data
    """
    print("Looking for data files...")
    all_files = glob.glob(os.path.join(raw_data_path, "**/*.csv"), recursive=True)
    total_files = len(all_files)

    if total_files == 0:
        print("No CSV files found! Check the path.")
        return None

    print(f"Found {total_files} CSV files to process.")

    # Read files with progress tracking
    df_list = []
    for i, file in enumerate(all_files):
        if i % 10 == 0:  # Show progress update every 10 files
            print(f"Processing file {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%)")
        df_list.append(pd.read_csv(file))

    # Combine all dataframes into one
    df = pd.concat(df_list, ignore_index=True)
    print(f"Data Loaded: {df.shape[0]:,} rows and {df.shape[1]} columns")

    return df


def convert_date_columns(df):
    """
    Convert date columns to the appropriate datetime format.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with date columns

    Returns:
    --------
    pd.DataFrame
        DataFrame with converted date columns
    """
    # Convert date columns to datetime format
    df['TradingDate'] = pd.to_datetime(df['TradingDate'])
    if 'PublishDateTime' in df.columns:
        df['PublishDateTime'] = pd.to_datetime(df['PublishDateTime'])

    return df


def check_data_quality(df):
    """
    Check data quality and print summary statistics.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to check

    Returns:
    --------
    pd.DataFrame
        DataFrame with missing value statistics
    """
    # Display basic information about the dataset
    print("\nData Structure Overview:")
    print(df.info())

    # Generate statistical summary
    print("\nStatistical Summary:")
    stats = df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    stats['range'] = stats['max'] - stats['min']  # Add range column
    print(stats)

    # Check for missing values
    print("\nMissing Values Analysis:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    })

    return missing_df[missing_df['Missing Values'] > 0]


def save_dataset(df, output_path):
    """
    Save the processed dataset to a CSV file.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    output_path : str
        Path where to save the dataset
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")


def main(raw_data_path, output_path):
    """
    Main function to load raw data, process it, and save the processed dataset.

    Parameters:
    -----------
    raw_data_path : str
        Path to the directory containing raw data files
    output_path : str
        Path where to save the processed dataset
    """
    # Load the raw data
    df = load_raw_data(raw_data_path)
    if df is None:
        return

    # Convert date columns
    df = convert_date_columns(df)

    # Check data quality
    missing_df = check_data_quality(df)
    print(missing_df)

    # Save the dataset
    save_dataset(df, output_path)


if __name__ == "__main__":
    # Define paths - change these to match your project structure
    raw_data_path = "../../data/raw"
    output_path = "../../data/processed/cleaned_data.csv"

    main(raw_data_path, output_path)
