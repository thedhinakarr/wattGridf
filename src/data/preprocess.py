#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script handles data preprocessing and cleaning for the electricity price forecasting project.
"""

import os
import pandas as pd
import numpy as np


def remove_outliers(df, column='DollarsPerMegawattHour', min_value=0, max_value=5000):
    """
    Remove extreme outliers from the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name to check for outliers
    min_value : float
        Minimum allowed value
    max_value : float
        Maximum allowed value

    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers removed
    """
    original_count = len(df)
    df = df[(df[column] >= min_value) & (df[column] <= max_value)]
    removed_count = original_count - len(df)

    print(f"Removed {removed_count} outliers ({removed_count/original_count*100:.2f}% of data)")
    print(f"Remaining rows: {len(df)}")

    return df


def fill_missing_values(df):
    """
    Fill missing values in the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled
    """
    # Print missing values before handling
    print("\nMissing Values Before Handling:")
    print(df.isnull().sum())

    # Fill missing values for categorical columns
    if "Island" in df.columns:
        df["Island"].fillna("Unknown", inplace=True)

    # For time series data, we can use forward fill followed by backward fill
    # This assumes we have already sorted the data by time
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    # Confirm missing values handled
    print("\nMissing Values After Handling:")
    print(df.isnull().sum())

    return df


def create_features(df):
    """
    Create features for the forecasting models.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame with additional features
    """
    print("Creating features...")

    # Create a copy to avoid modifying the original DataFrame
    featured_df = df.copy()

    # Rolling statistics (with 7-day window)
    if "DollarsPerMegawattHour" in featured_df.columns:
        featured_df["RollingMean_7"] = featured_df.groupby(["PointOfConnection"])["DollarsPerMegawattHour"].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        featured_df["RollingStd_7"] = featured_df.groupby(["PointOfConnection"])["DollarsPerMegawattHour"].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )

        # Price volatility
        featured_df["PriceVolatility"] = featured_df["RollingStd_7"] / featured_df["RollingMean_7"]

        # Lag features
        featured_df["Lag_1"] = featured_df.groupby(["PointOfConnection"])["DollarsPerMegawattHour"].shift(1)
        featured_df["Lag_7"] = featured_df.groupby(["PointOfConnection"])["DollarsPerMegawattHour"].shift(7)

        # Lag features for rolling statistics
        featured_df["Lag_RollingMean_1"] = featured_df.groupby(["PointOfConnection"])["RollingMean_7"].shift(1)

    # Temporal features
    if "TradingDate" in featured_df.columns:
        # Day of week
        featured_df["DayOfWeek"] = featured_df["TradingDate"].dt.dayofweek

        # Hour of day (from trading period)
        if "TradingPeriod" in featured_df.columns:
            featured_df["HourOfDay"] = ((featured_df["TradingPeriod"] - 1) / 2).astype(int)

            # Is peak hour (7-9 AM or 5-8 PM)
            featured_df["IsPeakHour"] = (
                ((featured_df["HourOfDay"] >= 7) & (featured_df["HourOfDay"] <= 9)) |
                ((featured_df["HourOfDay"] >= 17) & (featured_df["HourOfDay"] <= 20))
            ).astype(int)

        # Month
        featured_df["Month"] = featured_df["TradingDate"].dt.month

        # Is weekend
        featured_df["IsWeekend"] = (featured_df["DayOfWeek"] >= 5).astype(int)

    # Drop rows with NaN values
    original_count = len(featured_df)
    featured_df.dropna(inplace=True)
    print(f"Dropped {original_count - len(featured_df)} rows with NaN values")

    print(f"Created {len(featured_df.columns) - len(df.columns)} new features")
    print(f"Final dataset shape: {featured_df.shape}")

    return featured_df


def main(input_path, output_path):
    """
    Main function to preprocess the raw dataset and create features.

    Parameters:
    -----------
    input_path : str
        Path to the input dataset
    output_path : str
        Path where to save the processed dataset
    """
    # Load the data
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    # Convert date columns if needed
    if "TradingDate" in df.columns and not pd.api.types.is_datetime64_dtype(df["TradingDate"]):
        df["TradingDate"] = pd.to_datetime(df["TradingDate"])
    if "PublishDateTime" in df.columns and not pd.api.types.is_datetime64_dtype(df["PublishDateTime"]):
        df["PublishDateTime"] = pd.to_datetime(df["PublishDateTime"])

    # Remove outliers
    df = remove_outliers(df)

    # Fill missing values
    df = fill_missing_values(df)

    # Create features
    featured_df = create_features(df)

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    featured_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    # Define paths - change these to match your project structure
    input_path = "../../data/processed/cleaned_data.csv"
    output_path = "../../data/processed/featured_data.csv"

    main(input_path, output_path)
