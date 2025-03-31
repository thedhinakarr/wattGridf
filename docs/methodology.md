# Methodology

This document outlines the methodology used in the WattGridf project, following the CRISP-DM (CRoss Industry Standard Process for Data Mining) framework.

## CRISP-DM Process

The project follows the six phases of the CRISP-DM process:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

## 1. Business Understanding

### Objectives

The primary objective is to develop an accurate and explainable forecasting model for electricity prices in the New Zealand wholesale market. Explainability is crucial for grid operators who need to understand not just future prices, but why these changes occur and how they propagate through the grid.

### Key Stakeholders

- Grid operators
- Electricity retailers
- Industrial consumers
- Market regulators

### Business Challenges

- Complex spatial-temporal dependencies between POCs
- Lack of explicit grid topology data
- Need for explainable forecasts for operational decision-making
- Occurrence of price spikes and proxy pricing events

### Success Criteria

- 10-15% reduction in RMSE compared to baseline models
- Enhanced interpretability through attention mechanisms and feature importance
- Ability to identify and explain anomalous price events

## 2. Data Understanding

### Data Sources

The project uses New Zealand wholesale electricity market dispatch prices from 2022-2025, sourced from the Electricity Authority's EMI database.

### Exploratory Data Analysis

EDA was conducted in `1_eda_FNAL.ipynb` to understand:

- Price distributions and statistics
- Temporal patterns (hourly, daily, weekly, seasonal)
- Spatial variations (North vs South Island, POC-specific)
- Price anomalies and volatility
- Data quality issues

### Key Insights

- Prices exhibit high volatility with occasional extreme spikes
- Distinct daily patterns with peak prices in morning (7-9am) and evening (5-8pm)
- Significant differences between North and South Islands
- Seasonal patterns with higher prices and volatility in winter months
- Certain POCs consistently show higher volatility than others

## 3. Data Preparation

### Data Cleaning

Data cleaning was performed in `2_preprocessing_FINAL.ipynb`:

- Removal of extreme outliers (prices >$5000/MWh)
- Handling missing values through forward and backward filling
- Standardization of date formats

### Feature Engineering

- **Rolling statistics**: 7-day windows for mean, standard deviation
- **Lag features**: 1-period, 7-period, 48-period (daily), 336-period (weekly)
- **Volatility measures**: Relative volatility (RollingStd_7 / RollingMean_7)
- **Temporal features**: Hour of day, day of week, month, weekend flags, peak hour flags

### Dynamic Time Warping for Spatial Dependencies

Implemented in `3_dtw_FINAL.ipynb`:

1. **Pivot table creation**: POCs as columns, time indices as rows, prices as values
2. **DTW computation**: Pairwise distances between all POC time series
3. **Similarity transformation**: Gaussian kernel transformation of DTW distances
4. **Adjacency matrix**: Creation of graph structure for GAT component

### Data Splitting

- **Time-based splitting**: Maintaining temporal integrity
- **Training set**: 80% of time periods
- **Validation set**: 10% of time periods
- **Test set**: 10% of time periods

## 4. Modeling

### Temporal Fusion Transformer (TFT)

Implemented in `4_tft_FINAL.ipynb`:

- Variable selection networks
- LSTM encoder-decoder
- Multi-head attention mechanism
- Quantile outputs for probabilistic forecasting

### Graph Attention Network (GAT)

Implemented in `5_gat_FINAL.ipynb`:

- Graph structure from DTW-based adjacency matrix
- Multi-head graph attention layers
- Node features from POC characteristics

### Hybrid Model Integration

Implemented in `6_hybrid_FINAL.ipynb`:

- TFT for temporal pattern processing
- GAT for spatial dependency processing
- Fusion layer combining outputs
- Decoder for multi-horizon forecasts

### Training Approach

- Adam optimizer with learning rate scheduling
- Quantile loss function
- Early stopping and gradient clipping
- Batch size scheduling

## 5. Evaluation

### Performance Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Quantile losses for probabilistic forecasts
- R² and correlation coefficients

### Comparative Analysis

Comparison against baseline models:

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| ARIMA | 18.36 | 12.54 | 21.8% |
| LSTM | 15.42 | 10.87 | 18.4% |
| TFT | 14.28 | 9.76 | 16.2% |
| GAT | 15.63 | 10.92 | 18.6% |
| TFT-GAT Hybrid | 12.38 | 8.43 | 14.9% |

### Explainability Assessment

- Temporal attention weight analysis
- Variable importance analysis
- Graph attention visualization
- SHAP value analysis for local explanations

### Special Case Evaluation

- Price spike prediction
- Performance during proxy pricing events
- Island separation events

## 6. Deployment

While full deployment was beyond the project scope, the following considerations were made:

### Deployment Strategy

- Batch prediction system for hourly forecasts
- Integration with existing power systems dashboards
- Performance monitoring system

### Maintenance Plan

- Periodic retraining with new data
- Dynamic update of adjacency matrices as grid topology evolves
- Alert system for model drift detection

### Documentation and Knowledge Transfer

- Comprehensive documentation (this repository)
- Visualization tools for model interpretability
- Training materials for grid operators

## Dynamic Graph Learning Approach

A key innovation in this project is the Dynamic Graph Learning approach, which infers spatial relationships without requiring explicit grid topology data:

1. **Problem**: Electricity grid topology data is often proprietary, outdated, or unavailable

2. **Solution**: Use DTW to discover latent spatial relationships directly from price data

3. **Implementation**:
   - Calculate pairwise DTW distances between POC price time series
   - Transform distances to similarities using Gaussian kernel
   - Create graph structure for GAT component

4. **Advantages**:
   - Topology-free modeling
   - Captures effective relationships that may differ from physical connections
   - Adaptable to changing grid conditions
   - Applicable to other markets without requiring grid data

5. **Validation**:
   - 86% alignment with known physical connections in test areas
   - Improved forecasting during congestion events
   - More robust to grid reconfiguration

This approach enables spatial-temporal modeling in situations where grid topology information is unavailable, making the methodology more widely applicable across different electricity markets.
