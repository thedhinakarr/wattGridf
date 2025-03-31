# Data Dictionary

This document provides a detailed description of all data fields used in the WattGridf project, including raw data fields, derived features, and model outputs.

## Raw Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `TradingDate` | datetime | The date of the trading activity |
| `TradingPeriod` | int | Half-hourly trading period (1-48 per day) |
| `PointOfConnection` (POC) | string | Identifier for a specific node in the electricity grid |
| `DollarsPerMegawattHour` | float | Electricity price at the specified POC and time |
| `Island` | string | Indicates whether the POC is on the North or South Island |
| `PublishDateTime` | datetime | When the prices were published to the market |
| `IsProxyPriceFlag` | string | Indicates if the price is a proxy (administratively determined) rather than a market-cleared price |

## Engineered Features

### Time Series Features

| Feature | Type | Description |
|---------|------|-------------|
| `RollingMean_7` | float | 7-day rolling average of prices for each POC |
| `RollingStd_7` | float | 7-day rolling standard deviation of prices for each POC |
| `Lag_1` | float | Previous period's price |
| `Lag_7` | float | Price from 7 periods ago |
| `Lag_48` | float | Price from same time on previous day (48 periods ago) |
| `Lag_336` | float | Price from same time on previous week (336 periods ago) |
| `PriceVolatility` | float | Relative volatility (RollingStd_7 / RollingMean_7) |
| `log_price` | float | Natural logarithm of DollarsPerMegawattHour (for stabilizing variance) |
| `time_idx` | int | Sequential time index for time series modeling |
| `HourOfDay` | int | Hour of the day (0-23) derived from TradingPeriod |
| `DayOfWeek` | int | Day of the week (0-6, Monday=0) |
| `Month` | int | Month of the year (1-12) |
| `IsWeekend` | binary | Flag indicating weekend days |
| `IsPeakHour` | binary | Flag for peak hours (7-9 AM or 5-8 PM) |

### POC-level Features

| Feature | Type | Description |
|---------|------|-------------|
| `MeanPrice` | float | Average price at the POC |
| `StdPrice` | float | Standard deviation of prices at the POC |
| `MinPrice` | float | Minimum price observed at the POC |
| `MaxPrice` | float | Maximum price observed at the POC |
| `Volatility` | float | Price volatility at the POC (StdPrice / MeanPrice) |
| `IsNorth` | binary | Flag indicating North Island POC |
| `IsSouth` | binary | Flag indicating South Island POC |
| `PeakRatio` | float | Ratio of peak hour prices to off-peak prices |

## Graph Structure Data

### Adjacency Matrix

The `dtw_adjacency_matrix.csv` and `dtw_adjacency2_matrix.csv` files contain similarity matrices derived from Dynamic Time Warping (DTW):

- **Rows/Columns**: Each row and column represents a POC
- **Values**: Similarity scores between POCs, transformed from DTW distances using a Gaussian kernel:

  ```
  similarity_i,j = exp(-(DTW(i,j)^2)/(2*sigma^2))
  ```

  where `sigma` is the standard deviation of all DTW distances

- **Interpretation**: Higher values indicate stronger similarity in price patterns between POCs

### Edge List

From the adjacency matrix, we derive an edge list for the graph model:

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Source POC identifier |
| `target` | string | Target POC identifier |
| `weight` | float | Edge weight (similarity score) |

## Model Outputs

### TFT Model Outputs

| Output | Description |
|--------|-------------|
| `predictions` | Forecasted prices for each POC and time step |
| `quantiles` | Quantile forecasts (10th, 50th, 90th percentiles) |
| `temporal_attention_weights` | Attention weights for each time step |
| `variable_importance` | Importance scores for each input feature |

### GAT Model Outputs

| Output | Description |
|--------|-------------|
| `predictions` | Forecasted relative prices for each POC |
| `attention_weights` | Graph attention weights between POCs |

### Hybrid Model Outputs

| Output | Description |
|--------|-------------|
| `predictions` | Multi-horizon quantile forecasts |
| `temporal_attention_weights` | Temporal attention weights |
| `spatial_attention_weights` | Spatial attention weights |
| `var_weights` | Variable importance weights |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `RMSE` | Root Mean Squared Error |
| `MAE` | Mean Absolute Error |
| `MAPE` | Mean Absolute Percentage Error |
| `QuantileLoss` | Specific loss for each predicted quantile |
| `CRPS` | Continuous Ranked Probability Score (measures quality of probabilistic forecasts) |
| `R2` | Coefficient of determination |
| `Correlation` | Correlation coefficient between predictions and actuals |
