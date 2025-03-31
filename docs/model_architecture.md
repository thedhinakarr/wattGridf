# Model Architecture

## Hybrid TFT-GAT Model

The WattGridf project implements a novel hybrid architecture that combines Temporal Fusion Transformers (TFT) for temporal forecasting and Graph Attention Networks (GAT) for spatial modeling. This document details the architecture of each component and how they are integrated.

## Temporal Component: Temporal Fusion Transformer

The TFT component processes temporal patterns in electricity prices.

### Architecture Details

1. **Variable Selection Networks**
   - Identify the most relevant input variables for encoder (historical inputs) and decoder (future inputs)
   - Provide initial variable importance scores
   - Improve model interpretability by highlighting key features

2. **Encoder-Decoder LSTM**
   - Process historical data (encoder) and generate future predictions (decoder)
   - 2-layer architecture with hidden dimension 16
   - Balance between model capacity and overfitting risk

3. **Temporal Self-Attention Mechanism**
   - Multi-head attention (4 heads) focusing on relevant historical time steps
   - Captures complex temporal patterns (daily, weekly cycles)
   - Provides interpretability through attention weights

4. **Static Covariate Encoders**
   - Process time-invariant features (POC identity, Island location)
   - Condition temporal patterns on static attributes

5. **Quantile Outputs**
   - Produce 10th, 50th, and 90th percentile forecasts
   - Enable uncertainty quantification in predictions
   - Particularly valuable for risk assessment in volatile market conditions

### Hyperparameters

- Learning rate: 0.03
- Hidden size: 16
- Attention head size: 4
- Dropout: 0.1
- Hidden continuous size: 8

## Spatial Component: Graph Attention Network

The GAT component models spatial dependencies between POCs using graph structure derived from DTW similarities.

### Architecture Details

1. **Graph Structure**
   - Nodes represent POCs with edges determined by DTW similarity
   - Edge weights represent strength of relationship between POCs
   - Allows model to consider influence of prices at related POCs

2. **Node Features**
   - Static characteristics (average price, volatility)
   - Island identification
   - Temporal embeddings from TFT component

3. **Multi-head Graph Attention**
   - Compute attention coefficients between connected nodes
   - Determine influence of neighboring POCs on target POC's price forecast
   - 4 attention heads capture different types of relationships

4. **Layer Structure**
   - 2-layer architecture
   - First layer: hidden dimension 64 with 4 attention heads
   - Second layer: dimension 64 with single attention head for aggregation

### Hyperparameters

- Hidden graph size: 64
- Graph heads: 4
- Dropout: 0.2

## Integration Mechanism

The fusion of temporal and spatial components is a key innovation in our approach:

1. **TFT-GAT Fusion**
   - Dedicated fusion layer combines temporal and spatial representations
   - Implemented as multi-layer perceptron [128, 64]
   - Layer normalization and ReLU activations

2. **Quantile Decoder**
   - Processes combined representation
   - Produces multi-horizon, multi-quantile forecasts
   - Ensures spatial information properly influences uncertainty estimates

3. **Attention Preservation**
   - Preserves attention weights from both components
   - Enables detailed model explanation
   - Maintains interpretability throughout the fusion process

## Training Approach

The hybrid model is trained with considerations specific to electricity price forecasting:

1. **Loss Function**
   - Quantile loss for probabilistic forecasting
   - Asymmetrically penalizes underestimation and overestimation
   - Reduces to Mean Absolute Error (MAE) for median (50th percentile) forecast

2. **Optimization**
   - Adam optimizer with initial learning rate of 0.001
   - Learning rate scheduler reducing rate by 50% after 3 epochs without improvement
   - Gradient clipping with maximum norm of 1.0 to prevent exploding gradients

3. **Regularization**
   - Dropout (0.1) throughout the network
   - L2 weight regularization (1e-5)
   - Early stopping with patience of 10 epochs

## Interpretability Features

The model provides multiple layers of interpretability:

1. **Variable Importance**
   - From variable selection networks in TFT
   - Highlights most influential features globally

2. **Temporal Attention Weights**
   - Show which historical time points influence predictions
   - Reveal temporal patterns like daily or weekly seasonality

3. **Spatial Attention Weights**
   - Visualize strength of connections between POCs
   - Identify critical nodes in the electricity grid

4. **SHAP Values**
   - Provide local explanations for individual predictions
   - Decompose forecasts into feature contributions
