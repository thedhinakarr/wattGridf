# Explainable Spatial-Temporal Forecasting of Electricity Prices with Dynamic Graph Learning

## Project Overview

This repository implements a hybrid spatial-temporal forecasting model for electricity price prediction in the New Zealand market. The model integrates Temporal Fusion Transformers (TFT) with Graph Attention Networks (GAT) to capture complex temporal patterns and spatial dependencies between Points of Connection (POCs) without requiring explicit grid topology information.

## Repository Structure

```
wattGrid/
├── .ropeproject/               # Rope project configuration
├── data/                       # Data storage directory
│   ├── processed/              # Processed and transformed datasets
│   └── raw/                    # Raw electricity market price data
├── docs/                       # Documentation
│   ├── data_dictionary.md      # Data fields and definitions
│   ├── future_work.md          # Potential improvements and extensions
│   ├── index.md                # Documentation entry point
│   ├── installation_guide.md   # Detailed setup instructions
│   ├── methodology.md          # Technical approach and methodology
│   ├── model_architecture.md   # Model architecture description
│   └── results_analysis.md     # Analysis of model performance
├── notebooks/                  # Google Colab notebooks
│   ├── 1_eda.ipynb             # Exploratory Data Analysis
│   ├── 2_preprocessing.ipynb   # Data cleaning and feature engineering
│   ├── 3_dtw.ipynb             # Dynamic Time Warping implementation
│   ├── 4_tft.ipynb             # Temporal Fusion Transformer model
│   ├── 5_gat.ipynb             # Graph Attention Network model
│   ├── 6_hybrid.ipynb          # Hybrid TFT-GAT model
│   └── index.md                # Notebook guide
├── results/                    # Model outputs and results
│   ├── models/                 # Saved model files
│   │   ├── __init__.py         # Python package initialization
│   │   └── tft_model_1M.pth    # Trained TFT model (1M samples)
│   └── plots/                  # Generated visualizations
├── src/                        # Source code
│   ├── data/                   # Data processing scripts
│   │   ├── make_dataset.py     # Dataset creation
│   │   └── preprocess.py       # Preprocessing utilities
│   └── features/               # Feature engineering
│       ├── __init__.py         # Package initialization
│       └── build_features.py   # Feature creation utilities
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Google Colab Implementation

This project is implemented using Google Colab notebooks. Access the notebooks through these direct links:

- **Exploratory Data Analysis**:
  - [1_eda.ipynb](https://colab.research.google.com/drive/1id5WP6gZgAipPrvEMaQow_sXiSnFXKXw)
  - Direct URL: https://colab.research.google.com/drive/1id5WP6gZgAipPrvEMaQow_sXiSnFXKXw

- **Data Preprocessing**:
  - [2_preprocessing.ipynb](https://colab.research.google.com/drive/1tWlCnCovujRVCOLGuitaiwzocoNu4Hrk)
  - Direct URL: https://colab.research.google.com/drive/1tWlCnCovujRVCOLGuitaiwzocoNu4Hrk

- **Dynamic Time Warping**:
  - [3_dtw.ipynb](https://colab.research.google.com/drive/10_jVLxxvGZsvY9K3R0dO__DMS7paus_N)
  - Direct URL: https://colab.research.google.com/drive/10_jVLxxvGZsvY9K3R0dO__DMS7paus_N

- **Temporal Fusion Transformer**:
  - [4_tft.ipynb](https://colab.research.google.com/drive/1g3zcN_hJ1NHwiVYTZIB_LB41Pw65tzkb)
  - Direct URL: https://colab.research.google.com/drive/1g3zcN_hJ1NHwiVYTZIB_LB41Pw65tzkb

- **Graph Attention Network**:
  - [5_gat.ipynb](https://colab.research.google.com/drive/1YLn2KRSFi3wNwIJcoVFwl_n2ZUsXmVdK)
  - Direct URL: https://colab.research.google.com/drive/1YLn2KRSFi3wNwIJcoVFwl_n2ZUsXmVdK

- **Hybrid Model**:
  - [6_hybrid.ipynb](https://colab.research.google.com/drive/1YuoDryvxMACkR5TMPfb1CMwSbx8V6jDx)
  - Direct URL: https://colab.research.google.com/drive/1YuoDryvxMACkR5TMPfb1CMwSbx8V6jDx

## Data Access

**Important**: Some files in this project exceed GitHub's file size limits and could not be uploaded to the repository:
- `data/processed/cleaned_data.csv` (~100 MB)
- `data/processed/featured_data.csv` (~1.6 GB)

Access the complete dataset and processed files via this Google Drive link:

[https://drive.google.com/drive/folders/1GgSX5nYtxqsZ4l_VwCbjYTzLSmbDSUD1?usp=sharing](https://drive.google.com/drive/folders/1GgSX5nYtxqsZ4l_VwCbjYTzLSmbDSUD1?usp=sharing)

## Setup Instructions

### Google Drive Setup

1. Create the following folder structure in your Google Drive:
   ```
   MyDrive/
   └── WattGrid/
       ├── data/
       │   ├── raw/             # For raw electricity price data
       │   └── processed/       # For processed datasets
       └── results/
           ├── models/          # For saved model files
           └── plots/           # For visualizations
   ```

2. Download the data files from the Google Drive link above and place them in the appropriate directories

### Running Notebooks in Google Colab

1. Open each notebook using the provided Google Colab links

2. Set up the runtime environment:
   - Select Runtime > Change runtime type > Hardware accelerator: GPU

3. Mount your Google Drive in each notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Execute each notebook in sequence, following the instructions within

## Execution Pipeline

The implementation follows this sequential workflow:

1. **Data Exploration** (`1_eda.ipynb`)
   - Analyzes and visualizes raw electricity price data
   - Output: Visualizations of price distributions, temporal patterns, and POC characteristics

2. **Data Preprocessing** (`2_preprocessing.ipynb`)
   - Cleans data and creates features including rolling statistics and lag variables
   - Output: `data/processed/cleaned_data.csv`, `data/processed/featured_data.csv`

3. **DTW Computation** (`3_dtw.ipynb`)
   - Computes pairwise DTW distances between POCs to infer spatial relationships
   - Output: `data/processed/dtw_adjacency_matrix.csv`, `data/processed/dtw_adjacency2_matrix.csv`

4. **TFT Model** (`4_tft.ipynb`)
   - Implements and trains Temporal Fusion Transformer for time series forecasting
   - Output: `results/models/tft_model_1M.pth`

5. **GAT Model** (`5_gat.ipynb`)
   - Implements and trains Graph Attention Network using DTW-derived adjacency matrix
   - Output: GAT model weights, evaluation metrics

6. **Hybrid Model** (`6_hybrid.ipynb`)
   - Integrates TFT and GAT into a unified architecture
   - Output: Hybrid model weights, evaluation metrics, visualizations

## Google Colab Execution Notes

1. **Long-running operations**:
   - Enable "Settings > Miscellaneous > Receive notification when execution completes"
   - For notebooks that take several hours (TFT, GAT, Hybrid), using Colab Pro may prevent timeouts

2. **Memory management**:
   - Use appropriate sample sizes based on available memory
   - The code includes parameters to adjust sample size (e.g., `sample_size=1000000` in some notebooks)
   - Clear output and restart runtime if memory issues occur

3. **GPU acceleration**:
   - Always use GPU runtime for model training
   - Check GPU allocation with `!nvidia-smi` at the beginning of each notebook

## Hardware Requirements (Google Colab)

- Free tier: Suitable for data exploration and preprocessing (Notebooks 1-3)
- Colab Pro recommended for full model training (Notebooks 4-6):
  - Higher memory allocation (25GB+ RAM)
  - Longer runtime limits
  - Priority access to better GPUs (T4/P100)

## Troubleshooting

- **Session Disconnects**: For long-running cells, enable notifications or use Colab Pro
- **Memory Issues**: Reduce `sample_size` parameters in the code if encountering memory limitations
- **GPU Unavailability**: The code can run on CPU but will be significantly slower; adjust batch sizes accordingly
- **Drive Mount Failures**: Re-authenticate and check folder paths
- **Package Version Conflicts**: Restart runtime after installing packages

## Local Setup (Alternative)

For local execution:

1. Clone this repository
2. Download data from the Google Drive link
3. Create a Python environment: `python -m venv env`
4. Activate the environment and install dependencies: `pip install -r requirements.txt`
5. Run notebooks locally using Jupyter: `jupyter notebook`

Note that local execution requires sufficient computational resources, particularly for the model training notebooks.
