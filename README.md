# Explainable Spatial-Temporal Forecasting of Electricity Prices with Dynamic Graph Learning

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
├── README.txt                  # This file
└── requirements.txt            # Python dependencies
```

## Local Setup Instructions

### Environment Setup

1. Clone or download this repository to your local machine

2. Create a Python virtual environment (Python 3.8+ recommended):
   ```bash
   # Using venv
   python -m venv env

   # Activate on Windows
   env\Scripts\activate

   # Activate on macOS/Linux
   source env/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. Create the necessary directory structure if it doesn't exist:
   ```bash
   mkdir -p data/raw data/processed results/plots results/models
   ```

2. Place the raw electricity price data files in the `data/raw/` directory

3. Run the notebooks in sequence (see Running the Code section below)

## Google Colab Setup Instructions

### Google Drive Setup

1. Create the following folder structure in your Google Drive:
   ```
   MyDrive/
   └── WattGrid/
       ├── data/
       │   ├── raw/             # For raw electricity price data
       │   └── processed/       # For processed datasets
       ├── notebooks/           # Optional: copies of the notebooks
       └── results/
           ├── models/          # For saved model files
           └── plots/           # For visualizations
   ```

2. Upload the raw electricity price data to the `WattGrid/data/raw/` directory in your Google Drive

### Accessing Notebooks

Access the notebooks directly through these Google Colab links:

- **Exploratory Data Analysis**: [1_eda.ipynb](https://colab.research.google.com/drive/1id5WP6gZgAipPrvEMaQow_sXiSnFXKXw)
- **Data Preprocessing**: [2_preprocessing.ipynb](https://colab.research.google.com/drive/1tWlCnCovujRVCOLGuitaiwzocoNu4Hrk)
- **Dynamic Time Warping**: [3_dtw.ipynb](https://colab.research.google.com/drive/10_jVLxxvGZsvY9K3R0dO__DMS7paus_N)
- **Temporal Fusion Transformer**: [4_tft.ipynb](https://colab.research.google.com/drive/1g3zcN_hJ1NHwiVYTZIB_LB41Pw65tzkb)
- **Graph Attention Network**: [5_gat.ipynb](https://colab.research.google.com/drive/1YLn2KRSFi3wNwIJcoVFwl_n2ZUsXmVdK)
- **Hybrid Model**: [6_hybrid.ipynb](https://colab.research.google.com/drive/1YuoDryvxMACkR5TMPfb1CMwSbx8V6jDx)

## Running the Code

### Local Execution

1. Start Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. Navigate to the notebooks directory and run each notebook in sequence:
   - `1_eda.ipynb`: Exploratory Data Analysis
   - `2_preprocessing.ipynb`: Data Preprocessing
   - `3_dtw.ipynb`: Dynamic Time Warping
   - `4_tft.ipynb`: TFT Model
   - `5_gat.ipynb`: GAT Model
   - `6_hybrid.ipynb`: Hybrid Model

3. Each notebook includes detailed instructions and comments explaining each step

### Google Colab Execution

1. Open each notebook using the links provided above

2. Set up the runtime environment:
   - Select Runtime > Change runtime type > Hardware accelerator: GPU

3. Mount your Google Drive in each notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Execute each notebook in sequence, following the instructions within

5. For long-running notebooks (4, 5, and 6), enable notifications:
   - Settings > Miscellaneous > Receive notification when execution completes

## Data Pipeline

The execution pipeline follows this sequence:

1. **Data Exploration** (`1_eda.ipynb`)
   - Analyzes and visualizes raw electricity price data
   - Output: Visualizations in `results/plots/`

2. **Data Preprocessing** (`2_preprocessing.ipynb`)
   - Cleans data and creates features
   - Output: `data/processed/cleaned_data.csv`, `data/processed/featured_data.csv`

3. **DTW Computation** (`3_dtw.ipynb`)
   - Computes spatial relationships between POCs
   - Output: `data/processed/dtw_adjacency_matrix.csv`, `data/processed/dtw_adjacency2_matrix.csv`

4. **TFT Model** (`4_tft.ipynb`)
   - Implements and trains Temporal Fusion Transformer
   - Output: `results/models/tft_model_1M.pth`

5. **GAT Model** (`5_gat.ipynb`)
   - Implements and trains Graph Attention Network
   - Output: GAT model weights, evaluation metrics

6. **Hybrid Model** (`6_hybrid.ipynb`)
   - Combines TFT and GAT models
   - Output: Hybrid model, evaluation metrics, visualizations

## Troubleshooting

### Common Local Issues

- **Package Installation Errors**: Ensure you're using Python 3.8+ and try installing packages individually
- **Memory Issues**: Reduce batch sizes or sample sizes in the code
- **GPU Support**: Check PyTorch installation matches your CUDA version

### Common Google Colab Issues

- **Session Disconnects**: For long-running cells, use smaller data samples or Colab Pro
- **Memory Errors**: Restart runtime and reduce batch/sample sizes
- **Google Drive Mount Failures**: Re-authenticate and check folder paths
- **Package Version Conflicts**: Clear output and restart runtime after installing packages

## Hardware Requirements

### Local Machine
- 16GB+ RAM recommended
- CUDA-compatible GPU highly recommended
- ~5GB free disk space

### Google Colab
- Free tier: Sufficient for notebooks 1-3
- Colab Pro recommended for notebooks 4-6 (TFT, GAT, Hybrid models)
