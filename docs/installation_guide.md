# Installation Guide

This guide provides detailed instructions for setting up the WattGridf environment and running the project components.

## System Requirements

### Hardware Requirements

- **CPU**: Intel Core i5/AMD Ryzen 5 or better (8+ cores recommended)
- **RAM**: Minimum 16GB, 32GB+ recommended for full dataset processing
- **GPU**: NVIDIA GPU with CUDA support (GTX 1650+ or RTX series) recommended for model training
- **Storage**: Minimum 10GB free space (50GB+ recommended for full dataset)

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Python**: Version 3.8+ (3.10 recommended)
- **CUDA**: Version 11.7+ (for GPU acceleration)

## Setting Up the Environment

### Option 1: Using Conda (Recommended)

1. Install Miniconda or Anaconda from [https://docs.conda.io/projects/conda/en/latest/user-guide/install/](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

2. Create a new conda environment:
   ```bash
   conda create -n wattgridf python=3.10
   conda activate wattgridf
   ```

3. Install PyTorch with GPU support (adjust based on your CUDA version):
   ```bash
   # For CUDA 11.7
   conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

   # For CPU only
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```

4. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using pip and venv

1. Create a virtual environment:
   ```bash
   python -m venv wattgridf-env

   # On Windows
   wattgridf-env\Scripts\activate

   # On macOS/Linux
   source wattgridf-env/bin/activate
   ```

2. Install PyTorch:
   ```bash
   # For CUDA 11.7
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

   # For CPU only
   pip install torch torchvision torchaudio
   ```

3. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 3: Using Google Colab

For those without access to a local GPU or sufficient compute resources, Google Colab provides a free alternative.

1. Upload the notebooks to Google Drive
2. Open notebooks with Google Colab
3. Add the following code at the beginning of each notebook:
   ```python
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')

   # Install dependencies
   !pip install -q pytorch_forecasting torch_geometric fastdtw
   ```

4. Adjust all file paths in the notebooks to point to your Google Drive directory

## Data Setup

1. Create the necessary directories if they don't exist:
   ```bash
   mkdir -p data/raw data/processed results/figures results/models results/tables
   ```

2. Obtain the electricity price data from the New Zealand Electricity Authority's EMI database: [https://www.emi.ea.govt.nz/](https://www.emi.ea.govt.nz/)

3. Place the raw CSV files in the `data/raw` directory

4. Process the raw data:
   ```bash
   # Generate the processed dataset
   python src/data/make_dataset.py

   # Preprocess and create features
   python src/data/preprocess.py

   # Generate DTW features
   python src/features/build_features.py
   ```

## Running the Models

### Option 1: Using the Command Line Interface

Train each model separately:

```bash
# Train the TFT model
python src/models/tft_model.py

# Train the GAT model
python src/models/gat_model.py

# Train the hybrid model
python src/models/hybrid_model.py
```

### Option 2: Running the Notebooks

Follow the notebooks in sequence for a step-by-step walkthrough:

1. Start Jupyter Lab or Notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. Open and run the notebooks in order:
   - `1_eda_FNAL.ipynb`
   - `2_preprocessing_FINAL.ipynb`
   - `3_dtw_FINAL.ipynb`
   - `4_tft_FINAL.ipynb`
   - `5_gat_FINAL.ipynb`
   - `6_hybrid_FINAL.ipynb`

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

If you encounter CUDA out of memory errors:
1. Reduce the batch size in the model training parameters
2. Use a smaller subset of the data by adjusting `max_samples` parameter
3. Reduce model size by adjusting `hidden_size` and other hyperparameters

#### PyTorch Geometric Installation Issues

If you have trouble installing PyTorch Geometric:
```bash
# Install PyTorch Geometric dependencies manually
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")
```

Then install PyTorch Geometric:
```bash
pip install torch-geometric
```

#### FastDTW Performance Issues

If DTW computation is too slow:
1. Reduce the number of POCs to process by setting `max_pocs` parameter in `build_features.py`
2. Increase the window constraint parameter for faster but less accurate DTW computation
3. Use the pre-computed DTW matrices in `data/processed/` if available

#### Memory Issues with Large Datasets

If you're experiencing memory errors with the full dataset:
1. Process data in chunks by modifying `make_dataset.py`
2. Sample the data before processing:
   ```python
   # In preprocessing scripts
   data = data.sample(n=1000000, random_state=42)  # Sample 1M rows
   ```

## Verification

To verify your installation is working correctly:

1. Run the basic test script:
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); import torch_geometric; print('PyG version:', torch_geometric.__version__)"
   ```

2. Check that you can load the sample data:
   ```bash
   python -c "import pandas as pd; print(pd.read_csv('data/processed/featured_data.csv').shape)"
   ```

## Next Steps

After installation:

1. Review the [project documentation](../README.md) for an overview
2. Explore the [model architecture](model_architecture.md) documentation
3. Read the [methodology](methodology.md) to understand the approach
4. Check the [API reference](api_reference.md) if you plan to use the code as a library
