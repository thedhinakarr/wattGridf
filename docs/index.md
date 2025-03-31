# WattGridf Documentation

Welcome to the documentation for WattGridf - an explainable spatial-temporal forecasting model for electricity prices with dynamic graph learning.

## Documentation Contents

### Project Overview
- [Project Structure](../README.md) - Directory structure and project organization
- [Installation Guide](installation_guide.md) - Setup instructions and requirements

### Methodology
- [Methodology](methodology.md) - CRISP-DM process and approach
- [Model Architecture](model_architecture.md) - Detailed description of model components
- [Data Dictionary](data_dictionary.md) - Description of data fields and features

### Results and Analysis
- [Results Analysis](results_analysis.md) - Performance metrics and interpretation

### Technical Reference
- [API Reference](api_reference.md) - Function and class documentation

### Future Development
- [Future Work](future_work.md) - Potential enhancements and research directions

## Quick Start

To get started with WattGridf:

1. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Process the data**:
   ```bash
   python src/data/make_dataset.py
   python src/data/preprocess.py
   python src/features/build_features.py
   ```

3. **Train the models**:
   ```bash
   python src/models/tft_model.py
   python src/models/gat_model.py
   ```

4. **Or run the notebooks in sequence**:
   - 1_eda_FNAL.ipynb
   - 2_preprocessing_FINAL.ipynb
   - 3_dtw_FINAL.ipynb
   - 4_tft_FINAL.ipynb
   - 5_gat_FINAL.ipynb
   - 6_hybrid_FINAL.ipynb

## Key Features

- **Hybrid Spatial-Temporal Architecture**: Combines Temporal Fusion Transformers (TFT) and Graph Attention Networks (GAT)
- **Dynamic Graph Learning**: Infers spatial relationships without requiring explicit grid topology
- **Multi-layered Explainability**: Provides insights through attention mechanisms, feature importance, and SHAP values
- **Probabilistic Forecasting**: Outputs quantile forecasts for uncertainty quantification
- **Anomaly Detection**: Identifies and explains price spikes and unusual market behavior

## Project Background

The electricity market exhibits complex spatial-temporal dependencies that influence price fluctuations. Traditional forecasting models often fail to capture these dependencies or lack interpretability. WattGridf addresses these challenges through a novel hybrid approach that combines temporal pattern analysis with spatial relationship modeling.

The project was developed using the New Zealand electricity market as a case study, but the methodology is applicable to other electricity markets and similar spatial-temporal forecasting problems.

## Citing This Work

## Contact

For questions or feedback about WattGridf, reach out to:

Dhinakar Ramthu - dhra24@student.bth.se
