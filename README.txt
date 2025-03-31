EXPLAINABLE SPATIAL-TEMPORAL FORECASTING OF ELECTRICITY PRICES
======================================================

Overview
--------
This project implements a hybrid spatial-temporal forecasting model for electricity price prediction in the New Zealand market. The model integrates Temporal Fusion Transformers (TFT) with Graph Attention Networks (GAT) to capture complex temporal patterns and spatial dependencies between Points of Connection (POCs).

For complete documentation, code, and instructions, please visit the GitHub repository:
https://github.com/thedhinakarr/wattGridf

Quick Setup Instructions
-----------------------
1. GOOGLE COLAB ACCESS:
   The implementation is available through these Google Colab notebooks:

   - Exploratory Data Analysis:
     https://colab.research.google.com/drive/1id5WP6gZgAipPrvEMaQow_sXiSnFXKXw

   - Data Preprocessing:
     https://colab.research.google.com/drive/1tWlCnCovujRVCOLGuitaiwzocoNu4Hrk

   - Dynamic Time Warping:
     https://colab.research.google.com/drive/10_jVLxxvGZsvY9K3R0dO__DMS7paus_N

   - Temporal Fusion Transformer:
     https://colab.research.google.com/drive/1g3zcN_hJ1NHwiVYTZIB_LB41Pw65tzkb

   - Graph Attention Network:
     https://colab.research.google.com/drive/1YLn2KRSFi3wNwIJcoVFwl_n2ZUsXmVdK

   - Hybrid Model:
     https://colab.research.google.com/drive/1YuoDryvxMACkR5TMPfb1CMwSbx8V6jDx

2. DATA ACCESS:
   Due to GitHub file size limits, download the data files from:
   https://drive.google.com/drive/folders/1GgSX5nYtxqsZ4l_VwCbjYTzLSmbDSUD1?usp=sharing

3. GOOGLE DRIVE SETUP:
   Create this folder structure in Google Drive:
   - MyDrive/WattGrid/data/raw/
   - MyDrive/WattGrid/data/processed/
   - MyDrive/WattGrid/results/models/
   - MyDrive/WattGrid/results/plots/

4. EXECUTION:
   - Open notebooks through Colab links above
   - Set runtime type to GPU (Runtime > Change runtime type)
   - Mount Google Drive in each notebook
   - Execute notebooks in sequence (1 through 6)

Hardware Requirements
-------------------
- For notebooks 1-3: Google Colab free tier is sufficient
- For notebooks 4-6 (model training): Google Colab Pro recommended

Troubleshooting
--------------
If you encounter issues:
- For memory errors: Reduce sample_size parameters in the code
- For session disconnects: Enable notifications for long-running cells
- For package conflicts: Restart runtime after installations

For more detailed instructions and technical documentation, refer to the GitHub repository.
