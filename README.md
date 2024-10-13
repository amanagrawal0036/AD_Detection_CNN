# Alzheimer Disease Detection Using Deep Learning

**Contributors**:  
- Aman Agrawal  
- Ishaan Gupta

**Supervisor**:  
Dr. Bharat Richhariya

This project focuses on detecting Alzheimer’s Disease using deep learning models trained on structural MRI scans. We adapted the original code from a study by Johannes Rieke et al., performing manual hyperparameter tuning and updating the deprecated code to work with the latest versions of Python and PyTorch. 

---

## Abstract

Alzheimer’s Disease detection is a critical challenge in the field of medical imaging and diagnosis. This study leverages convolutional neural networks (CNNs) to identify Alzheimer’s Disease using MRI scans from the ADNI dataset. Our modifications include tuning hyperparameters for optimal performance and updating the code to ensure compatibility with current Python 3 and PyTorch releases. Visualization methods such as sensitivity analysis and occlusion maps are employed to interpret the CNN's predictions, highlighting relevant regions of the brain associated with Alzheimer’s Disease.

---

## Code Structure

The codebase utilizes **PyTorch** and **Jupyter notebooks** for both training the model and generating visualizations.

- **`training.ipynb`**: Train the CNN model using cross-validation.
- **`interpretation-mri.ipynb`**: Generate relevance heatmaps for model interpretation using various visualization methods.

### Python files supporting these notebooks include:
- **`interpretation.py`**: Methods for visualizing CNN decisions.
- **`utils.py`**: Utility functions for MRI scan processing and visualization.

---

## Model and Data

The MRI data used in this study is sourced from the Alzheimer's Disease Neuroimaging Initiative (ADNI). Access to the MRI scans is free, but you must apply for access.

For consistency with the codebase, organize the MRI scans according to the folder structure specified in datasets.py. We utilized the preprocessed version of this dataset provided by our supervisor, and the preprocessing of images was performed using ANTs non-linear registration.

---

## Setup Instructions

### Step 1: Create and Activate Conda Environment

```bash
conda create -n pyt python=3.12 anaconda
conda activate pyt
```
### Step 2: Install Required Dependencies

First, add the `conda-forge` channel to your Conda configuration and install the necessary dependencies:

```bash
conda config --append channels conda-forge
conda install numpy pandas scipy matplotlib tqdm scikit-learn pip git torchvision nibabel tabulate torchmetrics ipywidgets
```
### Step 3: Install Additional Packages

Next, install the additional Python packages required for the project:

```bash
pip install git+https://github.com/jrieke/torchsample
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
