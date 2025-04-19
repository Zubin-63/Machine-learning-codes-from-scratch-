# Machine Learning Algorithms from Scratch

This repository contains Python implementations of core machine learning algorithms from scratch, along with corresponding datasets in CSV format. Each script is self-contained and avoids using high-level machine learning libraries for core logic, emphasizing an educational, hands-on approach.

## 📁 Contents

### 1. 📉 Principal Component Analysis (PCA)
- **File**: `pca.py`
- **Description**: Reduces the dimensionality of a dataset while preserving as much variance as possible.
- **Input**: CSV file with high-dimensional features.
- **Output**: Reduced dataset with principal components, and a variance plot.

### 2. 🧩 K-Medoids Clustering
- **File**: `kmedoids.py`
- **Description**: Partitions data into `k` clusters using medoids as cluster centers, more robust to noise and outliers than K-Means.
- **Input**: Feature dataset in CSV format.
- **Output**: Cluster assignments, medoid points, and optional visualization (2D).

### 3. 📊 Multiple Linear Regression
- **File**: `multiple_linear_regression.py`
- **Description**: Predicts a continuous target variable based on multiple input features using gradient descent.
- **Input**: CSV file containing independent and dependent variables.
- **Output**: Trained model parameters (theta), cost over iterations, and a cost vs epochs plot.

## 📦 Datasets

Each algorithm includes a sample dataset (CSV format) placed in the same directory:
- `boston.csv` (used in multiple linear regression)
- Other dataset names specific to `pca.py` and `kmedoids.py`

## 🔧 Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib` (optional for plotting)

Install dependencies (if needed):

```bash
pip install numpy pandas matplotlib
