# Machine Learning Algorithms from Scratch

This repository contains Python implementations of core machine learning algorithms from scratch, along with corresponding datasets in CSV format. Each script is self-contained and avoids using high-level machine learning libraries for understanding core logic, emphasizing an educational approach.

## 📁 Contents

### 1.  Principal Component Analysis (PCA)
- **File**: `pca.py`
- **Description**: Reduces the dimensionality of a dataset while preserving as much variance as possible.
- **Input**: Iris CSV file or Dataset with high-dimensional features.
- **Output**: Reduced dataset with 2 principal components, and a plot(optional)

### 2.  K-Medoids Clustering
- **File**: `kmedoid.py`
- **Description**: Partitions data into `k` clusters using medoids as cluster centers, more robust to noise and outliers than K-Means.
- **Input**:  Mall Feature dataset in CSV format.
- **Output**: Cluster assignments, medoid points, and optional visualization (2D).

### 3.  Multiple Linear Regression
- **File**: `multipleregression.py`
- **Description**: Predicts a continuous target variable based on multiple input features using gradient descent.
- **Input**: Boston CSV file or dataset containing independent and dependent variables.
- **Output**: Trained model parameters (theta), cost over iterations, and a cost vs epochs plot.

### 4. XOR Multilayer Perceptron Model with forward and backpropagation(hard coded for two bit input only)
- **File**: `XOR_multilayer.py`
- **Description**: Implements a multilayer perceptron model to solve the XOR problem, using one hidden layer. Weights and biases are randomly initialized.
- **Activation Function**: Sigmoid activation function is used. The output is 1 if f(z) ≥ 0.5, otherwise 0.
- **Input**: Takes two-bit binary inputs.
- **Output**: Classifies the XOR output (either 0 or 1) correctly for each input pair. Also prints final weights and biases after training.

### 5. Logistic Regression
- **File**: `Iris_logistic.py`
- **Description**: Predicts categorical variable based on multiple input features using logistic regression.
- **Input**: Iris dataset containing independent and dependent variable.
- **Output**: Trained model parameters(theta), cost over iterations, and a cost vs epochs plot.
## 📦 Datasets

Each algorithm includes a sample dataset (CSV format) placed in the same directory:
- `boston.csv` (used in multiple linear regression)
- `iris.csv`(used in pca and logistic)
- `mall.csv`(used in clustering)

## 🔧 Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib` (optional for plotting)

Install dependencies (if needed):

```bash
pip install numpy pandas matplotlib
