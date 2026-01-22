# Mathematical Background: PCA Quantization

This document explains the mathematical foundations of the Dimensionality Reduction technique used in the `Quantizer`.

## Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.

In this project, we use PCA to reduce the dimensionality of vector embeddings (e.g., from 768 dimensions to 128) before storage, significantly reducing memory and disk usage while preserving the most important information (variance).

### The Linear Algebra

The implementation in `pkg/quantizer/pca.go` relies on **Eigendecomposition of the Covariance Matrix**. While the user prompt mentioned SVD (Singular Value Decomposition), it is worth noting the deep mathematical relationship: the eigenvectors of the covariance matrix are equivalent to the right singular vectors of the centered data matrix.

#### 1. Centering the Data
Given a dataset matrix $X$ of shape $n \times d$ (where $n$ is the number of vectors and $d$ is the original dimension), we first compute the mean vector $\mu$:

$$ \mu_j = \frac{1}{n} \sum_{i=1}^{n} X_{ij} $$

The data is then centered:
$$ B = X - \mu $$

#### 2. Covariance Matrix
We compute the covariance matrix $C$ of the centered data. Since we want to find the relationships between dimensions:

$$ C = \frac{1}{n-1} B^T B $$

This results in a $d \times d$ symmetric matrix.

#### 3. Eigendecomposition
We solve for the eigenvectors ($V$) and eigenvalues ($\lambda$) of $C$:

$$ C V = V \Lambda $$

*   **Eigenvectors ($V$)**: Represents the directions (components) of the data.
*   **Eigenvalues ($\lambda$)**: Represents the magnitude of variance in those directions.

#### 4. Dimensionality Reduction (Projection)
To reduce dimensions from $d$ to $k$ (e.g., 128), we sort the eigenvectors by their eigenvalues in descending order and select the top $k$ vectors to form the transformation matrix $W_k$ ($d \times k$).

The projection of a new vector $x$ is calculated as:

$$ x_{reduced} = (x - \mu) \cdot W_k $$

This operation maps the vector into the subspace with the highest variance.

### Storage Efficiency
By combining PCA with Scalar Quantization (8-bit integers), we achieve two levels of compression:
1.  **Dimensional**: $d \rightarrow k$ (e.g., $768 \rightarrow 128$ floats).
2.  **Scalar**: float64 (8 bytes) $\rightarrow$ uint8 (1 byte).

**Total Compression Factor:**
$$ \frac{768 \times 8}{128 \times 1} \approx 48x $$
