import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin

def compute_class_covariances(X, y):
    """
    Compute class-wise mean normalized spatial covariance matrices.
    
    Args:
        X (np.ndarray): EEG epochs of shape (n_trials, n_channels, n_times).
        y (np.ndarray): Labels of shape (n_trials,).
        
    Returns:
        tuple: (Sigma_0, Sigma_1) mean covariance matrices.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"CustomCSP currently supports binary classification. Found classes: {classes}")

    covs = []
    for target in classes:
        # Filter trials for specific class
        X_class = X[y == target]
        
        # Batch compute covariance: (N, C, T) @ (N, T, C) -> (N, C, C)
        trial_covs = np.matmul(X_class, X_class.transpose(0, 2, 1))
        
        # Normalize by trace (total energy per trial)
        traces = np.einsum("ijj->i", trial_covs)
        normalized_trial_covs = trial_covs / traces[:, np.newaxis, np.newaxis]
        
        # Mean covariance for this class
        covs.append(np.mean(normalized_trial_covs, axis=0))
        
    return covs[0], covs[1]

class CustomCSP(BaseEstimator, TransformerMixin):
    """
    Custom Common Spatial Patterns (CSP) transformer for BCI.
    
    Implements a manual spatial transformation determined by a projection matrix W
    such that W^T X = X_transformed.
    """

    def __init__(self, n_components=6, reg=1e-6, log=True):
        """
        Args:
            n_components (int): Total number of spatial filters to retain.
            reg (float): Shrinkage regularization (alpha) for covariance matrices.
            log (bool): If True, return log-variance of projected signals.
        """
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.filters_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Estimate the optimal spatial filters.
        """
        self.classes_ = np.unique(y)
        
        # 1. Compute Class Covariances
        Sigma_0, Sigma_1 = compute_class_covariances(X, y)
        
        # 2. Regularization (Shrinkage): Sigma_reg = (1-reg)*Sigma + reg*I
        if self.reg is not None and self.reg > 0:
            C = Sigma_0.shape[0]
            eye = np.eye(C)
            Sigma_0 = (1 - self.reg) * Sigma_0 + self.reg * eye
            Sigma_1 = (1 - self.reg) * Sigma_1 + self.reg * eye

        # 3. Solve Generalized Eigenvalue Problem: Sigma_0 * v = lambda * (Sigma_0 + Sigma_1) * v
        # eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = linalg.eigh(Sigma_0, Sigma_0 + Sigma_1)
        
        # 4. Select components (m extreme pairs)
        m = self.n_components // 2
        # Indices for the smallest (Class 1 variance) and largest (Class 0 variance) eigenvalues
        ix = np.concatenate([np.arange(m), np.arange(-m, 0)])
        self.filters_ = eigenvectors[:, ix]
        
        return self

    def transform(self, X):
        """
        Apply spatial filters and extract features.
        """
        if self.filters_ is None:
            raise RuntimeError("CustomCSP must be fitted before calling transform.")
            
        # 1. Project signals: (N, C, T) -> (N, n_components, T)
        # W.T @ X_i for each trial i
        Z = np.matmul(self.filters_.T, X)
        
        # 2. Feature Extraction: Variance across the time dimension
        feat = np.var(Z, axis=2)
        
        # 3. Log transformation for normality
        if self.log:
            feat = np.log(feat + 1e-10)
            
        return feat
