#!/usr/bin/env python3
"""
PCA and ICA implementations for dimension reduction of tweet cascade data.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

class PCA:
    """
    Principal Component Analysis implemented from scratch using NumPy.
    Uses Singular Value Decomposition (SVD) for computing principal components.
    """
    
    def __init__(self, n_components=10, whiten=False):
        """
        Initialize PCA.
        
        Args:
            n_components (int): Number of components to keep
            whiten (bool): Whether to whiten the data
        """
        self.n_components = n_components
        self.whiten = whiten
        self.components_ = None
        self.mean_ = None
        self.scale_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
        """
        Fit PCA to data.
        
        Args:
            X (np.ndarray): Data matrix [n_samples, n_features]
        
        Returns:
            self: Returns self
        """
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Store components (transpose of V gives eigenvectors)
        self.components_ = Vt[:self.n_components]
        
        # Compute explained variance
        n_samples = X.shape[0]
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)
        
        # Compute scaling factor for whitening
        if self.whiten:
            self.scale_ = np.sqrt(self.explained_variance_[:self.n_components])
        
        return self
    
    def transform(self, X):
        """
        Transform data to principal component space.
        
        Args:
            X (np.ndarray): Data matrix [n_samples, n_features]
        
        Returns:
            np.ndarray: Transformed data [n_samples, n_components]
        """
        X_centered = X - self.mean_
        
        if self.whiten:
            return X_centered.dot(self.components_.T) / self.scale_
        else:
            return X_centered.dot(self.components_.T)
    
    def fit_transform(self, X):
        """
        Fit PCA to data and transform it.
        
        Args:
            X (np.ndarray): Data matrix [n_samples, n_features]
        
        Returns:
            np.ndarray: Transformed data [n_samples, n_components]
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Inverse transform data from principal component space.
        
        Args:
            X_transformed (np.ndarray): Transformed data [n_samples, n_components]
        
        Returns:
            np.ndarray: Original space data [n_samples, n_features]
        """
        if self.whiten:
            return X_transformed * self.scale_.dot(self.components_) + self.mean_
        else:
            return X_transformed.dot(self.components_) + self.mean_
    
    def save(self, filepath):
        """
        Save PCA model.
        
        Args:
            filepath (str): Path to save model
        """
        np.savez(
            filepath,
            components=self.components_,
            mean=self.mean_,
            scale=self.scale_,
            explained_variance=self.explained_variance_,
            explained_variance_ratio=self.explained_variance_ratio_,
            n_components=self.n_components,
            whiten=self.whiten
        )
    
    def load(self, filepath):
        """
        Load PCA model.
        
        Args:
            filepath (str): Path to load model from
        """
        data = np.load(filepath)
        self.components_ = data['components']
        self.mean_ = data['mean']
        self.scale_ = data['scale']
        self.explained_variance_ = data['explained_variance']
        self.explained_variance_ratio_ = data['explained_variance_ratio']
        self.n_components = int(data['n_components'])
        self.whiten = bool(data['whiten'])


class FastICA:
    """
    Fast Independent Component Analysis implemented from scratch using NumPy.
    Uses fixed-point iteration for finding independent components.
    """
    
    def __init__(self, n_components=10, max_iter=1000, tol=1e-4, whiten=True, random_state=None):
        """
        Initialize FastICA.
        
        Args:
            n_components (int): Number of components to extract
            max_iter (int): Maximum number of iterations
            tol (float): Tolerance for convergence
            whiten (bool): Whether to whiten the data
            random_state (int): Random seed for reproducibility
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.whiten = whiten
        self.random_state = random_state
        
        # PCA for whitening
        self.pca = None
        
        # ICA components and mixing matrix
        self.components_ = None
        self.mixing_ = None
    
    def _g(self, x):
        """
        Non-linear function for contrast (tanh).
        
        Args:
            x (np.ndarray): Input data
        
        Returns:
            tuple: Function value and derivative
        """
        gx = np.tanh(x)
        g_prime = 1 - gx ** 2
        return gx, g_prime
    
    def fit(self, X):
        """
        Fit ICA to data.
        
        Args:
            X (np.ndarray): Data matrix [n_samples, n_features]
        
        Returns:
            self: Returns self
        """
        n_samples, n_features = X.shape
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Whiten data using PCA
        if self.whiten:
            self.pca = PCA(n_components=self.n_components, whiten=True)
            X_whitened = self.pca.fit_transform(X)
        else:
            X_whitened = X
            
            # Center data if not whitening
            self.mean_ = np.mean(X, axis=0)
            X_whitened = X - self.mean_
        
        # Initialize random weights
        W = np.random.rand(self.n_components, self.n_components)
        
        # Decorrelate and normalize weights
        W, _ = np.linalg.qr(W)
        
        # Iterate to find independent components
        for _ in range(self.max_iter):
            W_old = W.copy()
            
            # Fixed-point iteration for each component
            for i in range(self.n_components):
                w = W[i].reshape(self.n_components, 1)
                
                # Projection
                x_proj = X_whitened.dot(w)
                
                # Non-linear contrast and expectation
                gx, g_prime = self._g(x_proj)
                
                # Update weight vector
                w_new = (X_whitened.T.dot(gx) / n_samples - 
                        np.mean(g_prime) * w)
                
                # Decorrelate from other components
                for j in range(i):
                    w_j = W[j].reshape(self.n_components, 1)
                    w_new = w_new - w_new.T.dot(w_j) * w_j
                
                # Normalize
                w_new = w_new / np.sqrt(w_new.T.dot(w_new))
                
                # Update component
                W[i] = w_new.ravel()
            
            # Check convergence
            if np.max(np.abs(np.abs(np.diag(W.dot(W_old.T))) - 1)) < self.tol:
                break
        
        # Set components and mixing matrix
        self.components_ = W
        
        if self.whiten:
            # Account for PCA transformation
            self.components_ = self.components_.dot(self.pca.components_)
            
            # Compute the mixing matrix
            self.mixing_ = np.linalg.pinv(self.components_)
        else:
            # Compute the mixing matrix
            self.mixing_ = np.linalg.pinv(self.components_)
        
        return self
    
    def transform(self, X):
        """
        Transform data to independent component space.
        
        Args:
            X (np.ndarray): Data matrix [n_samples, n_features]
        
        Returns:
            np.ndarray: Transformed data [n_samples, n_components]
        """
        if self.whiten:
            X_whitened = self.pca.transform(X)
        else:
            X_whitened = X - self.mean_
            
        return X_whitened.dot(self.components_.T)
    
    def fit_transform(self, X):
        """
        Fit ICA to data and transform it.
        
        Args:
            X (np.ndarray): Data matrix [n_samples, n_features]
        
        Returns:
            np.ndarray: Transformed data [n_samples, n_components]
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Inverse transform data from independent component space.
        
        Args:
            X_transformed (np.ndarray): Transformed data [n_samples, n_components]
        
        Returns:
            np.ndarray: Original space data [n_samples, n_features]
        """
        X_orig = X_transformed.dot(self.mixing_.T)
        
        if self.whiten:
            return self.pca.inverse_transform(X_orig)
        else:
            return X_orig + self.mean_
    
    def save(self, filepath):
        """
        Save ICA model.
        
        Args:
            filepath (str): Path to save model
        """
        data_dict = {
            'components': self.components_,
            'mixing': self.mixing_,
            'n_components': self.n_components,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'whiten': self.whiten
        }
        
        if self.whiten:
            pca_dict = {
                'pca_components': self.pca.components_,
                'pca_mean': self.pca.mean_,
                'pca_scale': self.pca.scale_,
                'pca_explained_variance': self.pca.explained_variance_,
                'pca_explained_variance_ratio': self.pca.explained_variance_ratio_
            }
            data_dict.update(pca_dict)
        else:
            data_dict['mean'] = self.mean_
        
        np.savez(filepath, **data_dict)
    
    def load(self, filepath):
        """
        Load ICA model.
        
        Args:
            filepath (str): Path to load model from
        """
        data = np.load(filepath)
        self.components_ = data['components']
        self.mixing_ = data['mixing']
        self.n_components = int(data['n_components'])
        self.max_iter = int(data['max_iter'])
        self.tol = float(data['tol'])
        self.whiten = bool(data['whiten'])
        
        if self.whiten:
            self.pca = PCA(n_components=self.n_components, whiten=True)
            self.pca.components_ = data['pca_components']
            self.pca.mean_ = data['pca_mean']
            self.pca.scale_ = data['pca_scale']
            self.pca.explained_variance_ = data['pca_explained_variance']
            self.pca.explained_variance_ratio_ = data['pca_explained_variance_ratio']
        else:
            self.mean_ = data['mean']


class DimensionReducer:
    """
    Wrapper class for dimension reduction techniques (PCA/ICA) with classifier.
    """
    
    def __init__(self, reducer='pca', n_components=10, whiten=True, random_state=None, config=None):
        """
        Initialize dimension reducer with classifier.
        
        Args:
            reducer (str): Type of reducer ('pca' or 'ica')
            n_components (int): Number of components
            whiten (bool): Whether to whiten the data
            random_state (int): Random state for reproducibility
            config (dict): Configuration dictionary
        """
        self.reducer_type = reducer
        
        if config is not None:
            # Get configuration from YAML
            reducer_config = config['models']['pca_ica']
            n_components = reducer_config['n_components']
            whiten = reducer_config['whiten']
        
        # Initialize dimension reducer
        if reducer == 'pca':
            self.reducer = PCA(n_components=n_components, whiten=whiten)
        elif reducer == 'ica':
            self.reducer = FastICA(
                n_components=n_components, 
                whiten=whiten,
                max_iter=config['models']['pca_ica']['max_iter'] if config else 1000,
                tol=config['models']['pca_ica']['tol'] if config else 1e-4,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown reducer type: {reducer}")
        
        # For storing fitted classifier
        self.classifier = None
    
    def fit(self, X, y=None):
        """
        Fit dimension reducer to data.
        
        Args:
            X (np.ndarray): Data matrix [n_samples, n_features]
            y (np.ndarray, optional): Target values
        
        Returns:
            self: Returns self
        """
        self.reducer.fit(X)
        return self
    
    def transform(self, X):
        """
        Transform data.
        
        Args:
            X (np.ndarray): Data matrix [n_samples, n_features]
        
        Returns:
            np.ndarray: Transformed data [n_samples, n_components]
        """
        return self.reducer.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform data.
        
        Args:
            X (np.ndarray): Data matrix [n_samples, n_features]
            y (np.ndarray, optional): Target values
        
        Returns:
            np.ndarray: Transformed data [n_samples, n_components]
        """
        return self.reducer.fit_transform(X)
    
    def fit_classifier(self, X, y, classifier):
        """
        Fit classifier on transformed data.
        
        Args:
            X (np.ndarray): Data matrix [n_samples, n_features]
            y (np.ndarray): Target values
            classifier: Classifier object with fit and predict methods
        
        Returns:
            self: Returns self
        """
        # Transform data
        X_transformed = self.transform(X)
        
        # Fit classifier
        self.classifier = classifier
        self.classifier.fit(X_transformed, y)
        
        return self
    
    def predict(self, X):
        """
        Predict using fitted classifier.
        
        Args:
            X (np.ndarray): Data matrix [n_samples, n_features]
        
        Returns:
            np.ndarray: Predicted values
        """
        if self.classifier is None:
            raise ValueError("Classifier not fitted. Call fit_classifier first.")
        
        # Transform data and predict
        X_transformed = self.transform(X)
        return self.classifier.predict(X_transformed)
    
    def score(self, X, y):
        """
        Score fitted classifier.
        
        Args:
            X (np.ndarray): Data matrix [n_samples, n_features]
            y (np.ndarray): True values
        
        Returns:
            float: Accuracy score
        """
        if self.classifier is None:
            raise ValueError("Classifier not fitted. Call fit_classifier first.")
        
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def save(self, filepath):
        """
        Save reducer.
        
        Args:
            filepath (str): Path to save model
        """
        self.reducer.save(filepath)
    
    def load(self, filepath):
        """
        Load reducer.
        
        Args:
            filepath (str): Path to load model from
        """
        self.reducer.load(filepath) 