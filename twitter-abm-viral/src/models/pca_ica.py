import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

class PCA:

    def __init__(self, n_components=10, whiten=False):

        self.n_components = n_components
        self.whiten = whiten
        self.components_ = None
        self.mean_ = None
        self.scale_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.components_ = Vt[:self.n_components]

        n_samples = X.shape[0]
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)

        if self.whiten:
            self.scale_ = np.sqrt(self.explained_variance_[:self.n_components])
        
        return self
    
    def transform(self, X):

        X_centered = X - self.mean_
        
        if self.whiten:
            return X_centered.dot(self.components_.T) / self.scale_
        else:
            return X_centered.dot(self.components_.T)
    
    def fit_transform(self, X):

        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):

        if self.whiten:
            return X_transformed * self.scale_.dot(self.components_) + self.mean_
        else:
            return X_transformed.dot(self.components_) + self.mean_
    
    def save(self, filepath):

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

        data = np.load(filepath)
        self.components_ = data['components']
        self.mean_ = data['mean']
        self.scale_ = data['scale']
        self.explained_variance_ = data['explained_variance']
        self.explained_variance_ratio_ = data['explained_variance_ratio']
        self.n_components = int(data['n_components'])
        self.whiten = bool(data['whiten'])


class FastICA:

    def __init__(self, n_components=10, max_iter=1000, tol=1e-4, whiten=True, random_state=None):

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.whiten = whiten
        self.random_state = random_state

        self.pca = None

        self.components_ = None
        self.mixing_ = None
    
    def _g(self, x):

        gx = np.tanh(x)
        g_prime = 1 - gx ** 2
        return gx, g_prime
    
    def fit(self, X):

        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.whiten:
            self.pca = PCA(n_components=self.n_components, whiten=True)
            X_whitened = self.pca.fit_transform(X)
        else:
            X_whitened = X

            self.mean_ = np.mean(X, axis=0)
            X_whitened = X - self.mean_

        W = np.random.rand(self.n_components, self.n_components)

        W, _ = np.linalg.qr(W)

        for _ in range(self.max_iter):
            W_old = W.copy()

            for i in range(self.n_components):
                w = W[i].reshape(self.n_components, 1)

                x_proj = X_whitened.dot(w)

                gx, g_prime = self._g(x_proj)

                w_new = (X_whitened.T.dot(gx) / n_samples - 
                        np.mean(g_prime) * w)

                for j in range(i):
                    w_j = W[j].reshape(self.n_components, 1)
                    w_new = w_new - w_new.T.dot(w_j) * w_j

                w_new = w_new / np.sqrt(w_new.T.dot(w_new))

                W[i] = w_new.ravel()

            if np.max(np.abs(np.abs(np.diag(W.dot(W_old.T))) - 1)) < self.tol:
                break

        self.components_ = W
        
        if self.whiten:
            self.components_ = self.components_.dot(self.pca.components_)

            self.mixing_ = np.linalg.pinv(self.components_)
        else:
            self.mixing_ = np.linalg.pinv(self.components_)
        
        return self
    
    def transform(self, X):
        if self.whiten:
            X_whitened = self.pca.transform(X)
        else:
            X_whitened = X - self.mean_
            
        return X_whitened.dot(self.components_.T)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        X_orig = X_transformed.dot(self.mixing_.T)
        
        if self.whiten:
            return self.pca.inverse_transform(X_orig)
        else:
            return X_orig + self.mean_
    
    def save(self, filepath):

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

    def __init__(self, reducer='pca', n_components=10, whiten=True, random_state=None, config=None):

        self.reducer_type = reducer
        
        if config is not None:
            reducer_config = config['models']['pca_ica']
            n_components = reducer_config['n_components']
            whiten = reducer_config['whiten']

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

        self.reducer.fit(X)
        return self
    
    def transform(self, X):

        return self.reducer.transform(X)
    
    def fit_transform(self, X, y=None):

        return self.reducer.fit_transform(X)
    
    def fit_classifier(self, X, y, classifier):

        X_transformed = self.transform(X)

        self.classifier = classifier
        self.classifier.fit(X_transformed, y)
        
        return self
    
    def predict(self, X):

        if self.classifier is None:
            raise ValueError("Classifier not fitted. Call fit_classifier first.")

        X_transformed = self.transform(X)
        return self.classifier.predict(X_transformed)
    
    def score(self, X, y):

        if self.classifier is None:
            raise ValueError("Classifier not fitted. Call fit_classifier first.")
        
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def save(self, filepath):

        self.reducer.save(filepath)
    
    def load(self, filepath):

        self.reducer.load(filepath) 