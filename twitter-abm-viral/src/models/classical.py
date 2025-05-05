#!/usr/bin/env python3
"""
Classical ML algorithms for tweet virality prediction.
Implements MLP and SVM from scratch using NumPy.
"""

import numpy as np

from ..models.base import (
    Layer, Linear, ReLU, Sigmoid, Dropout, Tanh,
    binary_crossentropy_loss, hinge_loss, Adam, SGD
)

class MLP:
    """
    Multi-Layer Perceptron (MLP) for virality prediction.
    Pure NumPy implementation.
    """
    
    def __init__(self, input_shape, hidden_layers=[64, 32], dropout_rate=0.3, activation='relu'):
        """
        Initialize MLP model.
        
        Args:
            input_shape (int): Number of input features
            hidden_layers (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate (0-1)
            activation (str): Activation function ('relu' or 'tanh')
        """
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        
        # Set activation function
        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            self.activation = ReLU()  # Default
        
        # Build network layers
        self.layers = []
        
        # Input layer to first hidden layer
        self.layers.append(Linear(input_shape, hidden_layers[0]))
        self.layers.append(self.activation)
        self.layers.append(Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(self.activation)
            self.layers.append(Dropout(dropout_rate))
        
        # Output layer (binary classification)
        self.layers.append(Linear(hidden_layers[-1], 1))
        self.layers.append(Sigmoid())
        
        # Optimizer and loss function
        self.optimizer = None
        self.loss_fn = binary_crossentropy_loss
        
    def forward(self, X, training=True):
        """Forward pass through the network."""
        output = X
        for layer in self.layers:
            if isinstance(layer, Dropout) and not training:
                continue  # Skip dropout during inference
            if hasattr(layer, 'training'):
                output = layer.forward(output)
                if training:
                    layer.training = True
                else:
                    layer.training = False
            else:
                output = layer.forward(output)
        return output
    
    def backward(self, d_loss):
        """Backward pass through the network."""
        grad = d_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def predict(self, X):
        """Predict class probabilities."""
        return self.forward(X, training=False)
    
    def fit(self, X_train, y_train, validation_data=None, batch_size=32, learning_rate=0.001, 
            epochs=100, early_stopping_patience=5):
        """
        Train the MLP model.
        
        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
            validation_data (tuple): Validation data and labels
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            epochs (int): Number of epochs
            early_stopping_patience (int): Patience for early stopping
            
        Returns:
            list: Training metrics
        """
        # Initialize optimizer with the layers that have parameters
        param_layers = [layer for layer in self.layers if hasattr(layer, 'update')]
        self.optimizer = Adam(param_layers, learning_rate=learning_rate)
        
        # Initialize metrics tracking
        metrics = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Reshape y if needed
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        
        # If validation data is provided, prepare it
        if validation_data is not None:
            X_val, y_val = validation_data
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
        
        num_samples = X_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Batch training
            epoch_loss = 0
            epoch_correct = 0
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, num_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(X_batch)
                
                # Calculate loss
                loss, d_loss = self.loss_fn(predictions, y_batch)
                epoch_loss += loss
                
                # Count correct predictions
                binary_preds = (predictions > 0.5).astype(int)
                epoch_correct += np.sum(binary_preds == y_batch)
                
                # Backward pass
                self.backward(d_loss)
                
                # Update weights
                self.optimizer.step()
            
            # Calculate training metrics
            train_loss = epoch_loss / num_batches
            train_acc = epoch_correct / num_samples
            
            # Validation
            val_loss = None
            val_acc = None
            
            if validation_data is not None:
                # Forward pass on validation data
                val_predictions = self.predict(X_val)
                
                # Calculate validation loss
                val_loss, _ = self.loss_fn(val_predictions, y_val)
                
                # Calculate validation accuracy
                val_binary_preds = (val_predictions > 0.5).astype(int)
                val_acc = np.mean(val_binary_preds == y_val)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Log metrics
            metrics.append({
                'train_loss': float(train_loss),
                'train_accuracy': float(train_acc),
                'val_loss': float(val_loss) if val_loss is not None else None,
                'val_accuracy': float(val_acc) if val_acc is not None else None
            })
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}", end="")
                if validation_data is not None:
                    print(f", val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
                else:
                    print("")
        
        return metrics


class SVM:
    """
    Support Vector Machine wrapper for scikit-learn's SVM.
    """
    
    def __init__(self, config=None, kernel='rbf', C=1.0, gamma='scale', max_iter=1000):
        """
        Initialize SVM model.
        
        Args:
            config (dict): Configuration dictionary
            kernel (str): Kernel type ('linear', 'rbf', 'poly')
            C (float): Regularization parameter
            gamma (str or float): Kernel coefficient
            max_iter (int): Maximum number of iterations
        """
        # Import sklearn here to avoid dependency issues
        try:
            from sklearn.svm import SVC
        except ImportError:
            print("scikit-learn is required for SVM. Please install: pip install scikit-learn")
            raise
        
        if config is not None:
            # Get configuration from YAML
            svm_config = config['models']['svm']
            self.kernel = svm_config.get('kernel', 'rbf')
            self.C = float(svm_config.get('C', 1.0))
            self.gamma = svm_config.get('gamma', 'scale')
            self.max_iter = int(svm_config.get('max_iter', 1000))
        else:
            self.kernel = kernel
            self.C = float(C)
            self.gamma = gamma
            self.max_iter = int(max_iter)
        
        # Initialize sklearn SVM
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            max_iter=self.max_iter,
            probability=True
        )
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit SVM model.
        
        Args:
            X (np.ndarray): Training data [n_samples, n_features]
            y (np.ndarray): Training labels [n_samples]
            X_val (np.ndarray, optional): Validation data
            y_val (np.ndarray, optional): Validation labels
        
        Returns:
            list: Dictionary of metrics for compatibility with other models
        """
        # Make sure y is flattened
        if len(y.shape) > 1:
            y = y.flatten()
        
        # Fit the model
        self.model.fit(X, y)
        
        # Calculate training accuracy
        train_preds = self.model.predict(X)
        train_accuracy = np.mean(train_preds == y)
        
        # Calculate validation accuracy if provided
        val_accuracy = None
        if X_val is not None and y_val is not None:
            if len(y_val.shape) > 1:
                y_val = y_val.flatten()
            val_preds = self.model.predict(X_val)
            val_accuracy = np.mean(val_preds == y_val)
            print(f"Training accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}")
        else:
            print(f"Training accuracy: {train_accuracy:.4f}")
        
        # Return metrics in a format compatible with other models
        metrics = [
            {
                'train_loss': 0.0,  # SVM doesn't calculate loss in the same way
                'train_accuracy': float(train_accuracy),
                'val_loss': 0.0 if val_accuracy else None,
                'val_accuracy': float(val_accuracy) if val_accuracy else None
            }
        ]
        
        return metrics
    
    def predict(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (np.ndarray): Input data [n_samples, n_features]
        
        Returns:
            np.ndarray: Predicted probabilities [n_samples, 1]
        """
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)
    
    def save(self, filepath):
        """
        Save model.
        
        Args:
            filepath (str): Path to save model
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath):
        """
        Load model.
        
        Args:
            filepath (str): Path to load model from
        """
        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f) 