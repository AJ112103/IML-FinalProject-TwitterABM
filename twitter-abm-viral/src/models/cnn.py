import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..models.base import (
    Layer, Conv2DValid, ReLU, Sigmoid, Flatten, Linear, Dropout,
    binary_crossentropy_loss, Adam
)

class CNN:

    def __init__(self, config):

        self.config = config['models']['cnn']
        self.input_shape = (1, 24, 1)  # (channels, time steps, features)
        self.filters = self.config.get('filters', [32, 64])
        self.kernel_sizes = self.config.get('kernel_sizes', [3, 3])
        self.pool_sizes = self.config.get('pool_sizes', [2, 2])
        self.fc_layers = self.config.get("fc_layers",
                                self.config.get("dense_layers", [64, 32]))
        self.dropout_rate = self.config.get("dropout_rate", 0.3)
        
        # Training parameters
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.epochs = self.config.get('epochs', 50)

        self.layers = self._build_model()

        self.optimizer = Adam(
            self.layers, 
            learning_rate=self.learning_rate
        )

        self.train_losses = []
        self.val_losses = []
    
    def _build_model(self):

        layers = []
        
        # First convolutional layer
        layers.append(Conv2DValid(
            in_channels=self.input_shape[0],
            out_channels=self.filters[0],
            kernel_size=self.kernel_sizes[0]
        ))
        layers.append(ReLU())
        
        # Second convolutional layer
        layers.append(Conv2DValid(
            in_channels=self.filters[0],
            out_channels=self.filters[1],
            kernel_size=self.kernel_sizes[1]
        ))
        layers.append(ReLU())
        
        # Flatten for fully connected layers
        layers.append(Flatten())
        
        # First FC layer
        layers.append(Linear(
            input_size=self.filters[1] * (self.input_shape[1] - self.kernel_sizes[0] - self.kernel_sizes[1] + 2),
            output_size=self.fc_layers[0]
        ))
        layers.append(ReLU())
        layers.append(Dropout(self.dropout_rate))
        
        # Second FC layer
        layers.append(Linear(
            input_size=self.fc_layers[0],
            output_size=self.fc_layers[1]
        ))
        layers.append(ReLU())
        layers.append(Dropout(self.dropout_rate))
        
        # Output layer (binary classification)
        layers.append(Linear(
            input_size=self.fc_layers[1],
            output_size=1
        ))
        layers.append(Sigmoid())
        
        return layers
    
    def forward(self, X):

        X = X.reshape(X.shape[0], 1, X.shape[1], 1)

        activations = X
        for layer in self.layers:
            activations = layer.forward(activations)
        
        return activations
    
    def backward(self, gradient):

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        
        return gradient
    
    def train_step(self, X_batch, y_batch):

        y_pred = self.forward(X_batch)
        
        # Compute loss and gradient
        loss, gradient = binary_crossentropy_loss(y_pred, y_batch)
        
        # Backward pass
        self.backward(gradient)
        
        # Update weights
        self.optimizer.step()
        
        return loss
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None):

        if epochs is None:
            epochs = self.config['epochs']
        if batch_size is None:
            batch_size = self.config['batch_size']
        
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Training loop
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for layer in self.layers:
                if hasattr(layer, 'train'):
                    layer.train()

            epoch_loss = 0
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                batch_loss = self.train_step(X_batch, y_batch)
                epoch_loss += batch_loss
            
            epoch_loss /= n_batches
            self.train_losses.append(epoch_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                for layer in self.layers:
                    if hasattr(layer, 'eval'):
                        layer.eval()
                
                # Compute validation loss
                y_val_pred = self.forward(X_val)
                val_loss, _ = binary_crossentropy_loss(y_val_pred, y_val)
                self.val_losses.append(val_loss)
                
                print(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}")
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
    
    def predict(self, X):
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
        
        # Forward pass
        return self.forward(X)
    
    def save(self, filepath):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_params'):
                params[f'layer_{i}'] = layer.get_params()
        
        np.savez(filepath, **params)
    
    def load(self, filepath):
        params = np.load(filepath, allow_pickle=True)
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'set_params') and f'layer_{i}' in params:
                layer.set_params(params[f'layer_{i}'].item()) 