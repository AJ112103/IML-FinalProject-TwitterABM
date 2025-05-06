import numpy as np

from ..models.base import (
    Layer, Linear, ReLU, Sigmoid, Dropout, Tanh,
    binary_crossentropy_loss, hinge_loss, Adam, SGD
)

class MLP:
    
    def __init__(self, input_shape, hidden_layers=[64, 32], dropout_rate=0.3, activation='relu'):

        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            self.activation = ReLU()  # Default

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
        output = X
        for layer in self.layers:
            if isinstance(layer, Dropout) and not training:
                continue
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
        grad = d_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def predict(self, X):
        return self.forward(X, training=False)
    
    def fit(self, X_train, y_train, validation_data=None, batch_size=32, learning_rate=0.001, 
            epochs=100, early_stopping_patience=5):

        param_layers = [layer for layer in self.layers if hasattr(layer, 'update')]
        self.optimizer = Adam(param_layers, learning_rate=learning_rate)

        metrics = []
        best_val_loss = float('inf')
        patience_counter = 0
 
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        if validation_data is not None:
            X_val, y_val = validation_data
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
        
        num_samples = X_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        # Training loop
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

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

            train_loss = epoch_loss / num_batches
            train_acc = epoch_correct / num_samples

            val_loss = None
            val_acc = None
            
            if validation_data is not None:
                val_predictions = self.predict(X_val)

                val_loss, _ = self.loss_fn(val_predictions, y_val)

                val_binary_preds = (val_predictions > 0.5).astype(int)
                val_acc = np.mean(val_binary_preds == y_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

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

    def __init__(self, config=None, kernel='rbf', C=1.0, gamma='scale', max_iter=1000):

        try:
            from sklearn.svm import SVC
        except ImportError:
            print("pip install scikit-learn")
            raise
        
        if config is not None:
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

        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            max_iter=self.max_iter,
            probability=True
        )
    
    def fit(self, X, y, X_val=None, y_val=None):

        if len(y.shape) > 1:
            y = y.flatten()

        self.model.fit(X, y)

        train_preds = self.model.predict(X)
        train_accuracy = np.mean(train_preds == y)

        val_accuracy = None
        if X_val is not None and y_val is not None:
            if len(y_val.shape) > 1:
                y_val = y_val.flatten()
            val_preds = self.model.predict(X_val)
            val_accuracy = np.mean(val_preds == y_val)
            print(f"Training accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}")
        else:
            print(f"Training accuracy: {train_accuracy:.4f}")

        metrics = [
            {
                'train_loss': 0.0,  
                'train_accuracy': float(train_accuracy),
                'val_loss': 0.0 if val_accuracy else None,
                'val_accuracy': float(val_accuracy) if val_accuracy else None
            }
        ]
        
        return metrics
    
    def predict(self, X):

        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)
    
    def save(self, filepath):

        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath):

        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f) 