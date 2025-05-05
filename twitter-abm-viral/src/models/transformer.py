#!/usr/bin/env python3
"""
Transformer model for processing tweet cascade time series.
Uses encoder-only architecture with self-attention.
"""

import numpy as np
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..models.base import (
    Layer, Linear, Dropout, ReLU, PosEnc, Tanh, Sigmoid,
    mse_loss, binary_crossentropy_loss, Adam
)

class MultiHeadAttention(Layer):
    """
    Multi-head attention layer for Transformer.
    Implements scaled dot-product attention with multiple heads.
    """
    
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        """
        Initialize multi-head attention layer.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            dropout_rate (float): Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        # Linear projections for Q, K, V
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = Dropout(dropout_rate)
        
        # Cache for backward pass
        self.q = None
        self.k = None
        self.v = None
        self.attention_weights = None
    
    def _reshape_for_multihead(self, x):
        """
        Reshape tensor for multi-head attention.
        
        Args:
            x (np.ndarray): Input tensor [batch_size, seq_len, embed_dim]
        
        Returns:
            np.ndarray: Reshaped tensor [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
    
    def _reshape_from_multihead(self, x):
        """
        Reshape tensor back from multi-head format.
        
        Args:
            x (np.ndarray): Multi-head tensor [batch_size, num_heads, seq_len, head_dim]
        
        Returns:
            np.ndarray: Reshaped tensor [batch_size, seq_len, embed_dim]
        """
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
    
    def forward(self, query, key=None, value=None, mask=None):
        """
        Forward pass.
        
        Args:
            query (np.ndarray): Query tensor [batch_size, seq_len, embed_dim]
            key (np.ndarray, optional): Key tensor (defaults to query)
            value (np.ndarray, optional): Value tensor (defaults to key)
            mask (np.ndarray, optional): Attention mask
        
        Returns:
            np.ndarray: Output tensor [batch_size, seq_len, embed_dim]
        """
        # Default to self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = key
        
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape
        
        # Linear projections
        self.query = self.q_proj.forward(query)
        self.key = self.k_proj.forward(key)
        self.value = self.v_proj.forward(value)
        
        # Reshape for multi-head attention
        q = self._reshape_for_multihead(self.query)  # [batch, num_heads, seq_len_q, head_dim]
        k = self._reshape_for_multihead(self.key)    # [batch, num_heads, seq_len_k, head_dim]
        v = self._reshape_for_multihead(self.value)  # [batch, num_heads, seq_len_k, head_dim]
        
        # Scaled dot-product attention
        # Transpose key for matrix multiplication: [batch, num_heads, head_dim, seq_len_k]
        k_t = k.transpose(0, 1, 3, 2)
        
        # Compute attention scores: [batch, num_heads, seq_len_q, seq_len_k]
        scores = np.matmul(q, k_t) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax to get attention weights
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / (np.sum(attention_weights, axis=-1, keepdims=True) + 1e-9)
        
        # Apply dropout to attention weights
        self.attention_weights = self.dropout.forward(attention_weights)
        
        # Apply attention to values
        # [batch, num_heads, seq_len_q, head_dim]
        context = np.matmul(self.attention_weights, v)
        
        # Reshape back to [batch, seq_len_q, embed_dim]
        context = self._reshape_from_multihead(context)
        
        # Final linear projection
        output = self.output_proj.forward(context)
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass.
        
        Args:
            grad_output (np.ndarray): Gradient from next layer [batch_size, seq_len, embed_dim]
        
        Returns:
            np.ndarray: Gradient to previous layer [batch_size, seq_len, embed_dim]
        """
        # Backward through output projection
        grad_context = self.output_proj.backward(grad_output)
        
        # Reshape to multi-head format
        batch_size, seq_len_q, _ = grad_context.shape
        grad_context_multihead = self._reshape_for_multihead(grad_context)
        
        # Reshape stored values for backprop
        v_multihead = self._reshape_for_multihead(self.value)
        
        # Gradient of attention weights
        # [batch, num_heads, seq_len_q, seq_len_k]
        grad_attention_weights = np.matmul(grad_context_multihead, v_multihead.transpose(0, 1, 3, 2))
        
        # Backward through dropout
        grad_attention_weights = self.dropout.backward(grad_attention_weights)
        
        # Gradient through softmax
        # Simplified gradient calculation for softmax
        grad_scores = self.attention_weights * (grad_attention_weights - 
                                              np.sum(grad_attention_weights * self.attention_weights,
                                                     axis=-1, keepdims=True))
        
        # Scale by sqrt(d_k)
        grad_scores = grad_scores / np.sqrt(self.head_dim)
        
        # Gradient of values
        # [batch, num_heads, seq_len_k, head_dim]
        grad_v_multihead = np.matmul(self.attention_weights.transpose(0, 1, 3, 2), grad_context_multihead)
        grad_v = self._reshape_from_multihead(grad_v_multihead)
        
        # Gradient of keys
        # [batch, num_heads, seq_len_k, head_dim]
        q_multihead = self._reshape_for_multihead(self.query)
        grad_k_multihead = np.matmul(grad_scores.transpose(0, 1, 3, 2), q_multihead)
        grad_k = self._reshape_from_multihead(grad_k_multihead)
        
        # Gradient of queries
        # [batch, num_heads, seq_len_q, head_dim]
        k_multihead = self._reshape_for_multihead(self.key)
        grad_q_multihead = np.matmul(grad_scores, k_multihead)
        grad_q = self._reshape_from_multihead(grad_q_multihead)
        
        # Backward through linear projections
        grad_value = self.v_proj.backward(grad_v)
        grad_key = self.k_proj.backward(grad_k)
        grad_query = self.q_proj.backward(grad_q)
        
        # For self-attention, sum the gradients
        if np.array_equal(self.query, self.key) and np.array_equal(self.key, self.value):
            return grad_query + grad_key + grad_value
        
        return grad_query
    
    def update(self, learning_rate):
        """
        Update parameters.
        
        Args:
            learning_rate (float): Learning rate
        """
        self.q_proj.update(learning_rate)
        self.k_proj.update(learning_rate)
        self.v_proj.update(learning_rate)
        self.output_proj.update(learning_rate)
    
    def get_params(self):
        """Get layer parameters."""
        return {
            'q_proj': self.q_proj.get_params(),
            'k_proj': self.k_proj.get_params(),
            'v_proj': self.v_proj.get_params(),
            'output_proj': self.output_proj.get_params()
        }
    
    def set_params(self, params):
        """Set layer parameters."""
        self.q_proj.set_params(params['q_proj'])
        self.k_proj.set_params(params['k_proj'])
        self.v_proj.set_params(params['v_proj'])
        self.output_proj.set_params(params['output_proj'])


class FeedForward(Layer):
    """
    Feed-forward network for Transformer.
    Consists of two linear layers with ReLU activation.
    """
    
    def __init__(self, embed_dim, ff_dim, dropout_rate=0.1):
        """
        Initialize feed-forward network.
        
        Args:
            embed_dim (int): Embedding dimension
            ff_dim (int): Hidden dimension of feed-forward network
            dropout_rate (float): Dropout rate
        """
        super().__init__()
        
        # First linear layer
        self.linear1 = Linear(embed_dim, ff_dim)
        self.activation = ReLU()
        
        # Second linear layer
        self.linear2 = Linear(ff_dim, embed_dim)
        
        # Dropout
        self.dropout = Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (np.ndarray): Input tensor [batch_size, seq_len, embed_dim]
        
        Returns:
            np.ndarray: Output tensor [batch_size, seq_len, embed_dim]
        """
        # First linear layer with activation
        x = self.linear1.forward(x)
        x = self.activation.forward(x)
        
        # Dropout
        x = self.dropout.forward(x)
        
        # Second linear layer
        x = self.linear2.forward(x)
        
        return x
    
    def backward(self, grad_output):
        """
        Backward pass.
        
        Args:
            grad_output (np.ndarray): Gradient from next layer
        
        Returns:
            np.ndarray: Gradient to previous layer
        """
        # Backward through linear layers and activation
        grad_output = self.linear2.backward(grad_output)
        grad_output = self.dropout.backward(grad_output)
        grad_output = self.activation.backward(grad_output)
        grad_output = self.linear1.backward(grad_output)
        
        return grad_output
    
    def update(self, learning_rate):
        """
        Update parameters.
        
        Args:
            learning_rate (float): Learning rate
        """
        self.linear1.update(learning_rate)
        self.linear2.update(learning_rate)
    
    def get_params(self):
        """Get layer parameters."""
        return {
            'linear1': self.linear1.get_params(),
            'linear2': self.linear2.get_params()
        }
    
    def set_params(self, params):
        """Set layer parameters."""
        self.linear1.set_params(params['linear1'])
        self.linear2.set_params(params['linear2'])


class TransformerEncoderLayer(Layer):
    """
    Transformer encoder layer.
    Consists of multi-head self-attention and feed-forward network.
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        """
        Initialize Transformer encoder layer.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Hidden dimension of feed-forward network
            dropout_rate (float): Dropout rate
        """
        super().__init__()
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        
        # Feed-forward network
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout_rate)
        
        # Layer normalization
        self.norm1_scale = np.ones(embed_dim)
        self.norm1_bias = np.zeros(embed_dim)
        self.norm2_scale = np.ones(embed_dim)
        self.norm2_bias = np.zeros(embed_dim)
        
        # Dropout
        self.dropout = Dropout(dropout_rate)
        
        # For backward pass
        self.norm1_mean = None
        self.norm1_var = None
        self.norm1_input = None
        self.norm1_normalized = None
        self.norm2_mean = None
        self.norm2_var = None
        self.norm2_input = None
        self.norm2_normalized = None
        self.attn_output = None
        self.ff_output = None
    
    def _layer_norm(self, x, scale, bias, epsilon=1e-5):
        """
        Layer normalization.
        
        Args:
            x (np.ndarray): Input tensor [batch_size, seq_len, embed_dim]
            scale (np.ndarray): Scale parameter [embed_dim]
            bias (np.ndarray): Bias parameter [embed_dim]
            epsilon (float): Small constant for numerical stability
        
        Returns:
            tuple: (Normalized output, mean, variance, input)
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + epsilon)
        output = normalized * scale + bias
        
        return output, mean, var, x, normalized
    
    def _layer_norm_backward(self, grad_output, mean, var, input_data, normalized, scale, epsilon=1e-5):
        """
        Backward pass for layer normalization.
        
        Args:
            grad_output (np.ndarray): Gradient from next layer
            mean (np.ndarray): Mean used in forward pass
            var (np.ndarray): Variance used in forward pass
            input_data (np.ndarray): Input data in forward pass
            normalized (np.ndarray): Normalized data in forward pass
            scale (np.ndarray): Scale parameter
            epsilon (float): Small constant for numerical stability
        
        Returns:
            tuple: (Gradient to input, gradient to scale, gradient to bias)
        """
        # Extract dimensions
        batch_size, seq_len, embed_dim = grad_output.shape
        
        # Gradient to bias is sum over batch and sequence dimensions
        grad_bias = np.sum(grad_output, axis=(0, 1))
        
        # Gradient to scale
        grad_scale = np.sum(grad_output * normalized, axis=(0, 1))
        
        # Gradient to normalized input
        grad_normalized = grad_output * scale
        
        # Gradient through normalization
        std_inv = 1.0 / np.sqrt(var + epsilon)
        
        # Gradient to input data
        grad_input = (1.0 / (batch_size * seq_len)) * std_inv * (
            batch_size * seq_len * grad_normalized - 
            np.sum(grad_normalized, axis=-1, keepdims=True) - 
            normalized * np.sum(grad_normalized * normalized, axis=-1, keepdims=True)
        )
        
        return grad_input, grad_scale, grad_bias
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x (np.ndarray): Input tensor [batch_size, seq_len, embed_dim]
            mask (np.ndarray, optional): Attention mask
        
        Returns:
            np.ndarray: Output tensor [batch_size, seq_len, embed_dim]
        """
        # Self-attention block
        # Layer normalization
        normed_x, self.norm1_mean, self.norm1_var, self.norm1_input, self.norm1_normalized = \
            self._layer_norm(x, self.norm1_scale, self.norm1_bias)
        
        # Multi-head self-attention
        attn_output = self.self_attn.forward(normed_x, mask=mask)
        
        # Residual connection and dropout
        self.attn_output = x + self.dropout.forward(attn_output)
        
        # Feed-forward block
        # Layer normalization
        normed_attn, self.norm2_mean, self.norm2_var, self.norm2_input, self.norm2_normalized = \
            self._layer_norm(self.attn_output, self.norm2_scale, self.norm2_bias)
        
        # Feed-forward network
        ff_output = self.feed_forward.forward(normed_attn)
        
        # Residual connection and dropout
        self.ff_output = self.attn_output + self.dropout.forward(ff_output)
        
        return self.ff_output
    
    def backward(self, grad_output):
        """
        Backward pass.
        
        Args:
            grad_output (np.ndarray): Gradient from next layer
        
        Returns:
            np.ndarray: Gradient to previous layer
        """
        # Backward through feed-forward residual block
        grad_ff_output = grad_output
        grad_attn_output = grad_output
        
        # Backward through dropout in feed-forward residual
        grad_ff = self.dropout.backward(grad_ff_output)
        
        # Backward through feed-forward network
        grad_normed_attn = self.feed_forward.backward(grad_ff)
        
        # Backward through layer norm before feed-forward
        grad_attn_output_from_ff, grad_norm2_scale, grad_norm2_bias = \
            self._layer_norm_backward(
                grad_normed_attn, self.norm2_mean, self.norm2_var,
                self.norm2_input, self.norm2_normalized, self.norm2_scale
            )
        
        # Add gradient from residual connection
        grad_attn_output += grad_attn_output_from_ff
        
        # Backward through attention residual block
        grad_x = grad_attn_output
        
        # Backward through dropout in attention residual
        grad_attn = self.dropout.backward(grad_attn_output)
        
        # Backward through multi-head attention
        grad_normed_x = self.self_attn.backward(grad_attn)
        
        # Backward through layer norm before attention
        grad_x_from_attn, grad_norm1_scale, grad_norm1_bias = \
            self._layer_norm_backward(
                grad_normed_x, self.norm1_mean, self.norm1_var,
                self.norm1_input, self.norm1_normalized, self.norm1_scale
            )
        
        # Add gradient from residual connection
        grad_x += grad_x_from_attn
        
        # Update layer norm parameters
        self.norm1_scale -= 0.01 * grad_norm1_scale
        self.norm1_bias -= 0.01 * grad_norm1_bias
        self.norm2_scale -= 0.01 * grad_norm2_scale
        self.norm2_bias -= 0.01 * grad_norm2_bias
        
        return grad_x
    
    def update(self, learning_rate):
        """
        Update parameters.
        
        Args:
            learning_rate (float): Learning rate
        """
        self.self_attn.update(learning_rate)
        self.feed_forward.update(learning_rate)
    
    def get_params(self):
        """Get layer parameters."""
        return {
            'self_attn': self.self_attn.get_params(),
            'feed_forward': self.feed_forward.get_params(),
            'norm1_scale': self.norm1_scale,
            'norm1_bias': self.norm1_bias,
            'norm2_scale': self.norm2_scale,
            'norm2_bias': self.norm2_bias
        }
    
    def set_params(self, params):
        """Set layer parameters."""
        self.self_attn.set_params(params['self_attn'])
        self.feed_forward.set_params(params['feed_forward'])
        self.norm1_scale = params['norm1_scale']
        self.norm1_bias = params['norm1_bias']
        self.norm2_scale = params['norm2_scale']
        self.norm2_bias = params['norm2_bias']


class TransformerEncoder(Layer):
    """
    Transformer encoder.
    Stack of TransformerEncoderLayer.
    """
    
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate=0.1, max_seq_len=100):
        """
        Initialize Transformer encoder.
        
        Args:
            num_layers (int): Number of encoder layers
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Hidden dimension of feed-forward network
            dropout_rate (float): Dropout rate
            max_seq_len (int): Maximum sequence length for positional encoding
        """
        super().__init__()
        
        # Positional encoding
        self.pos_enc = PosEnc(embed_dim, max_seq_len)
        
        # Dropout
        self.dropout = Dropout(dropout_rate)
        
        # Stack of encoder layers
        self.layers = [
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x (np.ndarray): Input tensor [batch_size, seq_len, embed_dim]
            mask (np.ndarray, optional): Attention mask
        
        Returns:
            np.ndarray: Output tensor [batch_size, seq_len, embed_dim]
        """
        # Add positional encoding
        x = self.pos_enc.forward(x)
        
        # Apply dropout
        x = self.dropout.forward(x)
        
        # Process through encoder layers
        for layer in self.layers:
            x = layer.forward(x, mask)
        
        return x
    
    def backward(self, grad_output):
        """
        Backward pass.
        
        Args:
            grad_output (np.ndarray): Gradient from next layer
        
        Returns:
            np.ndarray: Gradient to previous layer
        """
        # Backward through encoder layers
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        
        # Backward through dropout
        grad_output = self.dropout.backward(grad_output)
        
        # Backward through positional encoding
        grad_output = self.pos_enc.backward(grad_output)
        
        return grad_output
    
    def update(self, learning_rate):
        """
        Update parameters.
        
        Args:
            learning_rate (float): Learning rate
        """
        for layer in self.layers:
            layer.update(learning_rate)
    
    def get_params(self):
        """Get layer parameters."""
        return {
            'layers': [layer.get_params() for layer in self.layers]
        }
    
    def set_params(self, params):
        """Set layer parameters."""
        for i, layer_params in enumerate(params['layers']):
            self.layers[i].set_params(layer_params)


class TransformerModel:
    """
    Transformer model for tweet cascade prediction.
    Uses encoder-only architecture.
    """
    
    def __init__(self, config):
        """
        Initialize Transformer model.
        
        Args:
            config (dict): Model configuration
        """
        # Get model configuration
        self.config = config['models']['transformer']
        
        # Model dimensions
        self.input_dim = 24  # 24-hour time series
        self.embed_dim = self.config.get('embed_dim', self.config.get('d_model', 64))
        self.num_heads = self.config.get('num_heads', 4)
        self.ff_dim = self.config.get('ff_dim', self.config.get('d_ff', 256))
        self.dropout_rate = self.config.get('dropout_rate', self.config.get('dropout', 0.1))
        self.num_layers = self.config.get('num_layers', 2)
        
        # Training parameters
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.epochs = self.config.get('epochs', 50)
        
        # Build model
        self.layers = self._build_model()
        
        # Set up optimizer
        self.optimizer = Adam(
            self.layers, 
            learning_rate=self.learning_rate
        )
        
        # For tracking training
        self.train_losses = []
        self.val_losses = []
    
    def _build_model(self):
        """
        Build model architecture.
        
        Returns:
            list: Model layers
        """
        layers = []
        
        # Input embedding
        layers.append(Linear(1, self.embed_dim))
        
        # Transformer encoder
        layers.append(TransformerEncoder(
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate,
            max_seq_len=self.input_dim
        ))
        
        # Global average pooling
        # (Implemented in forward pass)
        
        # Output layer
        layers.append(Linear(self.embed_dim, 1))
        layers.append(Sigmoid())
        
        return layers
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X (np.ndarray): Input data [batch_size, seq_len(24)]
        
        Returns:
            np.ndarray: Model predictions
        """
        batch_size, seq_len = X.shape
        
        # Reshape input to [batch_size, seq_len, 1]
        X = X.reshape(batch_size, seq_len, 1)
        
        # Forward through input embedding
        embeddings = self.layers[0].forward(X)
        
        # Forward through transformer encoder
        encoder_output = self.layers[1].forward(embeddings)
        
        # Global average pooling
        pooled = np.mean(encoder_output, axis=1)
        
        # Forward through output layer
        output = self.layers[2].forward(pooled)
        output = self.layers[3].forward(output)
        
        return output
    
    def backward(self, gradient):
        """
        Backward pass through the network.
        
        Args:
            gradient (np.ndarray): Gradient from loss function [batch_size, 1]
        
        Returns:
            np.ndarray: Gradient to input
        """
        batch_size = gradient.shape[0]
        
        # Backward through output layers
        gradient = self.layers[3].backward(gradient)
        gradient = self.layers[2].backward(gradient)
        
        # Reshape for encoder (un-pooling by broadcasting)
        # This distributes the gradient equally to all sequence positions
        gradient = np.tile(gradient[:, np.newaxis, :], (1, self.input_dim, 1)) / self.input_dim
        
        # Backward through transformer encoder
        gradient = self.layers[1].backward(gradient)
        
        # Backward through input embedding
        gradient = self.layers[0].backward(gradient)
        
        # Reshape to match input shape
        return gradient.reshape(batch_size, self.input_dim)
    
    def train_step(self, X_batch, y_batch):
        """
        Perform a single training step.
        
        Args:
            X_batch (np.ndarray): Batch of training data [batch_size, seq_len]
            y_batch (np.ndarray): Batch of training labels [batch_size, 1]
        
        Returns:
            float: Loss for this batch
        """
        # Forward pass
        y_pred = self.forward(X_batch)
        
        # Compute loss and gradient
        loss, gradient = binary_crossentropy_loss(y_pred, y_batch)
        
        # Backward pass
        self.backward(gradient)
        
        # Update weights
        self.optimizer.step()
        
        return loss
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None):
        """
        Train the model - alias for fit method to maintain consistent API.
        
        Args:
            X_train (np.ndarray): Training data [n_samples, seq_len]
            y_train (np.ndarray): Training labels [n_samples, 1]
            X_val (np.ndarray, optional): Validation data
            y_val (np.ndarray, optional): Validation labels
            epochs (int, optional): Number of epochs (overrides config)
            batch_size (int, optional): Batch size (overrides config)
        
        Returns:
            dict: Training history
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            
        return self.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=self.learning_rate,
            early_stopping_patience=5  # Default value
        )
    
    def predict(self, X):
        """
        Generate predictions for input data.
        
        Args:
            X (np.ndarray): Input data [n_samples, seq_len]
        
        Returns:
            np.ndarray: Model predictions [n_samples, 1]
        """
        # Set layers to evaluation mode
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
        
        # Forward pass
        return self.forward(X)
    
    def save(self, filepath):
        """
        Save model parameters.
        
        Args:
            filepath (str): Path to save model
        """
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_params'):
                params[f'layer_{i}'] = layer.get_params()
        
        np.savez(filepath, **params)
    
    def load(self, filepath):
        """
        Load model parameters.
        
        Args:
            filepath (str): Path to load model from
        """
        params = np.load(filepath, allow_pickle=True)
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'set_params') and f'layer_{i}' in params:
                layer.set_params(params[f'layer_{i}'].item()) 