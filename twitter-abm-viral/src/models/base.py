#!/usr/bin/env python3
import numpy as np

class Layer:
    
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def update(self, learning_rate):
        pass
    
    def get_params(self):
        return {}
    
    def set_params(self, params):
        pass


class Linear(Layer):
    
    def __init__(self, input_size, output_size, use_bias=True):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros(output_size) if use_bias else None
        self.use_bias = use_bias
        
        self.inputs = None
        self.grad_weights = None
        self.grad_bias = None
    
    def forward(self, inputs):
        self.inputs = inputs
        output = np.dot(inputs, self.weights)
        
        if self.use_bias:
            output += self.bias
        
        return output
    
    def backward(self, grad_output):
        self.grad_weights = np.dot(self.inputs.T, grad_output)
        
        if self.use_bias:
            self.grad_bias = np.sum(grad_output, axis=0)
        
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input
    
    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        
        if self.use_bias:
            self.bias -= learning_rate * self.grad_bias
    
    def get_params(self):
        return {
            'weights': self.weights,
            'bias': self.bias if self.use_bias else None
        }
    
    def set_params(self, params):
        self.weights = params['weights']
        self.bias = params['bias']


class Conv2DValid(Layer):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.kernels = np.random.randn(
            out_channels, in_channels, kernel_size[0], kernel_size[1]
        ) * np.sqrt(2.0 / (in_channels * kernel_size[0] * kernel_size[1]))
        
        self.bias = np.zeros(out_channels)
        
        self.inputs = None
        self.grad_kernels = None
        self.grad_bias = None
    
    def forward(self, inputs):
        self.inputs = inputs
        
        batch_size, in_channels, in_height, in_width = inputs.shape
        kernel_h, kernel_w = self.kernel_size
        
        out_height = (in_height - kernel_h) // self.stride + 1
        out_width = (in_width - kernel_w) // self.stride + 1
        
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(out_height):
                    h_start = h * self.stride
                    h_end = h_start + kernel_h
                    
                    for w in range(out_width):
                        w_start = w * self.stride
                        w_end = w_start + kernel_w
                        
                        patch = inputs[b, :, h_start:h_end, w_start:w_end]
                        
                        output[b, c_out, h, w] = np.sum(patch * self.kernels[c_out]) + self.bias[c_out]
        
        return output
    
    def backward(self, grad_output):
        batch_size, out_channels, out_height, out_width = grad_output.shape
        _, in_channels, in_height, in_width = self.inputs.shape
        kernel_h, kernel_w = self.kernel_size
        
        self.grad_kernels = np.zeros_like(self.kernels)
        self.grad_bias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(self.inputs)
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                self.grad_bias[c_out] += np.sum(grad_output[b, c_out])
                
                for h in range(out_height):
                    h_start = h * self.stride
                    h_end = h_start + kernel_h
                    
                    for w in range(out_width):
                        w_start = w * self.stride
                        w_end = w_start + kernel_w
                        
                        patch = self.inputs[b, :, h_start:h_end, w_start:w_end]
                        
                        self.grad_kernels[c_out] += patch * grad_output[b, c_out, h, w]
                        
                        grad_input[b, :, h_start:h_end, w_start:w_end] += self.kernels[c_out] * grad_output[b, c_out, h, w]
        
        return grad_input
    
    def update(self, learning_rate):
        self.kernels -= learning_rate * self.grad_kernels
        self.bias -= learning_rate * self.grad_bias
    
    def get_params(self):
        return {
            'kernels': self.kernels,
            'bias': self.bias
        }
    
    def set_params(self, params):
        self.kernels = params['kernels']
        self.bias = params['bias']


class Activation(Layer):
    
    def __init__(self):
        self.inputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        return self._activation(inputs)
    
    def backward(self, grad_output):
        return grad_output * self._gradient(self.inputs)
    
    def _activation(self, x):
        raise NotImplementedError
    
    def _gradient(self, x):
        raise NotImplementedError


class ReLU(Activation):
    
    def _activation(self, x):
        return np.maximum(0, x)
    
    def _gradient(self, x):
        return np.where(x > 0, 1, 0)


class Sigmoid(Activation):
    
    def _activation(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
    def _gradient(self, x):
        s = self._activation(x)
        return s * (1 - s)


class Tanh(Activation):
    
    def _activation(self, x):
        return np.tanh(x)
    
    def _gradient(self, x):
        return 1 - np.tanh(x) ** 2


class Softmax(Layer):
    
    def forward(self, inputs):
        shifted = inputs - np.max(inputs, axis=-1, keepdims=True)
        exp_values = np.exp(shifted)
        self.output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        return grad_output


class Flatten(Layer):
    
    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)


class Dropout(Layer):
    
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, inputs):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape) / (1 - self.dropout_rate)
            return inputs * self.mask
        else:
            return inputs
    
    def backward(self, grad_output):
        if self.training:
            return grad_output * self.mask
        else:
            return grad_output
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class PosEnc(Layer):
    
    def __init__(self, d_model, max_len=100):
        self.d_model = d_model
        self.max_len = max_len
        
        self.pe = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        self.pe[:, 0::2] = np.sin(pos * div_term)
        self.pe[:, 1::2] = np.cos(pos * div_term)
        
        self.pe = self.pe[np.newaxis, :, :]
    
    def forward(self, inputs):
        seq_len = inputs.shape[1]
        
        return inputs + self.pe[:, :seq_len, :]
    
    def backward(self, grad_output):
        return grad_output


class Optimizer:
    
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate
    
    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(self.learning_rate)


class SGD(Optimizer):
    
    def __init__(self, layers, learning_rate, momentum=0):
        super().__init__(layers, learning_rate)
        self.momentum = momentum
        self.velocities = {}
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_params'):
                params = layer.get_params()
                self.velocities[i] = {k: np.zeros_like(v) for k, v in params.items() if v is not None}
    
    def step(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'update') and hasattr(layer, 'get_params'):
                if hasattr(layer, 'grad_weights'):
                    self.velocities[i]['weights'] = self.momentum * self.velocities[i]['weights'] - self.learning_rate * layer.grad_weights
                    layer.weights += self.velocities[i]['weights']
                
                if hasattr(layer, 'grad_bias') and layer.bias is not None:
                    self.velocities[i]['bias'] = self.momentum * self.velocities[i]['bias'] - self.learning_rate * layer.grad_bias
                    layer.bias += self.velocities[i]['bias']
                
                if hasattr(layer, 'grad_kernels'):
                    self.velocities[i]['kernels'] = self.momentum * self.velocities[i]['kernels'] - self.learning_rate * layer.grad_kernels
                    layer.kernels += self.velocities[i]['kernels']


class Adam(Optimizer):
    
    def __init__(self, layers, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
        self.m = {}
        self.v = {}
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_params'):
                params = layer.get_params()
                self.m[i] = {k: np.zeros_like(v) for k, v in params.items() if v is not None}
                self.v[i] = {k: np.zeros_like(v) for k, v in params.items() if v is not None}
    
    def step(self):
        self.t += 1
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'update') and hasattr(layer, 'get_params'):
                if hasattr(layer, 'grad_weights'):
                    self.m[i]['weights'] = self.beta1 * self.m[i]['weights'] + (1 - self.beta1) * layer.grad_weights
                    self.v[i]['weights'] = self.beta2 * self.v[i]['weights'] + (1 - self.beta2) * (layer.grad_weights ** 2)
                    
                    m_hat = self.m[i]['weights'] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[i]['weights'] / (1 - self.beta2 ** self.t)
                    
                    layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                
                if hasattr(layer, 'grad_bias') and layer.bias is not None:
                    self.m[i]['bias'] = self.beta1 * self.m[i]['bias'] + (1 - self.beta1) * layer.grad_bias
                    self.v[i]['bias'] = self.beta2 * self.v[i]['bias'] + (1 - self.beta2) * (layer.grad_bias ** 2)
                    
                    m_hat = self.m[i]['bias'] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[i]['bias'] / (1 - self.beta2 ** self.t)
                    
                    layer.bias -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                
                if hasattr(layer, 'grad_kernels'):
                    self.m[i]['kernels'] = self.beta1 * self.m[i]['kernels'] + (1 - self.beta1) * layer.grad_kernels
                    self.v[i]['kernels'] = self.beta2 * self.v[i]['kernels'] + (1 - self.beta2) * (layer.grad_kernels ** 2)
                    
                    m_hat = self.m[i]['kernels'] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[i]['kernels'] / (1 - self.beta2 ** self.t)
                    
                    layer.kernels -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def mse_loss(y_pred, y_true):
    m = y_pred.shape[0]
    loss = np.mean((y_pred - y_true) ** 2)
    gradient = 2 * (y_pred - y_true) / m
    return loss, gradient


def binary_crossentropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    gradient = (y_pred - y_true) / (y_pred * (1 - y_pred) + epsilon) / m
    return loss, gradient


def categorical_crossentropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / m
    gradient = -y_true / y_pred / m
    return loss, gradient


def hinge_loss(y_pred, y_true):
    m = y_pred.shape[0]
    margin = 1 - y_true * y_pred
    loss = np.mean(np.maximum(0, margin))
    
    gradient = np.zeros_like(y_pred)
    mask = margin > 0
    gradient[mask] = -y_true[mask] / m
    
    return loss, gradient 