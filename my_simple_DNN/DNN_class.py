import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation='sigmoid'):
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = [np.random.randn(layers[i], layers[i+1]) for i in range(self.num_layers - 1)]
        self.biases = [np.random.randn(1, layers[i+1]) for i in range(self.num_layers - 1)]
        self.activation = activation

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward_propagation(self, X):
        activations = [X]
        for i in range(self.num_layers - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if self.activation == 'sigmoid':
                activation = self.sigmoid(z)
            elif self.activation == 'relu':
                activation = self.relu(z)
            activations.append(activation)
        return activations

    def backward_propagation(self, X, y, activations):
        gradients = []
        deltas = [None] * self.num_layers
        if self.activation == 'sigmoid':
            activation_derivative = self.sigmoid_derivative
        elif self.activation == 'relu':
            activation_derivative = self.relu_derivative

        error = y - activations[-1]
        delta = error * activation_derivative(activations[-1])
        deltas[-1] = delta

        for i in reversed(range(self.num_layers - 1)):
            delta = np.dot(deltas[i + 1], self.weights[i].T) * activation_derivative(activations[i + 1])
            deltas[i] = delta

        for i in range(self.num_layers - 1):
            gradient = np.dot(activations[i].T, deltas[i + 1])
            gradients.append(gradient)

        return gradients, deltas

    def update_parameters(self, gradients, learning_rate):
        for i in range(self.num_layers - 1):
            self.weights[i] += learning_rate * gradients[i]
            self.biases[i] += learning_rate * np.sum(gradients[i], axis=0)

    def train(self, X, y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            activations = self.forward_propagation(X)
            gradients, _ = self.backward_propagation(X, y, activations)
            self.update_parameters(gradients, learning_rate)

    def predict(self, X):
        activations = self.forward_propagation(X)
        return activations[-1]
