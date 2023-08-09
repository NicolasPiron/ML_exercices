# Example usage
import numpy as np
import DNN_class as dnn

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
nn = dnn.NeuralNetwork(layers=[2, 2, 1], activation='sigmoid')

# Train the neural network
nn.train(X, y, epochs=1000, learning_rate=0.1)

# Make predictions
predictions = nn.predict(X)
print("Predictions:")
print(predictions)