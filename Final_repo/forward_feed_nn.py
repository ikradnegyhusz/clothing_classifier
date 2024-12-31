import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.001, dropout_rate=0.5):
        # Initialize parameters
        self.layers = len(layer_sizes) - 1
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []

        # He initialization for weights
        for i in range(self.layers):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X, training=True):
        self.a = [X]
        for i in range(self.layers - 1):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            a = self.relu(z)

            # Apply dropout only during training
            if training:
                dropout_mask = np.random.rand(*a.shape) > self.dropout_rate
                a *= dropout_mask
                a /= (1 - self.dropout_rate)

            self.a.append(a)

        z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.a.append(self.softmax(z))
        return self.a[-1]


    def backward(self, X, y):
        m = y.shape[0]
        y_one_hot = np.eye(self.a[-1].shape[1])[y.astype(int)]

        # Compute gradients for output layer
        dz = self.a[-1] - y_one_hot
        gradients_w = []
        gradients_b = []

        # Backpropagation through layers
        for i in range(self.layers - 1, 0, -1):
            dW = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)

            dz = np.dot(dz, self.weights[i].T) * self.relu_derivative(self.a[i])

        # Compute gradients for first layer
        dW = np.dot(X.T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        gradients_w.insert(0, dW)
        gradients_b.insert(0, db)

        # Update weights and biases using gradients
        for i in range(self.layers):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def train(self, X, y, epochs=100, batch_size=64):
        for epoch in range(epochs):
            # Mini-batch gradient descent
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[indices[i:i + batch_size]]
                y_batch = y[indices[i:i + batch_size]]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch)

            if epoch % 10 == 0:
                loss = -np.mean(np.log(output[range(y_batch.size), y_batch.astype(int)]))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)

    def accuracy(self, predictions, labels):
        return np.mean(predictions == labels)