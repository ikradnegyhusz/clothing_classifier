import numpy as np

class NeuralNetwork:
    # initialize the neural network with layer sizes, learning rate, and dropout rate
    def __init__(self, layer_sizes, learning_rate=0.001, dropout_rate=0.5):
        self.layers = len(layer_sizes) - 1  # number of layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weights = []  # store weights for each layer
        self.biases = []   # store biases for each layer

        # initialize weights using he initialization and biases with zeros
        for i in range(self.layers):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    # relu activation function
    def relu(self, z):
        return np.maximum(0, z)

    # derivative of relu function
    def relu_derivative(self, z):
        return (z > 0).astype(float)

    # softmax activation function for output layer
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # forward propagation through the network
    def forward(self, X, training=True):
        self.a = [X]  # store activations for each layer
        for i in range(self.layers - 1):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            a = self.relu(z)

            # apply dropout during training only
            if training:
                dropout_mask = np.random.rand(*a.shape) > self.dropout_rate
                a *= dropout_mask
                a /= (1 - self.dropout_rate)

            self.a.append(a)  # store activation

        # compute output layer with softmax
        z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.a.append(self.softmax(z))
        return self.a[-1]

    # backward propagation through the network
    def backward(self, X, y):
        m = y.shape[0]  # number of samples

        # one-hot encode labels
        y_one_hot = np.eye(self.a[-1].shape[1])[y.astype(int)]

        # calculate gradients for output layer
        dz = self.a[-1] - y_one_hot
        gradients_w = []  # gradients for weights
        gradients_b = []  # gradients for biases

        # backpropagation through hidden layers
        for i in range(self.layers - 1, 0, -1):
            dW = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)

            dz = np.dot(dz, self.weights[i].T) * self.relu_derivative(self.a[i])

        # compute gradients for first layer
        dW = np.dot(X.T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        gradients_w.insert(0, dW)
        gradients_b.insert(0, db)

        # update weights and biases using computed gradients
        for i in range(self.layers):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    # train the neural network using mini-batch gradient descent
    def train(self, X, y, epochs=100, batch_size=64):
        for epoch in range(epochs):
            # shuffle data for each epoch
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            # process data in batches
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[indices[i:i + batch_size]]
                y_batch = y[indices[i:i + batch_size]]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch)

            # calculate and print loss every 10 epochs
            if epoch % 10 == 0:
                loss = -np.mean(np.log(output[range(y_batch.size), y_batch.astype(int)]))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # make predictions using the trained network
    def predict(self, X):
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)

    # compute accuracy of predictions
    def accuracy(self, predictions, labels):
        return np.mean(predictions == labels)
