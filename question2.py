import numpy as np
import wandb
from tensorflow.keras.datasets import fashion_mnist

# Initialize WandB
wandb.init(project="fashion-mnist-ffnn", name="numpy-implementation")

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize inputs (0 to 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten images (28x28 → 784)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# One-hot encode labels
num_classes = 10
y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]

class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = {i: np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2.0 / self.layers[i]) 
                        for i in range(len(self.layers)-1)}
        self.biases = {i: np.zeros((1, self.layers[i+1])) for i in range(len(self.layers)-1)}

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        activations = {0: X}
        for i in range(len(self.layers) - 2):
            activations[i+1] = self.relu(np.dot(activations[i], self.weights[i]) + self.biases[i])
        activations[len(self.layers)-1] = self.softmax(np.dot(activations[len(self.layers)-2], self.weights[len(self.layers)-2]) + self.biases[len(self.layers)-2])
        return activations

    def cross_entropy_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def backpropagation(self, X, y_true, activations, learning_rate):
        L = len(self.layers) - 1
        dZ = activations[L] - y_true
        grads = {f"dW{L-1}": np.dot(activations[L-1].T, dZ) / X.shape[0],
                 f"db{L-1}": np.mean(dZ, axis=0, keepdims=True)}

        for i in range(L-2, -1, -1):
            dZ = np.dot(dZ, self.weights[i+1].T) * self.relu_derivative(activations[i+1])
            grads[f"dW{i}"] = np.dot(activations[i].T, dZ) / X.shape[0]
            grads[f"db{i}"] = np.mean(dZ, axis=0, keepdims=True)

        for i in range(L-1):
            self.weights[i] -= learning_rate * grads[f"dW{i}"]
            self.biases[i] -= learning_rate * grads[f"db{i}"]

    def train(self, X_train, y_train, epochs=50, batch_size=32, learning_rate=0.001):
        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                activations = self.forward(X_batch)
                loss = self.cross_entropy_loss(y_batch, activations[len(self.layers)-1])
                self.backpropagation(X_batch, y_batch, activations, learning_rate)

            # Log to WandB
            wandb.log({"epoch": epoch+1, "loss": loss})

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# Define hidden_layers
hidden_layers = [128, 64]  

# Create an instance of the model
input_size = x_train.shape[1]  
output_size = num_classes  
model = FeedforwardNeuralNetwork(input_size, hidden_layers, output_size)

# Train model
model.train(X_train=x_train, y_train=y_train_onehot, epochs=50, batch_size=32, learning_rate=0.001)

# Test accuracy
def evaluate(model, X_test, y_test):
    activations = model.forward(X_test)
    predictions = np.argmax(activations[len(model.layers)-1], axis=1)
    accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
    
    # Log final accuracy to WandB
    wandb.log({"Test Accuracy": accuracy * 100})

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

evaluate(model, x_test, y_test_onehot)
print("Script is running...")

