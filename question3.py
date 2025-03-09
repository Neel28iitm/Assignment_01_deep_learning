
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import wandb

# Initialize Weights & Biases
wandb.init(project="fashion-mnist-feedforward")

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize inputs
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten images
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# OHE labels
num_classes = 10
y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]

class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, optimizer="sgd", learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = {i: np.random.randn(self.layers[i], self.layers[i+1]) * 0.01 for i in range(len(self.layers)-1)}
        self.biases = {i: np.zeros((1, self.layers[i+1])) for i in range(len(self.layers)-1)}
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.beta1 = beta1  
        self.beta2 = beta2  
        self.epsilon = epsilon
        
        # Initialize momentum and adaptive learning rate terms
        self.v = {i: np.zeros_like(self.weights[i]) for i in range(len(self.layers)-1)}
        self.s = {i: np.zeros_like(self.weights[i]) for i in range(len(self.layers)-1)}
        self.t = 0  # Time step for Adam/Nadam

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        activations = {0: X}
        for i in range(len(self.layers) - 2):
            activations[i+1] = self.sigmoid(np.dot(activations[i], self.weights[i]) + self.biases[i])
        activations[len(self.layers)-1] = self.softmax(np.dot(activations[len(self.layers)-2], self.weights[len(self.layers)-2]) + self.biases[len(self.layers)-2])
        return activations

    def cross_entropy_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
    
    def backpropagation(self, X, y_true, activations):
        """
        Performs backpropagation to calculate gradients.
        """
        L = len(self.layers) - 1
        dZ = activations[L] - y_true  
        grads = {f"dW{L-1}": np.dot(activations[L-1].T, dZ) / X.shape[0],
                 f"db{L-1}": np.mean(dZ, axis=0, keepdims=True)}

        # Calculate gradients for hidden layers
        for i in range(L-2, -1, -1):
            dZ = np.dot(dZ, self.weights[i+1].T) * (activations[i+1] * (1 - activations[i+1]))  # Sigmoid derivative
            grads[f"dW{i}"] = np.dot(activations[i].T, dZ) / X.shape[0]
            grads[f"db{i}"] = np.mean(dZ, axis=0, keepdims=True)

        return grads    


    def update_weights(self, grads):
        self.t += 1  # Increment timestep for Adam/Nadam
        for i in range(len(self.layers) - 1):
            if self.optimizer == "sgd":
                self.weights[i] -= self.learning_rate * grads[f"dW{i}"]
                self.biases[i] -= self.learning_rate * grads[f"db{i}"]
            
            elif self.optimizer == "momentum":
                self.v[i] = self.beta1 * self.v[i] + (1 - self.beta1) * grads[f"dW{i}"]
                self.weights[i] -= self.learning_rate * self.v[i]
                self.biases[i] -= self.learning_rate * grads[f"db{i}"]
                
            elif self.optimizer == "nesterov":
                v_prev = self.v[i]
                self.v[i] = self.beta1 * self.v[i] + (1 - self.beta1) * grads[f"dW{i}"]
                self.weights[i] -= self.learning_rate * (self.beta1 * v_prev + (1 - self.beta1) * grads[f"dW{i}"])
                self.biases[i] -= self.learning_rate * grads[f"db{i}"]
                
            elif self.optimizer == "rmsprop":
                self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * grads[f"dW{i}"]**2
                self.weights[i] -= self.learning_rate * grads[f"dW{i}"] / (np.sqrt(self.s[i]) + self.epsilon)
                self.biases[i] -= self.learning_rate * grads[f"db{i}"]
                
            elif self.optimizer == "adam":
                self.v[i] = self.beta1 * self.v[i] + (1 - self.beta1) * grads[f"dW{i}"]
                self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * grads[f"dW{i}"]**2
                v_corr = self.v[i] / (1 - self.beta1**self.t)
                s_corr = self.s[i] / (1 - self.beta2**self.t)
                self.weights[i] -= self.learning_rate * v_corr / (np.sqrt(s_corr) + self.epsilon)
                self.biases[i] -= self.learning_rate * grads[f"db{i}"]
                
            elif self.optimizer == "nadam":
                v_corr = (self.beta1 * self.v[i] + (1 - self.beta1) * grads[f"dW{i}"]) / (1 - self.beta1**self.t)
                s_corr = self.s[i] / (1 - self.beta2**self.t)
                self.weights[i] -= self.learning_rate * v_corr / (np.sqrt(s_corr) + self.epsilon)
                self.biases[i] -= self.learning_rate * grads[f"db{i}"]
    
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                activations = self.forward(X_batch)
                loss = self.cross_entropy_loss(y_batch, activations[len(self.layers)-1])
                grads = self.backpropagation(X_batch, y_batch, activations)
                self.update_weights(grads)
            
            wandb.log({"Loss": loss})
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# Example usage
model = FeedforwardNeuralNetwork(input_size=784, hidden_layers=[128, 64], output_size=10, optimizer="adam")
model.train(X_train=x_train, y_train=y_train_onehot, epochs=10, batch_size=32)