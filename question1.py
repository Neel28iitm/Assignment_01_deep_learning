import numpy as np
import matplotlib.pyplot as plt
import wandb
from keras.datasets import fashion_mnist

# Initialize WandB project
wandb.init(project="fashion_mnist_visualization")

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Class labels for Fashion-MNIST
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Find one sample per class
samples = []
for class_label in range(10):
    index = np.where(y_train == class_label)[0][0]  # First occurrence of each class
    samples.append((x_train[index], class_names[class_label]))

# Log images to WandB
wandb.log({"examples": [wandb.Image(img, caption=label) for img, label in samples]})

wandb.finish()
