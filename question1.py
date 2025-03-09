import numpy as np
import matplotlib.pyplot as plt
import wandb
from keras.datasets import fashion_mnist

# Class labels for Fashion-MNIST
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def get_sample_images(x_train, y_train):
    """Fetch one sample image per class."""
    samples = []
    for class_label in range(10):
        index = np.where(y_train == class_label)[0][0]  
        samples.append((x_train[index], class_names[class_label]))
    return samples

if __name__ == "__main__":
    # Initialize WandB 
    wandb.init(project="fashion_mnist_visualization", name="fashion_mnist_samples")

    # Load dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # sample image per class
    samples = get_sample_images(x_train, y_train)

    # Log images to WandB with captions
    images_to_log = [wandb.Image(img, caption=label) for img, label in samples]
    wandb.log({"fashion_mnist_examples": images_to_log})

    # Finish WandB run
    wandb.finish()
