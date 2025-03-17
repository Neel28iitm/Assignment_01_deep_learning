import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define the model
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Train function
def train_model(loss_function, loss_name):
    model = NeuralNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, test_losses, accuracies = [], [], []
    criterion = loss_function
    
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            
            if loss_name == "CrossEntropy":
                loss = criterion(outputs, labels)
            else:  # Squared Error Loss requires one-hot encoding
                labels_one_hot = torch.eye(10)[labels]  # Convert labels to one-hot
                loss = criterion(torch.softmax(outputs, dim=1), labels_one_hot)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        correct, total, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if loss_name == "CrossEntropy":
                    loss = criterion(outputs, labels)
                else:
                    labels_one_hot = torch.eye(10)[labels]
                    loss = criterion(torch.softmax(outputs, dim=1), labels_one_hot)

                test_loss += loss.item()
        
        test_losses.append(test_loss / len(test_loader))
        accuracies.append(correct / total)

        print(f"Epoch {epoch+1}, {loss_name} Loss: {train_losses[-1]:.4f}, Test Accuracy: {accuracies[-1]:.4f}")

    return train_losses, test_losses, accuracies

# Train with both loss functions
train_losses_ce, test_losses_ce, acc_ce = train_model(nn.CrossEntropyLoss(), "CrossEntropy")
train_losses_se, test_losses_se, acc_se = train_model(nn.MSELoss(), "SquaredError")

# Plot comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses_ce, label='CrossEntropy Loss')
plt.plot(train_losses_se, label='Squared Error Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_ce, label='CrossEntropy Accuracy')
plt.plot(acc_se, label='Squared Error Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.title('Accuracy Comparison')
plt.legend()

plt.show()
