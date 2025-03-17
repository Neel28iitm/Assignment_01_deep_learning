import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb

# Define the neural network class
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, activation):
        super(FeedForwardNN, self).__init__()
        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            in_features = hidden_size
        layers.append(nn.Linear(hidden_size, 10))  # Output layer
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x.view(x.size(0), -1))

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a feedforward neural network on MNIST/Fashion-MNIST")
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-l", "--loss", type=str, choices=["mse", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "xavier"], default="random")
    parser.add_argument("-nhl", "--num_layers", type=int, default=2)
    parser.add_argument("-sz", "--hidden_size", type=int, default=128)
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "relu"], default="relu")
    return parser.parse_args()

# Main training function
def train():
    args = parse_args()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
    train_loader = torch.utils.data.DataLoader(dataset("./data", train=True, download=True, transform=transform), batch_size=args.batch_size, shuffle=True)
    
    # Model setup
    model = FeedForwardNN(28*28, args.hidden_size, args.num_layers, args.activation)
    
    # Optimizer
    optimizer = {
        "sgd": optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay),
        "adam": optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.epsilon),
        "rmsprop": optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.9, eps=args.epsilon),
    }[args.optimizer]
    
    # Loss function
    criterion = nn.CrossEntropyLoss() if args.loss == "cross_entropy" else nn.MSELoss()
    
    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch+1, "loss": avg_loss})
    
    print("Training complete!")
    wandb.finish()

if __name__ == "__main__":
    train()
