import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb


def main():
    # Initialize wandb and retrieve hyperparameters from the sweep
    with wandb.init():
        config = wandb.config

        # Prepare the MNIST Dataset with transformation
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        trainset = datasets.MNIST("data", download=True, train=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)

        testset = datasets.MNIST("data", download=True, train=False, transform=transform)
        testloader = DataLoader(testset, batch_size=config.batch_size, shuffle=True)

        # Define the CNN Model
        class CNNClassifier(nn.Module):
            def __init__(self):
                super(CNNClassifier, self).__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
                self.fc1 = nn.Linear(32 * 7 * 7, 128)
                self.fc2 = nn.Linear(128, 10)
                self.max_pool = nn.MaxPool2d(2, 2)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.max_pool(x)
                x = self.relu(self.conv2(x))
                x = self.max_pool(x)
                x = x.view(-1, 32 * 7 * 7)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # Initialize the model, loss, and optimizer
        model = CNNClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Logs the gradients, parameters, and model topology
        wandb.watch(model, log="all")

        # Training loop
        for epoch in range(10):
            model.train()
            for images, labels in trainloader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation loop
            model.eval()
            total, correct = 0, 0
            with torch.no_grad():
                for images, labels in testloader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Logging metrics
            wandb.log({"epoch": epoch, "loss": loss.item(), "accuracy": correct / total})

if __name__ == "__main__":
    main()
