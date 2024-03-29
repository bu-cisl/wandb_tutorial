import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

# Configuration dictionary
config = {
    "project_name": "wandb_mnist_tutorial",
    "entity": "cisl-bu", #name of wandb team
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 5,
    "feature_save_frequency": 2,  # Save features every n epochs
}

# Initialize wandb
wandb.init(project=config["project_name"], entity=config["entity"], config=config)

# Prepare the MNIST Dataset with transformation
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = datasets.MNIST("data", download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)

testset = datasets.MNIST("data", download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=config["batch_size"], shuffle=True)

# Pre-select a batch for logging feature images
fixed_batch_images, _ = next(iter(testloader))


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
        x1 = self.relu(self.conv1(x))
        x = self.max_pool(x1)
        x2 = self.relu(self.conv2(x))
        x = self.max_pool(x2)
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x, x2


# Initialize the model, loss, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

#logs the gradients
wandb.watch(model, log="all") 

# Training and Validation Loop
for epoch in range(config["epochs"]):
    model.train()
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()

        outputs, _ = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        validation_loss = 0
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                outputs, _ = model(images)
                val_loss = criterion(outputs, labels)
                validation_loss += val_loss.item()

        wandb.log(
            {
                "training_loss": running_loss / len(trainloader),
                "validation_loss": validation_loss / len(testloader),
            }
        )

    # Log the model as an artifact at the end of each epoch
    artifact = wandb.Artifact("model_epoch_latest", type="model")
    model_file = "model_latest.pth"
    torch.save(model.state_dict(), model_file)
    artifact.add_file(model_file)
    wandb.log_artifact(artifact)

    # Log feature maps for the pre-selected batch every n epochs
    if (epoch + 1) % config["feature_save_frequency"] == 0:
        model.eval()
        with torch.no_grad():
            _, features = model(fixed_batch_images)
            mip_images = features.max(dim=1)[0]
            wandb.log(
                {
                    f"feature_maps_mip_epoch_{epoch}": [
                        wandb.Image(mip_image) for mip_image in mip_images
                    ]
                }
            )

print(f"Finished Training. Model logged and tracked with wandb.")
