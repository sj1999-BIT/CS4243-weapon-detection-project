import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
from tqdm import tqdm  # Import tqdm

# Set device to GPU if available
if torch.cuda.is_available():
    print("GPU is available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# global variables
training_data_path = "./data/train/" # path to the training data
train_test_ratio = 0.8 # ratio to divide the training dataset for training and testing
batch_size = 32 # number of img per batch

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

complete_dataset = ImageFolder(training_data_path, transform=transform)

# Split dataset into training and validation sets
train_size = int(train_test_ratio * len(complete_dataset))
val_size = len(complete_dataset) - train_size
train_dataset, val_dataset = random_split(complete_dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# define model for untrained resnet
# model = models.resnet50(pretrained=False)
# model_name = "untrained_resnet50"
# model.to(device) # make sure all data and model in same device
# save_model_dir = "torch_models/" + model_name +".pth" # define directory to save the model.

# define model for untrained googlenet
model = models.googlenet(pretrained=False)
model_name = "untrained_googlenet"
model.to(device) # make sure all data and model in same device
save_model_dir = "torch_models/" + model_name +".pth" # define directory to save the model.

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create a SummaryWriter for logging
log_dir = "logs/" + model_name
writer = SummaryWriter(log_dir)

# Training loop
print("start training")

for epoch in range(10):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch + 1}/{10}:")

    # Use tqdm to create a progress bar
    with tqdm(total=len(train_loader), unit="batch") as progress_bar:
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # # for resnet
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)

            # for googlenet
            outputs = model(inputs)
            predictions = outputs.logits
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            progress_bar.update(1)  # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Log loss to TensorBoard
            global_step = epoch * len(train_loader) + i
            writer.add_scalar("Loss/train", loss.item(), global_step)

    average_loss = running_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1}/{10}, Average Loss: {average_loss:.4f}\n")

    # Testing loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}\n")
    writer.add_scalar("Accuracy/test", accuracy, epoch)

# Save the trained model
torch.save(model.state_dict(), save_model_dir)