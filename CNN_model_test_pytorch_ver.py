import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm

# global variables
testing_data_path = "./data/test/" # path to the training data

# Set device to GPU if available
if torch.cuda.is_available():
    print("GPU is available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ImageFolder(testing_data_path, transform=transform)

# Create data loaders for training and validation
batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# define model

model_name = "untrained_googlenet"
save_model_dir = "torch_models/" + model_name +".pth" # define directory to save the model.

# Load saved model checkpoint
model = models.googlenet(pretrained=False)
model.load_state_dict(torch.load(save_model_dir))
model.to(device) # make sure all data and model in same device

# Training loop
print("start testing")

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    progress_bar = tqdm(total=len(test_loader), unit="batch")  # Initialize tqdm progress bar
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.update(1)  # Update progress bar
        progress_bar.set_postfix(accuracy=f"{correct / total:.4f}")

progress_bar.close()  # Close progress bar after testing loop

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}\n")

