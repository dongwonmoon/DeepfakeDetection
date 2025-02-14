import os
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from custom_data import ImageDataset
from lora import LoRALinear
import matplotlib.pyplot as plt

torch.set_num_threads(torch.get_num_threads())


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Calculate training accuracy in batch
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Train Batch {batch_idx+1}/{len(dataloader)} - Loss: {running_loss / 10:.4f}"
            )
            running_loss = 0.0
    avg_loss = running_loss / len(dataloader)
    accuracy = total_correct / total_samples * 100.0
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100.0
    return avg_loss, accuracy


# Settings
json_file = "./dataset/DFDCP.json"
images_root = "./dataset"
batch_size = 1024
num_epochs = 5
learning_rate = 3e-4

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Dataset and DataLoader for training
train_dataset = ImageDataset(
    json_file, images_root, transform=transform, cur_type="train"
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Dataset and DataLoader for validation
val_dataset = ImageDataset(json_file, images_root, transform=transform, cur_type="val")
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load pretrained ViT model and modify the classification head with LoRA.
model = vit_b_16(pretrained=True)
# Freeze all parameters.
for param in model.parameters():
    param.requires_grad = False
orig_head = model.heads.head
in_features = orig_head.in_features
out_features = 2  # For binary classification

lora_head = LoRALinear(
    in_features,
    out_features,
    r=4,
    alpha=32,
    dropout=0.1,
    bias=(orig_head.bias is not None),
)
# Enable training for LoRA adapter parameters.
lora_head.A.requires_grad = True
lora_head.B.requires_grad = True
if lora_head.bias is not None:
    lora_head.bias.requires_grad = True

model.heads.head = lora_head
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
)

# Lists to track loss and accuracy for plotting.
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print("Starting training with LoRA fine-tuning and plotting...")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    # Training pass (calculating running loss inside train function is averaged per batch)
    train_loss_epoch, train_acc_epoch = train(
        model, train_dataloader, criterion, optimizer, device
    )
    train_losses.append(train_loss_epoch)
    train_accuracies.append(train_acc_epoch)

    # Validation pass
    val_loss_epoch, val_acc_epoch = validate(model, val_dataloader, criterion, device)
    val_losses.append(val_loss_epoch)
    val_accuracies.append(val_acc_epoch)

    print(f"Training - Loss: {train_loss_epoch:.4f}, Accuracy: {train_acc_epoch:.2f}%")
    print(f"Validation - Loss: {val_loss_epoch:.4f}, Accuracy: {val_acc_epoch:.2f}%")

    # Save the fine-tuned model checkpoint.
    checkpoint_path = f"./weights/lora/vit_finetuned_lora_{epoch}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training finished. Model saved to {checkpoint_path}")

# Plotting loss and accuracy curves.
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, "b-", label="Train Loss")
plt.plot(epochs, val_losses, "r-", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, "b-", label="Train Accuracy")
plt.plot(epochs, val_accuracies, "r-", label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy per Epoch")
plt.legend()

plot_path = "training_plots_vit_lora.png"
plt.savefig(plot_path)
print(f"Plots saved to {plot_path}")
