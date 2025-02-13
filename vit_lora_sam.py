import os
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
import matplotlib.pyplot as plt

from custom_data import ImageDataset
from lora import LoRALinear
from sam_optimizer import SAM


def train(model, dataloader, criterion, optimizer, device):
    """
    Training function using SAM optimizer.
    For each batch, a closure is defined to compute the loss.
    Tracks cumulative loss and accuracy per epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Define a closure for recalculating the loss.
        def closure():
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            return loss

        loss = optimizer.step(closure)
        loss_value = loss.item()
        total_loss += loss_value * images.size(0)

        # Compute accuracy for this batch.
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Train Batch {batch_idx+1}/{len(dataloader)} - Current Batch Loss: {loss_value:.4f}"
            )
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100.0
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    Validation function to compute loss and accuracy over the validation set.
    """
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
batch_size = 128
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

# Datasets and DataLoaders for training and validation
train_dataset = ImageDataset(
    json_file, images_root, transform=transform, cur_type="train"
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ImageDataset(json_file, images_root, transform=transform, cur_type="val")
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load pretrained ViT model and modify the classification head with a LoRA adapter.
model = vit_b_16(pretrained=True)
# Freeze all parameters of the ViT backbone.
for param in model.parameters():
    param.requires_grad = False

orig_head = model.heads.head
in_features = orig_head.in_features
out_features = 2  # For binary classification (e.g., real vs. fake)

lora_head = LoRALinear(
    in_features,
    out_features,
    r=4,
    alpha=32,
    dropout=0.1,
    bias=(orig_head.bias is not None),
)
# Enable gradients only for the LoRA adapter matrices and bias if present.
lora_head.A.requires_grad = True
lora_head.B.requires_grad = True
if lora_head.bias is not None:
    lora_head.bias.requires_grad = True

model.heads.head = lora_head
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Use SAM optimizer wrapped around a base optimizer (e.g., SGD). Only trainable parameters are updated.
base_optimizer = optim.SGD
optimizer = SAM(
    filter(lambda p: p.requires_grad, model.parameters()),
    base_optimizer,
    rho=0.05,
    adaptive=False,
    lr=learning_rate,
)

# Lists for tracking epoch-wise losses and accuracies
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print("Starting training with LoRA + SAM fine-tuning and plotting...")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    # Training phase
    train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # Validation phase
    val_loss, val_acc = validate(model, val_dataloader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

# Plotting training and validation curves
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))

# Plot Loss Curves
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, "b-o", label="Train Loss")
plt.plot(epochs, val_losses, "r-o", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()

# Plot Accuracy Curves
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, "b-s", label="Train Accuracy")
plt.plot(epochs, val_accuracies, "r-s", label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy per Epoch")
plt.legend()

plot_path = "training_plots_vit_lora_sam.png"
plt.savefig(plot_path)
print(f"Training plots saved to {plot_path}")

# Save the fine-tuned model's state
checkpoint_path = "vit_finetuned_lora_sam.pth"
torch.save(model.state_dict(), checkpoint_path)
print(f"Training finished. Model saved to {checkpoint_path}")
