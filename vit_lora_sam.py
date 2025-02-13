import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from custom_data import ImageDataset
from lora import LoRALinear
from sam_optimizer import SAM


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Define a closure that re-computes the loss
        def closure():
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            return loss

        loss = optimizer.step(closure)
        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Batch {batch_idx+1}/{len(dataloader)} - Loss: {running_loss / 10:.4f}"
            )
            running_loss = 0.0


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

# Dataset and DataLoader
dataset = ImageDataset(json_file, images_root, transform=transform, cur_type="train")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = vit_b_16(pretrained=True)
# Freeze all original parameters.
for param in model.parameters():
    param.requires_grad = False

orig_head = model.heads.head
in_features = orig_head.in_features
out_features = 2

lora_head = LoRALinear(
    in_features,
    out_features,
    r=4,
    alpha=32,
    dropout=0.1,
    bias=(orig_head.bias is not None),
)
lora_head.A.requires_grad = True
lora_head.B.requires_grad = True
if lora_head.bias is not None:
    lora_head.bias.requires_grad = True

model.heads.head = lora_head
model = model.to(device)

criterion = nn.CrossEntropyLoss()
base_optimizer = optim.SGD
optimizer = SAM(
    filter(lambda p: p.requires_grad, model.parameters()),
    base_optimizer,
    rho=0.05,
    adaptive=False,
    lr=learning_rate,
)

# Training loop
print("Starting training with LoRA + SAM fine-tuning...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train(model, dataloader, criterion, optimizer, device)

# Save the fine-tuned model checkpoint
checkpoint_path = "vit_finetuned_lora_sam.pth"
torch.save(model.state_dict(), checkpoint_path)
print(f"Training finished. Model saved to {checkpoint_path}")
