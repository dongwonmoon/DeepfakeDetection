import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from custom_data import ImageDataset
from lora import LoRALinear


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Batch {batch_idx+1}/{len(dataloader)} - Loss: {running_loss / 10:.4f}"
            )
            running_loss = 0.0


def main():
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

    # Dataset and DataLoader
    dataset = ImageDataset(
        json_file, images_root, transform=transform, cur_type="train"
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load pretrained ViT model and modify the classifier head with LoRA adapter for binary classification
    model = vit_b_16(pretrained=True)
    # Freeze the original model parameters except for LoRA adapters and classification head
    for param in model.parameters():
        param.requires_grad = False

    # Retrieve the original classification head
    orig_head = model.heads.head
    in_features = orig_head.in_features
    out_features = 2  # two classes: real and fake

    # Create a new head based on LoRA adapter; load the original weight into the frozen portion.
    lora_head = LoRALinear(
        in_features,
        out_features,
        r=4,
        alpha=32,
        dropout=0.1,
        bias=(orig_head.bias is not None),
    )
    lora_head.weight.data.copy_(orig_head.weight.data)
    if orig_head.bias is not None:
        lora_head.bias.data.copy_(orig_head.bias.data)
    # Enable gradients only for the LoRA adapters and bias
    lora_head.A.requires_grad = True
    lora_head.B.requires_grad = True
    if lora_head.bias is not None:
        lora_head.bias.requires_grad = True

    model.heads.head = lora_head
    model = model.to(device)

    # Loss and optimizer: optimizer only updates LoRA parameters and head bias if available.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )

    # Training loop
    print("Starting training with LoRA fine-tuning...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model, dataloader, criterion, optimizer, device)

    # Save the fine-tuned model checkpoint
    checkpoint_path = "vit_finetuned_lora.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training finished. Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
