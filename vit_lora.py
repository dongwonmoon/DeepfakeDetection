import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from custom_data import ImageDataset
from base_vit import ViT
from lora import LoRA_ViT
import matplotlib.pyplot as plt


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
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

        print(f"Train Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")
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


if __name__ == "__main__":
    # 설정
    json_file = "./dataset/DFDCP.json"
    images_root = "./dataset"
    batch_size = 1024
    num_epochs = 10
    learning_rate = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 이미지 전처리
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # 학습 데이터 로더
    train_dataset = ImageDataset(
        json_file, images_root, transform=transform, cur_type="train"
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 검증 데이터 로더
    val_dataset = ImageDataset(
        json_file, images_root, transform=transform, cur_type="val"
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # pretrained ViT 모델 로드 및 LoRA로 분리(freeze)
    model = ViT("B_16", pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, out_features=2)
    lora_model = LoRA_ViT(model, r=4, alpha=4)
    num_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params/2**20:.4f}M")
    model = lora_model.to(device)
    model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 및 검증 손실, 정확도 기록
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print("Starting training with LoRA fine-tuning...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        # debug=True 옵션으로 각 배치의 gradient 정보를 출력
        train_loss_epoch, train_acc_epoch = train(
            model, train_dataloader, criterion, optimizer, device
        )
        train_losses.append(train_loss_epoch)
        train_accuracies.append(train_acc_epoch)

        val_loss_epoch, val_acc_epoch = validate(
            model, val_dataloader, criterion, device
        )
        val_losses.append(val_loss_epoch)
        val_accuracies.append(val_acc_epoch)

        print(
            f"Training   - Loss: {train_loss_epoch:.4f}, Accuracy: {train_acc_epoch:.2f}%"
        )
        print(
            f"Validation - Loss: {val_loss_epoch:.4f}, Accuracy: {val_acc_epoch:.2f}%"
        )

        checkpoint_path = f"./weights/lora/vit_finetuned_lora_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # 학습 결과를 시각화하여 저장
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, "b-", label="Train Loss")
    plt.plot(epochs_range, val_losses, "r-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, "b-", label="Train Accuracy")
    plt.plot(epochs_range, val_accuracies, "r-", label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy per Epoch")
    plt.legend()

    plot_path = "training_plots_vit_lora.png"
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")
