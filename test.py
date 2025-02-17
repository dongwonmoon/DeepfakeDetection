import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from custom_data import ImageDataset
from base_vit import ViT
from lora import LoRA_ViT
import matplotlib.pyplot as plt


def test(model, dataloader, criterion, device):
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
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 이미지 전처리
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # 테스트 데이터 로더
    test_dataset = ImageDataset(
        json_file, images_root, transform=transform, cur_type="test"
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # pretrained ViT 모델 로드 및 LoRA로 분리(freeze)
    model = ViT("B_16", pretrained=True)
    lora_model = LoRA_ViT(model, r=4, alpha=4)
    in_features = lora_model.lora_vit.fc.in_features
    lora_model.lora_vit.fc = nn.Linear(in_features, out_features=2)
    model = lora_model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(
        torch.load("./weights/lora/vit_finetuned_lora_1.pth", map_location=device)
    )

    criterion = nn.CrossEntropyLoss()

    print("Starting test with LoRA fine-tuning...")

    test_loss, test_acc = test(model, test_dataloader, criterion, device)
    print(f"Validation - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
