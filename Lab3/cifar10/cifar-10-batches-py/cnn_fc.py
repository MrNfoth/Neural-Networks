from datetime import datetime
from pathlib import Path
import random
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

# --------------------------------------------------------
# Фиксация seed для повторяемости
# --------------------------------------------------------
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# --------------------------------------------------------
# GPU-режим (строго)
# --------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется устройство:", DEVICE.type)


class ConvNet(nn.Module):
    """Сверточная нейронная сеть для CIFAR-10."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_name(self) -> str:
        return "conv"


class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),

            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)

    def get_name(self):
        return "linear_dropout03"

def get_data_loaders(
    batch_size: int = 50,
    train_subset_size: int = 10000,
    valid_size: int = 5000,
    data_dir: str = "data",
):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )

    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    full_train = torch.utils.data.Subset(full_train, list(range(train_subset_size)))

    # train / valid
    train_size = train_subset_size - valid_size
    train_ds, valid_ds = torch.utils.data.random_split(full_train, [train_size, valid_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader


def calculate_batch_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    return correct, labels.size(0)


# --------------------------------------------------------
# функция оценки модели
# --------------------------------------------------------
def evaluate_model(model: nn.Module, loader: DataLoader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    samples = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            c, _ = calculate_batch_accuracy(outputs, labels)
            correct += c
            samples += labels.size(0)

    return total_loss / samples, correct / samples


# --------------------------------------------------------
# ОБУЧЕНИЕ
# --------------------------------------------------------
def train_model(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 10,
):

    model.to(DEVICE)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mname = model.get_name()
    log_dir = f"runs/{mname}_{ts}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"Обучение началось. Логи: {log_dir}")

    for epoch in range(1, num_epochs + 1):

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            c, _ = calculate_batch_accuracy(outputs, labels)
            train_correct += c
            train_samples += labels.size(0)

        avg_train_loss = train_loss / train_samples
        train_acc = train_correct / train_samples

        # ---------------- VALID ----------------
        valid_loss, valid_acc = evaluate_model(model, valid_loader, loss_fn)

        # ---------------- TEST ----------------
        test_loss, test_acc = evaluate_model(model, test_loader, loss_fn)

        # --- LOGS (6 графиков) ---
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)

        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/valid", valid_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        print(
            f"Epoch {epoch}: "
            f"train_acc={train_acc:.4f}, valid_acc={valid_acc:.4f}, test_acc={test_acc:.4f}"
        )

    writer.close()
    print("Обучение завершено.")

    torch.save(model.state_dict(), f"{mname}_cifar10.pth")
    print(f"Модель сохранена в {mname}_cifar10.pth")


if __name__ == "__main__":
    model_type = "s"

    train_loader, valid_loader, test_loader = get_data_loaders(
        batch_size=50,
        train_subset_size=10000,
        valid_size=2000,
        data_dir="cifar10",
    )

    model = ConvNet() if model_type == "conv" else LinearNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


    train_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        num_epochs=40,
    )
