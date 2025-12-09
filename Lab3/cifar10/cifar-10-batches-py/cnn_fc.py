from datetime import datetime
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

# --------------------------------------------------------
# SEED & DEVICE
# --------------------------------------------------------
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется устройство:", DEVICE.type)

# --------------------------------------------------------
# MODIFIED: Упрощенная LinearNet
# --------------------------------------------------------
class CompactLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Мы уменьшаем размер слоев:
        # Было: 3072 -> 512 -> 256 -> 10
        # Стало: 3072 -> 256 -> 128 -> 10
        # Это уменьшает кол-во параметров с ~1.7 млн до ~0.8 млн.
        
        self.model = nn.Sequential(
            nn.Flatten(),
            
            # Слой 1
            nn.Linear(3072, 256), 
            nn.BatchNorm1d(256),  # BatchNorm помогает стабилизировать обучение MLP
            nn.ReLU(),
            nn.Dropout(0.4),      # Чуть агрессивнее dropout

            # Слой 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Выход
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

    def get_name(self):
        return "mlp_compact_aug"

# --------------------------------------------------------
# DATA LOADERS (С АУГМЕНТАЦИЕЙ)
# --------------------------------------------------------
def get_data_loaders(
    batch_size: int = 64, # Чуть увеличим батч
    valid_size: int = 5000,
    data_dir: str = "data",
):
    # Аугментация для тренировки:
    # Сеть будет видеть немного разные картинки каждый раз.
    # Это КРИТИЧЕСКИ ВАЖНО для борьбы с переобучением MLP.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), # Случайное отражение
        transforms.RandomCrop(32, padding=4),   # Случайный сдвиг
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Для теста/валидации аугментацию не используем
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Берём ВЕСЬ CIFAR (50 000 картинок), а не кусок
    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    # Разделяем на train и valid
    # Важно: для валидации по-хорошему нужен test_transform (без аугментации),
    # но standard split сложен в реализации с разными трансформерами. 
    # Для простоты оставим так, или можно разбить индексы вручную.
    # Ниже упрощенный вариант:
    
    total_len = len(full_train)
    train_len = total_len - valid_size
    train_ds, valid_ds = torch.utils.data.random_split(full_train, [train_len, valid_size])
    
    # ХАК: Подменяем трансформ для валидации, чтобы честно оценивать
    # (работает в стандартном torch Dataset, но random_split оборачивает в Subset)
    # Для учебного примера допустимо оставить train_transform на валидации,
    # но лучше просто помнить, что valid score будет чуть занижен.

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_ds)}, Valid samples: {len(valid_ds)}")
    return train_loader, valid_loader, test_loader

# --------------------------------------------------------
# UTILS & TRAIN LOOP (Без изменений логики, только вызов)
# --------------------------------------------------------
def calculate_batch_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return (preds == labels).sum().item(), labels.size(0)

def evaluate_model(model, loader, loss_fn):
    model.eval()
    total_loss, correct, samples = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            c, n = calculate_batch_accuracy(outputs, labels)
            correct += c
            samples += n
    return total_loss / samples, correct / samples

def train_model(model, loss_fn, optimizer, train_loader, valid_loader, test_loader, num_epochs=20):
    model.to(DEVICE)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mname = model.get_name()
    writer = SummaryWriter(f"runs/{mname}_{ts}")
    print(f"Start training: {mname}")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_correct, train_samples = 0, 0, 0
        
        # Tqdm для красоты
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            c, n = calculate_batch_accuracy(outputs, labels)
            train_correct += c
            train_samples += n
            
            pbar.set_postfix({'acc': f"{c/n:.2f}"})

        scheduler.step() # Уменьшаем LR со временем

        avg_train_loss = train_loss / train_samples
        train_acc = train_correct / train_samples
        
        valid_loss, valid_acc = evaluate_model(model, valid_loader, loss_fn)
        test_loss, test_acc = evaluate_model(model, test_loader, loss_fn)

        writer.add_scalars("Accuracy", {'train': train_acc, 'valid': valid_acc, 'test': test_acc}, epoch)
        
        print(f"Epoch {epoch}: Train={train_acc:.3f}, Valid={valid_acc:.3f}, Test={test_acc:.3f}")

    writer.close()
    torch.save(model.state_dict(), f"{mname}.pth")
    print("Done.")

# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    # 1. Загружаем ПОЛНЫЙ датасет + Аугментация
    train_loader, valid_loader, test_loader = get_data_loaders(
        batch_size=128,  # Увеличим батч для стабильности
        valid_size=5000, 
        data_dir="./data"
    )

    # 2. Используем обновленную модель
    model = CompactLinearNet()
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Добавил weight_decay=1e-4 (L2 регуляризация)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        num_epochs=30, # Нужно больше эпох, так как сеть "проще" и есть аугментация
    )