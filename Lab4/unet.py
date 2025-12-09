import os
import random
from datetime import datetime
from glob import glob
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import JaccardIndex
from torchvision import transforms
from tqdm import tqdm
from utils import ensure_glioma_mini


class ConvBlock(nn.Module):
    """
    ConvBlock реализует два последовательных свёрточных слоя
    с ядром 3×3 и паддингом 1, каждый из которых
    следует за функцией активации ReLU.

    Args:
        in_channels (int): число каналов на входе.
        out_channels (int): число каналов на выходе.

    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            # Первая свёртка: in_channels → out_channels
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
            # Вторая свёртка: out_channels → out_channels
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Метод, вызываемый при прямом проходе модели.

        Args:
            x (Tensor): тензор формы [B, in_channels, H, W]

        Returns:
            Tensor: выход формы [B, out_channels, H, W]

        """
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    EncoderBlock включает ConvBlock и операцию MaxPool2d,
    формируя шаг сжатия для U-Net.

    Args:
        in_channels (int): число входных каналов.
        out_channels (int): число выходных каналов после ConvBlock.

    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Сверточная часть (две свёртки + ReLU)
        self.conv = ConvBlock(in_channels, out_channels)
        # Пулинг для уменьшения пространственных размеров в 2 раза
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Метод, вызываемый при прямом проходе модели.

        Args:
            x (Tensor): входной тензор [B, in_channels, H, W]

        Returns:
            tuple:
                - feature_map (Tensor): выход ConvBlock [B, out_channels, H, W]
                - pooled (Tensor): результат MaxPool2d [B, out_channels, H/2, W/2]

        """
        feature_map = self.conv(x)
        pooled = self.pool(feature_map)
        return feature_map, pooled


class DecoderBlock(nn.Module):
    """
    DecoderBlock выполняет upsampling с помощью ConvTranspose2d,
    затем конкатенирует с feature-map из энкодера и прогоняет через ConvBlock.

    Args:
        in_channels (int): число каналов на входе (обычно из предыдущего декодера).
        out_channels (int): число каналов после свёрток.

    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Upsampling: увеличиваем H, W в 2 раза и уменьшаем каналы до out_channels
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        # После конкатенации будет in_channels = out_channels(from up) + out_channels(from skip)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Метод, вызываемый при прямом проходе модели.

        Args:
            x (Tensor): входной тензор из предыдущего шага декодера [B, in_channels, H, W]
            skip (Tensor): feature-map из соответствующего EncoderBlock [B, out_channels, H*2, W*2]

        Returns:
            Tensor: объединённый и свернутый выход [B, out_channels, H*2, W*2]

        """
        x = self.up(x)  # upsample
        x = torch.cat([x, skip], dim=1)  # конкатенация по канальному измерению
        return self.conv(x)  # две свёртки + ReLU


class UNet(nn.Module):
    """
    U-Net для сегментации изображений.
    Состоит из «сжимающей» (encoder) и «расширяющей» (decoder) частей,
    соединённых скип-коннекшенами.

    Args:
        in_channels (int): число каналов входного изображения (например, 3 для RGB).
        out_channels (int): число каналов выходной карты сегментации (классы).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Encoder (сжатие)
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck (глубокая часть)
        # Здесь увеличиваем представление перед декодером (традиционный UNet)
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder (расширение) — порядок от глубокого к мелкому
        # Замечание: параметры подобраны так, чтобы ConvTranspose и конкатенация
        # корректно согласовывались с Encoder блоками выше.
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        # Финальный 1x1 сверточный слой переводит в нужное число выходных каналов
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Метод, реализующий прямой проход модели.

        Args:
            x (Tensor): входной тензор [B, in_channels, H, W]

        Returns:
            Tensor: выходная карта сегментации [B, out_channels, H, W]
        """
        # Encoder: получаем feature maps (для skip) и pooled (для следующего уровня)
        f1, p1 = self.enc1(x)  # f1: 64 ch
        f2, p2 = self.enc2(p1)  # f2: 128 ch
        f3, p3 = self.enc3(p2)  # f3: 256 ch
        f4, p4 = self.enc4(p3)  # f4: 512 ch

        # Bottleneck
        b = self.bottleneck(p4)  # 1024 ch

        # Decoder: подаём bottleneck + skip-коннекты в обратном порядке
        d4 = self.dec4(b, f4)  # -> 512 ch
        d3 = self.dec3(d4, f3)  # -> 256 ch
        d2 = self.dec2(d3, f2)  # -> 128 ch
        d1 = self.dec1(d2, f1)  # -> 64 ch

        out = self.final_conv(d1)  # -> out_channels (логиты)
        return out


class CustomSegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Callable | None = None,
        preprocess: Callable | None = None,
    ) -> None:
        """Кастомный датасет для загрузки изображений и масок.

        Args:
            images_dir (str): Путь к папке с изображениями.
            masks_dir (str): Путь к папке с масками.
            transform (callable, optional): Трансформации для применения к изображениям и маскам.
            preprocess (callable, optional): Преобразования для применения только к изображениям.

        """
        self.image_paths = sorted(glob(images_dir))
        self.mask_paths = sorted(glob(masks_dir))
        self.transform = transform
        self.preprocess = preprocess

        # Список всех файлов изображений
        self.images: list[Image.Image] = []
        self.masks: list[Image.Image] = []

        self.__load_data_to_memory()

    def __load_data_to_memory(self):
        for image_path, mask_path in zip(
            self.image_paths, self.mask_paths, strict=True
        ):
            # Открываем изображение и преобразуем в RGB
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            with Image.open(mask_path) as msk:
                mask = msk.convert("L")  # Маска в оттенках серого

            # Применяем трансформации для изображения, если они заданы
            if self.preprocess is not None:
                image = self.preprocess(image)

            # Применяем общие трансформации, если они заданы
            if self.transform is not None:
                # Генерация seed для синхронизации случайных трансформаций
                seed = np.random.randint(
                    2147483647
                )  # Генерируем seed для синхронизации
                self.__set_seed(seed=seed)

                # Применение преобразований к изображению
                image = self.transform(image)

                # Установка seed снова для маски
                self.__set_seed(seed=seed)

                # Применение преобразований к маске
                mask = self.transform(mask)

            self.images.append(image)
            self.masks.append(mask)

    def __set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Image.Image, Image.Image]:
        return self.images[index], self.masks[index]


def get_train_transform() -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((512, 704)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    )


def get_valid_transform() -> Callable:
    return transforms.Compose(
        [transforms.Resize((512, 704)), transforms.ToTensor()]
    )


def get_preprocess() -> Callable:
    return transforms.Lambda(lambda x: TF.adjust_contrast(x, 2.5))


def __check_directory_exists(path: str) -> None:
    assert os.path.exists(f"{path}"), f"Не найдена директория [{path}]."


def get_train_dataset(
    transform: Callable, preprocess: Callable
) -> CustomSegmentationDataset:
    folder = "glioma_mini/data/train"
    __check_directory_exists(folder)
    return CustomSegmentationDataset(
        images_dir=f"{folder}/images/*.tif",
        masks_dir=f"{folder}/masks/*.tif",
        transform=transform,
        preprocess=preprocess,
    )


def get_valid_dataset(
    transform: Callable, preprocess: Callable
) -> CustomSegmentationDataset:
    folder = "glioma_mini/data/valid"
    __check_directory_exists(folder)
    return CustomSegmentationDataset(
        images_dir=f"{folder}/images/*.tif",
        masks_dir=f"{folder}/masks/*.tif",
        transform=transform,
        preprocess=preprocess,
    )


def create_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 5
) -> tuple[DataLoader, DataLoader]:
    """
    Создает загрузчики данных для обучающего, валидационного и тестового наборов.

    Args:
        train_dataset (Dataset): Набор данных для обучения.
        val_dataset (Dataset): Набор данных для валидации.
        batch_size (int): Размер батча.

    Returns:
        tuple(DataLoader, DataLoader): Загрузчики для обучающего и валидационного наборов.
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def run_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    phase: Literal["train", "valid", "test"],
    device: torch.device,
    current_epoch: int = 0,
) -> dict[str, float]:
    """
    Оценивает модель на заданном наборе данных (train, valid, test).

    Args:
        model (torch.nn.Module): Модель для оценки.
        criterion (torch.nn.Module): Функция потерь.
        optimizer (torch.optim.Optimizer): Оптимизатор.
        data_loader (DataLoader): Загрузчик данных для оценки.
        phase (str): Тип выборки: 'train', 'valid' или 'test'.
        device (torch.device): Устройство для вычислений.
        current_epoch (int): Номер текущей эпохи для отображения.

    Returns:
        dict(str, float): Словарь с ключами 'loss' и 'iou'.
    """
    # Устанавливаем режим работы модели
    if phase in ("valid", "test"):
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_iou = 0.0
    jaccard = JaccardIndex(task="binary", threshold=0.5).to(device)

    # Только для валидации и теста используем no_grad
    context = (
        torch.no_grad() if phase in ("valid", "test") else torch.enable_grad()
    )
    with context:
        for images, masks in tqdm(
            data_loader, desc=f"Epoch {current_epoch}", leave=False
        ):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks.float())
            total_loss += loss.item()

            if phase == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Расчет IoU (Jaccard Index)
            total_iou += jaccard(outputs, masks).item()

    # Средние значения
    avg_loss = total_loss / len(data_loader)
    avg_iou = total_iou / len(data_loader)

    return {"loss": avg_loss, "iou": avg_iou}


def train_and_validate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    save_path: str = "best_model.pth",
) -> None:
    """
    Выполняет обучение и валидацию модели на нескольких эпохах.

    Args:
        model (torch.nn.Module): Модель для обучения.
        criterion (torch.nn.Module): Функция потерь.
        optimizer (Optimizer): Оптимизатор.
        train_loader (DataLoader): Загрузчик данных для обучения.
        val_loader (DataLoader): Загрузчик данных для валидации.
        device (torch.device): Устройство для выполнения вычислений.
        num_epochs (int): Количество эпох.
        save_path (str): Путь для сохранения лучших весов модели. По умолчанию "best_model.pth".
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/ep{num_epochs}_{ts}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_val_loss = float("inf")

    # Основной цикл обучения с tqdm
    for epoch in range(1, num_epochs + 1):
        train_result = run_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=train_loader,
            phase="train",
            device=device,
            current_epoch=epoch,
        )
        valid_result = run_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=valid_loader,
            phase="valid",
            device=device,
            current_epoch=epoch,
        )

        # Сохранение лучших весов модели
        if valid_result["loss"] < best_val_loss:
            best_val_loss = valid_result["loss"]
            torch.save(model.state_dict(), save_path)

        # Логгирование потерь для текущей эпохи
        writer.add_scalar("Loss/train", train_result["loss"], epoch)
        writer.add_scalar("Loss/valid", valid_result["loss"], epoch)
        writer.add_scalar("IoU/train", train_result["iou"], epoch)
        writer.add_scalar("IoU/valid", valid_result["iou"], epoch)
        print(
            f"Epoch {epoch:02d}/{num_epochs} - Train Loss: {train_result['loss']:.4f}, "
            f"Valid Loss: {valid_result['loss']:.4f}, Train IoU: {train_result['iou']:.4f}, "
            f"Valid IoU: {valid_result['iou']:.4f}"
        )

    print("Обучение завершено.")


def __collect_examples(
    data_loader: DataLoader, num_examples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Собирает необходимое количество примеров из DataLoader, запрашивая дополнительные батчи, если необходимо.

    Args:
        data_loader (DataLoader): Загрузчик данных.
        num_examples (int): Количество примеров для сбора.

    Returns:
        tuple(torch.Tensor, torch.Tensor): Тензоры изображений и масок, содержащие num_examples примеров.
    """
    images_list, masks_list = [], []
    data_iter = iter(data_loader)

    # Наполнение списка изображениями и масками
    while len(images_list) * data_loader.batch_size < num_examples:
        try:
            images, masks = next(data_iter)
        except StopIteration:
            # Если примеры закончились, перезапускаем итератор
            data_iter = iter(data_loader)
            images, masks = next(data_iter)

        images_list.append(images)
        masks_list.append(masks)

    # Объединяем батчи в один тензор и обрезаем до num_examples
    images = torch.cat(images_list, dim=0)[:num_examples]
    masks = torch.cat(masks_list, dim=0)[:num_examples]
    return images, masks


def show_examples(data_loader: DataLoader, num_examples: int = 3):
    """
    Функция для отображения нескольких примеров из переданной выборки.

    Args:
        data_loader (DataLoader): Загрузчик данных для переданной выборки.
        num_examples (int): Количество примеров для отображения.
    """
    # Определяем общее количество доступных примеров
    dataset_size = len(data_loader.dataset)

    # Если запрашиваемое число примеров превышает доступное, выводим предупреждение
    if num_examples > dataset_size:
        print(
            f"Запрошено {num_examples} примеров, но в наборе данных доступно только {dataset_size}."
        )
        print(
            f"Показываю максимально доступное количество примеров: {dataset_size}."
        )
        num_examples = dataset_size

    # Получаем изображения и маски с помощью collect_examples
    images, masks = __collect_examples(data_loader, num_examples)

    fig, axes = plt.subplots(num_examples, 2, figsize=(10, num_examples * 5))
    fig.suptitle("Примеры", fontsize=16)

    axes[0, 0].set_title("Изображение")
    axes[0, 1].set_title("Маска")
    for i in range(num_examples):
        img = (
            images[i].permute(1, 2, 0).cpu().numpy()
        )  # Переставляем оси для matplotlib
        mask = masks[i].squeeze().cpu().numpy()  # Извлекаем маску

        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].axis("off")

    plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1)
    plt.subplots_adjust(top=0.95, hspace=0.05, wspace=0.02)
    plt.show()


def __collect_segmentation_examples(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_examples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Собирает необходимое количество примеров для сегментации, включая предсказанные маски.

    Args:
        model (torch.nn.Module): Обученная модель для сегментации.
        data_loader (DataLoader): Загрузчик данных.
        device (torch.device): Устройство для выполнения вычислений.
        num_examples (int): Количество примеров для сбора.

    Returns:
        tuple(torch.Tensor, torch.Tensor, torch.Tensor): Тензоры изображений, истинных и предсказанных масок.
    """
    images_list, masks_list, outputs_list = [], [], []
    model.eval()
    data_iter = iter(data_loader)

    with torch.no_grad():
        while len(images_list) * data_loader.batch_size < num_examples:
            try:
                # Получаем следующий батч изображений и масок
                images, masks = next(data_iter)
            except StopIteration:
                # Перезапуск итератора, если примеры закончились
                data_iter = iter(data_loader)
                images, masks = next(data_iter)

            # Переносим данные на устройство и делаем предсказания
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Сохраняем данные в списки
            images_list.append(images)
            masks_list.append(masks)
            outputs_list.append(outputs)

    # Объединяем батчи и обрезаем до нужного количества примеров
    images = torch.cat(images_list, dim=0)[:num_examples]
    masks = torch.cat(masks_list, dim=0)[:num_examples]
    outputs = torch.cat(outputs_list, dim=0)[:num_examples]

    return images, masks, outputs


def show_segmentation_results(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_examples: int = 3,
):
    """
    Отображает результаты сегментации тестовых данных, включая исходное
        изображение, предсказанную маску и целевую маску.

    Args:
        model (torch.nn.Module): Обученная модель для сегментации.
        data_loader (DataLoader): Загрузчик данных для тестовой выборки.
        device (torch.device): Устройство для выполнения вычислений.
        num_examples (int): Количество примеров для отображения.
    """
    # Определяем общее количество доступных данных в наборе
    dataset_size = len(data_loader.dataset)

    # Если запрашиваемое число примеров превышает доступное, выводим предупреждение
    if num_examples > dataset_size:
        print(
            f"Запрошено {num_examples} примеров, но в наборе данных доступно только {dataset_size}."
        )
        print(
            f"Показываю максимально доступное количество примеров: {dataset_size}."
        )
        num_examples = dataset_size

    # Получаем изображения, истинные маски и предсказанные маски
    images, masks, outputs = __collect_segmentation_examples(
        model, data_loader, device, num_examples
    )

    fig, axes = plt.subplots(num_examples, 3, figsize=(15, num_examples * 5))
    fig.suptitle("Результаты сегментации", fontsize=16, y=0.98)

    axes[0, 0].set_title("Изображение")
    axes[0, 1].set_title("Истинная маска")
    axes[0, 2].set_title("Предсказанная маска")
    for i in range(num_examples):
        img = (
            images[i].permute(1, 2, 0).cpu().numpy()
        )  # Перестановка осей для корректного отображения
        true_mask = masks[i].squeeze().cpu().numpy()  # Истинная маска
        pred_mask = (
            torch.sigmoid(outputs[i]).squeeze().cpu().numpy()
        )  # Предсказанная маска
        pred_mask = (pred_mask > 0.5).astype(np.float32)

        # Отображаем исходное изображение
        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")

        # Отображаем истинную маску
        axes[i, 1].imshow(true_mask, cmap="gray")
        axes[i, 1].axis("off")

        # Отображаем предсказанную маску
        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].axis("off")

    plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1)
    plt.subplots_adjust(top=0.90, hspace=0.05, wspace=0.02)
    plt.show()


def main(batch_size: int, num_epochs: int, model_save_path: str):
    """
    Args:
        batch_size (int): Размер батча для всех выборок.
        num_epochs (int): Количество эпох обучения.
        model_save_path (str): Путь для сохранения лучших весов модели.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение будет проходить на [{device}].")

    model = UNet(in_channels=3, out_channels=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Получение методов трансформаций для каждой выборки
    train_transform = get_train_transform()
    valid_test_transform = get_valid_transform()
    preprocess = get_preprocess()

    # Создание наборов данных
    train_dataset = get_train_dataset(
        transform=train_transform, preprocess=preprocess
    )
    valid_dataset = get_valid_dataset(
        transform=valid_test_transform, preprocess=preprocess
    )

    # Создание загрузчиков данных
    train_loader, valid_loader = create_dataloaders(
        train_dataset, valid_dataset, batch_size
    )

    # Отображаем примеры
    show_examples(train_loader, num_examples=2)

    # Обучение и валидация
    train_and_validate(
        model,
        criterion,
        optimizer,
        train_loader,
        valid_loader,
        device,
        num_epochs,
        model_save_path,
    )

    # Тестирование с лучшими весами
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    test_result = run_one_epoch(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        data_loader=valid_loader,
        phase="test",
        device=device,
    )
    print(
        f"\nTest Loss: {test_result['loss']:.4f}, Average Jaccard Index (IoU): {test_result['iou']:.4f}"
    )

    # Визуализации результатов на тестовых данных
    show_segmentation_results(model, valid_loader, device, num_examples=2)


if __name__ == "__main__":
    ensure_glioma_mini()
    os.makedirs("weights", exist_ok=True)
    main(batch_size=1, num_epochs=90, model_save_path="weights/best_model.pth")
