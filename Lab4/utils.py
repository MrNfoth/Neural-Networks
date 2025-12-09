import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_from_zenodo(
    url: str, target_dir: Path, filename: str, force: bool = False
) -> Path:
    """Скачивает архив filename в папку target_dir, если его там нет или force=True.

    Args:
        url (str): Прямая ссылка на скачивание с Zenodo.
        target_dir (Path): Папка для сохранения архива.
        filename (str): Имя файла архива (например, 'glioma_mini.zip').
        force (bool): Если True, всегда перекачивает архив.

    Returns:
        Path: Путь к скачанному архиву.

    """
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / filename

    if archive_path.exists() and not force:
        return archive_path

    response = requests.get(url, stream=True, timeout=10)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with (
        archive_path.open("wb") as f,
        tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc=f"Downloading {filename}",
        ) as progress_bar,
    ):
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                written = f.write(chunk)
                progress_bar.update(written)

    return archive_path


def extract_zip(archive_path: Path, extract_dir: Path) -> None:
    """
    Распаковывает zip-архив archive_path в папку extract_dir,
    но пропускает распаковку, если папка уже существует и не пуста.

    Args:
        archive_path (Path): Путь к zip-файлу.
        extract_dir (Path): Папка для распаковки содержимого.
    """
    # Если папка существует и содержит файлы/папки — выходим без распаковки
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return

    # Иначе создаём папку (если её ещё нет) и распаковываем
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as z:
        z.extractall(path=extract_dir)


def ensure_glioma_mini(
    zenodo_url: str = "https://zenodo.org/records/15395623/files/glioma_mini.zip?download=1",
    root_dir: Path = Path("."),
    archive_name: str = "glioma_mini.zip",
    force_download: bool = False,
) -> Path:
    """
    Обеспечивает загрузку и распаковку архива glioma_mini.zip в структуру:
      my_project/
      └── glioma_mini/
          ├── glioma_mini.zip
          └── data/
              ├── train/
              └── valid/

    Args:
        zenodo_url (str): Прямая ссылка Zenodo для скачивания архива.
        root_dir (Path): Корневая папка проекта (по умолчанию текущая).
        archive_name (str): Имя архива (по умолчанию 'glioma_mini.zip').
        force_download (bool): Если True, скачивает архив даже если он уже есть.

    Returns:
        Path: Путь к папке data/ внутри glioma_mini.

    """
    glioma_root = root_dir / "glioma_mini"
    archive_path = download_from_zenodo(
        url=zenodo_url,
        target_dir=glioma_root,
        filename=archive_name,
        force=force_download,
    )

    data_dir = glioma_root / "data"
    extract_zip(archive_path, data_dir)

    # Проверяем наличие и содержимое папки train
    train_dir = data_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Не найдена папка: {train_dir}")
    if not any(train_dir.iterdir()):
        raise FileNotFoundError(f"Папка пуста: {train_dir}")

    # Проверяем наличие и содержимое папки valid
    valid_dir = data_dir / "valid"
    if not valid_dir.is_dir():
        raise FileNotFoundError(f"Не найдена папка: {valid_dir}")
    if not any(valid_dir.iterdir()):
        raise FileNotFoundError(f"Папка пуста: {valid_dir}")

    return data_dir
