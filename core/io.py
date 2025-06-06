from PIL import Image
import numpy as np
import os

def load_image(path: str) -> np.ndarray:
    """
    Функция загружает изображение из файла и возвращает NumPy-массив (RGB)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(img_array: np.ndarray, path: str) -> None:
    """
    Функция сохраняет NumPy-массив изображения (RGB) в указанный файл
    """
    img = Image.fromarray(img_array)
    img.save(path)
