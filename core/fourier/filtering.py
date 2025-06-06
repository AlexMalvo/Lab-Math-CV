import numpy as np
from core.fourier.enlarge import dct2, idct2
from core.grayscale import to_grayscale


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Масштабирует изображение от 0 до 255."""
    image = image - np.min(image)
    if np.max(image) == 0:
        return np.zeros_like(image, dtype=np.uint8)
    image = image / np.max(image) * 255
    return image.astype(np.uint8)


def low_pass_filter(image: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
    """
    Функция применяет низкочастотную фильтрацию через DCT.
    cutoff_ratio — доля сохраняемых частот (0.0 < r ≤ 1.0).
    """
    if image.ndim == 3:
        return np.stack([
            low_pass_filter(image[:, :, c], cutoff_ratio) for c in range(3)
        ], axis=2)

    gray = image.astype(float)
    h, w = gray.shape
    cutoff_h = int(h * cutoff_ratio)
    cutoff_w = int(w * cutoff_ratio)

    dct_img = dct2(gray)
    mask = np.zeros_like(dct_img)
    mask[:cutoff_h, :cutoff_w] = 1  # сохраняем только низкие частоты

    filtered_dct = dct_img * mask
    result = idct2(filtered_dct)

    return normalize_image(result)


def high_pass_filter(image: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
    """
    Применяет высокочастотную фильтрацию через DCT.
    cutoff_ratio — доля нижних частот, которые обнуляются.
    """
    if image.ndim == 3:
        return np.stack([
            high_pass_filter(image[:, :, c], cutoff_ratio) for c in range(3)
        ], axis=2)

    gray = image.astype(float)
    h, w = gray.shape
    cutoff_h = int(h * cutoff_ratio)
    cutoff_w = int(w * cutoff_ratio)

    dct_img = dct2(gray)
    mask = np.ones_like(dct_img)
    mask[:cutoff_h, :cutoff_w] = 0  # обнуляем низкие частоты

    filtered_dct = dct_img * mask
    result = idct2(filtered_dct)

    return normalize_image(result)
