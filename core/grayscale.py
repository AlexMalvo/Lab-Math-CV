import numpy as np

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Универсальное преобразование изображения в grayscale.
    Поддержка RGB, RGBA, grayscale, grayscale+alpha.
    """
    if img.ndim == 2:
        return img  # уже grayscale

    if img.ndim == 3:
        channels = img.shape[2]
        if channels >= 3:
            R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.299 * R + 0.587 * G + 0.114 * B
            return np.clip(gray, 0, 255).astype(np.uint8)
        elif channels == 1:
            return img[:, :, 0]
        else:
            raise ValueError(f"to_grayscale: неизвестное число каналов: {channels}")

    raise ValueError(f"to_grayscale: неподдерживаемая размерность: shape={img.shape}")
