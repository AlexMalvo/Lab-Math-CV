import numpy as np

def pad_image(image: np.ndarray, pad_y: int, pad_x: int, mode: str = 'reflect') -> np.ndarray:
    """
    Функция добавляет отступы по краям изображения.
    Поддерживает:
    - 'reflect' (зеркальное отражение)
    - 'constant' (нули)
    - 'edge' (повторение границ)
    """
    if image.ndim == 2:
        return np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode=mode)
    elif image.ndim == 3:
        return np.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode=mode)
    else:
        raise ValueError("Ожидалось 2D или 3D изображение")


def convolve(image: np.ndarray, kernel: np.ndarray, padding: str = 'reflect') -> np.ndarray:
    """
    Функция выполняет свёртку изображения с заданным ядром вручную.
    Поддерживает как чёрно-белые, так и цветные изображения.
    """
    if kernel.ndim != 2:
        raise ValueError("Ядро должно быть двумерным")

    kernel = np.flipud(np.fliplr(kernel))  # переворачиваем ядро по формуле свёртки
    kH, kW = kernel.shape
    pad_y, pad_x = kH // 2, kW // 2

    padded = pad_image(image, pad_y, pad_x, mode=padding)
    output = np.zeros_like(image)

    # Обработка серого изображения
    if image.ndim == 2:
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                region = padded[y:y + kH, x:x + kW]
                output[y, x] = np.sum(region * kernel)
    # Обработка цветного изображения
    elif image.ndim == 3:
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    region = padded[y:y + kH, x:x + kW, c]
                    output[y, x, c] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)
