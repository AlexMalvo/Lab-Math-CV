import numpy as np
from scipy.fftpack import dct, idct


def dct2(image: np.ndarray) -> np.ndarray:
    """
    Применяет 2D-дискретное косинусное преобразование ко всему изображению.
    """
    return dct(dct(image.T, norm='ortho').T, norm='ortho')


def idct2(coefficients: np.ndarray) -> np.ndarray:
    """
    Применяет обратное 2D-дискретное косинусное преобразование.
    """
    return idct(idct(coefficients.T, norm='ortho').T, norm='ortho')


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Масштабирует изображение от 0 до 255.
    """
    image = image - np.min(image)
    image = image / np.max(image) * 255
    return image.astype(np.uint8)


def upscale_image_fourier(image: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Увеличивает изображение с помощью DCT + обратной DCT.
    Расширяет спектр нулями, чтобы получить более высокое разрешение.
    """
    if image.ndim == 3:
        return np.stack([
            upscale_image_fourier(image[:, :, c], scale) for c in range(3)
        ], axis=2)

    h, w = image.shape
    H, W = h * scale, w * scale

    # Преобразование DCT
    coeff = dct2(image.astype(float))

    # Расширим DCT-матрицу нулями
    extended = np.zeros((H, W))
    extended[:h, :w] = coeff

    # Обратное преобразование на новой сетке
    enlarged = idct2(extended)
    return normalize_image(enlarged)
