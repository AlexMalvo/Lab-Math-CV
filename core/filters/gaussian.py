import numpy as np
from core.filters.base import BaseFilter
from core.convolution import convolve


def generate_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Функция генерирует двумерное Гауссово ядро заданного размера и σ.
    """
    if size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным!")
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


class GaussianFilter(BaseFilter):
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        self.kernel = generate_gaussian_kernel(kernel_size, sigma)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return convolve(image, self.kernel, padding='reflect')
