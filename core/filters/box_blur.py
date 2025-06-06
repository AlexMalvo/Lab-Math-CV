import numpy as np
from core.filters.base import BaseFilter
from core.convolution import convolve


class BoxBlurFilter(BaseFilter):
    def __init__(self, kernel_size: int = 3):
        if kernel_size % 2 == 0:
            raise ValueError("Размер ядра должен быть нечётным!")
        self.kernel_size = kernel_size
        self.kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size ** 2)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return convolve(image, self.kernel, padding='reflect')
