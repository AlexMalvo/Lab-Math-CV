import numpy as np
from core.filters.base import BaseFilter
from core.convolution import pad_image


class MedianFilter(BaseFilter):
    def __init__(self, kernel_size: int = 3):
        if kernel_size % 2 == 0:
            raise ValueError("Размер ядра должен быть нечётным!")
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def apply(self, image: np.ndarray) -> np.ndarray:
        padded = pad_image(image, self.pad, self.pad, mode='reflect')
        output = np.zeros_like(image)

        if image.ndim == 2:
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    region = padded[y:y + self.kernel_size, x:x + self.kernel_size]
                    output[y, x] = np.median(region)
        else:
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    for c in range(image.shape[2]):
                        region = padded[y:y + self.kernel_size, x:x + self.kernel_size, c]
                        output[y, x, c] = np.median(region)

        return output.astype(np.uint8)
