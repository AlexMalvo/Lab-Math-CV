import numpy as np
from core.filters.base import BaseFilter
from core.convolution import convolve
from core.grayscale import to_grayscale

class SobelFilter(BaseFilter):
    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image.astype(float)
        else:
            gray = to_grayscale(image).astype(float)

        kernel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=float)

        kernel_y = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=float)


        gx = convolve(gray, kernel_x, padding='reflect')
        gy = convolve(gray, kernel_y, padding='reflect')

        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # Защита от деления на ноль
        max_val = magnitude.max()
        if max_val == 0:
            return np.zeros_like(magnitude, dtype=np.uint8)

        normalized = (magnitude / max_val) * 255
        return normalized.astype(np.uint8)
