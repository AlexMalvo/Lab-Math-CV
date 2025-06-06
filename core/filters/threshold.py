import numpy as np
from core.filters.base import BaseFilter
from core.grayscale import to_grayscale


class ThresholdFilter(BaseFilter):
    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = to_grayscale(image)
        binary = np.where(gray >= self.threshold, 255, 0)
        return binary.astype(np.uint8)
