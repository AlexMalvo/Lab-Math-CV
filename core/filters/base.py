from abc import ABC, abstractmethod
import numpy as np


class BaseFilter(ABC):
    """
    Абстрактный базовый класс фильтра.
    Все фильтры должны реализовать метод apply().
    """

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        pass
