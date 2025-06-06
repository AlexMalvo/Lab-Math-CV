import numpy as np
import matplotlib.pyplot as plt
from core.channels import split_channels
from core.grayscale import to_grayscale

def plot_color_intensity(img: np.ndarray, row_index: int) -> None:
    """
    Функция строит графики значений каналов R, G, B по выбранной точке
    """
    R, G, B = split_channels(img)
    x = np.arange(img.shape[1]) # ось x - индексы слотбцов

    plt.figure(figsize=(12, 6))
    plt.plot(x, R[row_index], "r-" ,label="R")
    plt.plot(x, G[row_index], "g-", label="G")
    plt.plot(x, B[row_index], "b-", label="B")
    plt.title(f"Интенсивность RGB по строке {row_index}")
    plt.xlabel("Положение по ширине")
    plt.ylabel("Интенсивность (от 0 до 255")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_grayscale_intensity(img: np.ndarray, row_index: int) -> None:
    """
    Функция строит график значений серого изображения по выбранной строке.
    """
    gray = to_grayscale(img)
    x = np.arange(gray.shape[1])
    plt.figure(figsize=(8, 4))
    plt.plot(x, gray[row_index], 'k-', label='Grayscale')
    plt.title(f"Интенсивность серого по строке {row_index}")
    plt.xlabel("Положение по ширине")
    plt.ylabel("Интенсивность (от 0 до 255)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
