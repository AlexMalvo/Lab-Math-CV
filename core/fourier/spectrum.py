import numpy as np
import matplotlib.pyplot as plt
from core.fourier.enlarge import dct2
from core.grayscale import to_grayscale


def show_dct_spectrum(image: np.ndarray, log_scale: bool = True, title: str = "Спектр DCT") -> None:
    """
    Функция строит и отображает DCT-спектр изображения в градациях серого
    """
    gray = to_grayscale(image)
    dct_img = dct2(gray.astype(float))

    # Чтобы увидеть детали, берём логарифм амплитуды
    if log_scale:
        spectrum = np.log(np.abs(dct_img) + 1)
    else:
        spectrum = np.abs(dct_img)

    plt.figure(figsize=(8, 6))
    plt.imshow(spectrum, cmap='hot', extent=(0, spectrum.shape[1], spectrum.shape[0], 0))
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
