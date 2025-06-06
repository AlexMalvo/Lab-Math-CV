import numpy as np
import matplotlib.pyplot as plt

def split_channels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if img.ndim != 3 or img.shape[2] != 3:
        print(f"[split_channels] FAIL: shape = {img.shape}")
        raise ValueError(f"Ожидалось 3-канальное RGB-изображение, получено: shape={img.shape}")
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B

def merge_channels(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.stack((R, G, B), axis=2).astype(np.uint8)

def show_rgb_combinations(image: np.ndarray):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Ожидается 3-канальное изображение для поканального анализа. Получено: {image.shape}")

    R, G, B = split_channels(image)

    RG = np.stack((R, G, np.zeros_like(R)), axis=2)
    GB = np.stack((np.zeros_like(G), G, B), axis=2)
    RB = np.stack((R, np.zeros_like(R), B), axis=2)

    titles = ['R', 'G', 'B', 'RG', 'GB', 'RB']
    images = [np.stack([R, np.zeros_like(R), np.zeros_like(R)], axis=2),
              np.stack([np.zeros_like(G), G, np.zeros_like(G)], axis=2),
              np.stack([np.zeros_like(B), np.zeros_like(B), B], axis=2),
              RG, GB, RB]

    plt.figure(figsize=(12, 6))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img.astype(np.uint8))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
