import numpy as np
from core.io import load_image, save_image
from core.channels import split_channels, merge_channels

class ImageProcessor:
    def __init__(self, image_path: str):
        self.image = load_image(image_path)
        self.path = image_path

    def save(self, path: str) -> None:
        save_image(self.image, path)

    def get_shape(self) -> tuple[int, int, int]:
        return self.image.shape

    def get_channels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return split_channels(self.image)

    def set_image(self, new_image: np.ndarray) -> None:
        self.image = new_image

    def get_image(self) -> np.ndarray:
        return self.image
