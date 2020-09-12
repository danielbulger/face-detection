from typing import List

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch import Tensor

_num_scales = 12


def _get_image_scales(scale_factor: float, min_size: int, width: int, height: int) -> List[float]:
    smallest_scale = _num_scales / min_size
    min_layer = np.min([width, height]) * smallest_scale

    scales = []

    for factor in range(0, _num_scales):
        if min_layer < _num_scales:
            break
        scales.append(smallest_scale * np.power(scale_factor, factor))
        min_layer *= scale_factor

    return scales


def _scale_image(image: Image, scale: float) -> Image:
    width, height = image.size

    scaled_width = int(np.ceil(width * scale))
    scaled_height = int(np.ceil(height * scale))

    scaled_image = image.resize((scaled_width, scaled_height))

    return scaled_image


def get_image_tensors(file: str, scale_factor: float, min_size: int) -> List[Tensor]:
    test = Image.open(file).convert('RGB')

    scales = _get_image_scales(scale_factor, min_size, test.width, test.height)

    tensors = []

    for scale in scales:
        tensors.append(to_tensor(_scale_image(test, scale)))

    return tensors


def to_tensor(image: Image) -> Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(image)

    # No batches for now
    return tensor.unsqueeze(0)
