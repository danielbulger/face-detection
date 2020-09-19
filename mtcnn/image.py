from typing import List, Tuple

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from .box import crop_bboxes

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


def get_image_tensors(image: Image, scale_factor: float, min_size: int) -> Tuple[List[Tensor], List[float]]:
    image = image.convert('RGB')

    scales = _get_image_scales(scale_factor, min_size, image.width, image.height)

    tensors = []

    for scale in scales:
        tensors.append(to_tensor(_scale_image(image, scale)))

    return tensors, scales


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


def extract_image_box(image: Image, bboxes: np.ndarray, size: int) -> np.ndarray:

    num_boxes = len(bboxes)

    width, height = image.size

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = crop_bboxes(bboxes, width, height)

    img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

    for i in range(num_boxes):
        box = np.zeros((h[i], w[i], 3), 'uint8')

        array = np.asarray(image, 'uint8')

        box[dy[i]: edy[i] + 1, dx[i]: edx[i] + 1, :] = array[y[i]: ey[i] + 1, x[i]: ex[i] + 1, :]

        box = Image.fromarray(box)
        box = box.resize((size, size), Image.BILINEAR)
        box = np.asarray(box, 'float32')

        img_boxes[i, :, :, :] = _ndarray_to_image(box)

    return img_boxes


def _ndarray_to_image(array: np.ndarray):
    # Convert from [h, w, c] to [c, w, h]
    array = array.transpose((2, 0, 1))

    # Convert from [c, w, h] to [1, c, w, h]
    array = np.expand_dims(array, 0)

    # Normalise the pixel data
    array = (array - 127.5) * 0.0078125

    return array
