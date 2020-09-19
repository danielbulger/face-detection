import numpy as np
import torch
from PIL import Image

import mtcnn.image
from mtcnn import RNet, ONet, PNet
from mtcnn import box

pnet, rnet, onet = PNet(), RNet(), ONet()

image = Image.open('data/image.jpg')

tensors, scales = mtcnn.image.get_image_tensors(image, 0.709, 20)


def stage1(threshold: float):
    bounding_boxes = []

    for tensor, scale in zip(tensors, scales):
        output = pnet(tensor)

        classification = output['classification'].squeeze().data.numpy()[1, :, :]

        regression = output['regression'].squeeze().data.numpy()

        bboxes = box.generate_bbox(classification, regression, scale, threshold)

        if bboxes.size == 0:
            continue

        pick = box.nms(bboxes[:, :5], 0.5)
        bounding_boxes.append(bboxes[pick])

    if len(bounding_boxes) > 0:
        return box.process_bboxes(np.vstack(bounding_boxes))

    return []


def stage2(image: Image, bboxes: np.ndarray, threshold: float):
    image_boxes = mtcnn.image.extract_image_box(image, bboxes, 24)

    image_boxes = torch.from_numpy(image_boxes)

    output = rnet(image_boxes)

    classification = output['classification'].squeeze().data.numpy()

    regression = output['regression'].squeeze().data.numpy()

    pick = np.where(classification[:, 1] > threshold)[0]
    # Filter the bounding boxes by the new classification
    bboxes = bboxes[pick]
    # Update the existing probabilities with the latest from the refine network
    bboxes[:, 4] = classification[pick, 1].reshape((-1,))
    # Filter out any predicted regressions that did not meet the threshold
    regression = regression[pick]

    # Apply NMS to the remaining bounding boxes
    pick = box.nms(bboxes, 0.5)
    # Translate & adjust the bounding box coordinates
    bboxes = box.translate_bbox(bboxes[pick], regression[pick])
    # Convert the bounding boxes to a square
    bboxes = box.convert_to_square(bboxes)
    # Round the bounding boxes to an integer.
    bboxes[:, :4] = np.round(bboxes[:, :4])

    return bboxes


def stage3(image: Image, bboxes: np.ndarray, threshold: float):
    image_boxes = mtcnn.image.extract_image_box(image, bboxes, 48)

    image_boxes = torch.from_numpy(image_boxes)

    output = onet(image_boxes)

    classification = output['classification'].squeeze().data.numpy()
    regression = output['regression'].squeeze().data.numpy()
    landmarks = output['regression'].squeeze().data.numpy()

    pick = np.where(classification[:, 1] > threshold)
    bboxes = bboxes[pick]
    # Update the probabilities with the latest from the ONet
    bboxes[:, 4] = classification[pick, 1].reshape((-1,))
    # Filter the regression by the ones that meet the threshold
    regression = regression[pick]
    # Filter the landmarks by the ones that meet the threshold
    landmarks = landmarks[pick]

    # Now we calculate the landmark positions
    x, y = bboxes[:, 0], bboxes[:, 1]
    width = bboxes[:, 2] - x + 1
    height = bboxes[:, 3] - y + 1

    landmarks[:, :5] = np.expand_dims(x, 1) + np.expand_dims(width, 1) * landmarks[:, :5]
    landmarks[:, 5:10] = np.expand_dims(y, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

    bboxes = box.translate_bbox(bboxes, regression)
    pick = box.nms(bboxes, 0.5)

    bboxes = bboxes[pick]

    # Convert the bounding boxes to a square
    bboxes = box.convert_to_square(bboxes)
    # Round the bounding boxes to an integer.
    bboxes[:, :4] = np.round(bboxes[:, :4])

    return {
        'bboxes': bboxes,
        'landmarks': landmarks[pick]
    }


bboxes = stage1(0.1)

bboxes = stage2(image, bboxes, 0.1)

print(stage3(image, bboxes, 0.1))
