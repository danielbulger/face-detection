from typing import List

import torch
import numpy as np
from PIL import Image

import mtcnn.image
import mtcnn.box

from .network import PNet, RNet, ONet


class MTCNN:

    def __init__(self, scale_factor: float, min_size: int, thresholds: List[float]):
        self.scale_factor = scale_factor
        self.min_size = min_size
        self.thresholds = thresholds
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

    def forward(self, image: Image):
        bboxes = self.stage1(image)
        if len(bboxes) == 0:
            return None
        bboxes = self.stage2(image, bboxes)
        if len(bboxes) == 0:
            return None
        return self.stage3(image, bboxes)

    def stage1(self, image: Image):
        tensors, scales = mtcnn.image.get_image_tensors(image, 0.709, 20)

        bounding_boxes = []

        for tensor, scale in zip(tensors, scales):
            output = self.pnet(tensor)

            classification = output['classification'].squeeze().data.numpy()[1, :, :]

            regression = output['regression'].squeeze().data.numpy()

            bboxes = mtcnn.box.generate_bbox(classification, regression, scale, self.thresholds[0])

            if bboxes.size == 0:
                continue

            pick = mtcnn.box.nms(bboxes[:, :5], 0.5)
            bounding_boxes.append(bboxes[pick])

        if len(bounding_boxes) > 0:
            return mtcnn.box.process_bboxes(np.vstack(bounding_boxes))

        return []

    def stage2(self, img: Image, bboxes: np.ndarray):
        image_boxes = mtcnn.image.extract_image_box(img, bboxes, 24)

        image_boxes = torch.from_numpy(image_boxes)

        output = self.rnet(image_boxes)

        classification = output['classification'].squeeze().data.numpy()

        regression = output['regression'].squeeze().data.numpy()

        pick = np.where(classification[:, 1] > self.thresholds[1])[0]
        # Filter the bounding boxes by the new classification
        bboxes = bboxes[pick]
        # Update the existing probabilities with the latest from the refine network
        bboxes[:, 4] = classification[pick, 1].reshape((-1,))
        # Filter out any predicted regressions that did not meet the threshold
        regression = regression[pick]

        # Apply NMS to the remaining bounding boxes
        pick = mtcnn.box.nms(bboxes, 0.5)
        # Translate & adjust the bounding box coordinates
        bboxes = mtcnn.box.translate_bbox(bboxes[pick], regression[pick])
        # Convert the bounding boxes to a square
        bboxes = mtcnn.box.convert_to_square(bboxes)
        # Round the bounding boxes to an integer.
        bboxes[:, :4] = np.round(bboxes[:, :4])

        return bboxes

    def stage3(self, image: Image, bboxes: np.ndarray):
        image_boxes = mtcnn.image.extract_image_box(image, bboxes, 48)

        image_boxes = torch.from_numpy(image_boxes)

        output = self.onet(image_boxes)

        classification = output['classification'].squeeze().data.numpy()
        regression = output['regression'].squeeze().data.numpy()
        landmarks = output['regression'].squeeze().data.numpy()

        pick = np.where(classification[:, 1] > self.thresholds[2])
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

        bboxes = mtcnn.box.translate_bbox(bboxes, regression)
        pick = mtcnn.box.nms(bboxes, 0.5)

        # Convert the bounding boxes to a square
        bboxes = mtcnn.box.convert_to_square(bboxes[pick])
        # Round the bounding boxes to an integer.
        bboxes[:, :4] = np.round(bboxes[:, :4])

        return {
            'bboxes': bboxes,
            'landmarks': landmarks[pick]
        }
