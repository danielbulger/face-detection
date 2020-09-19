"""
Bounding Box transformation functions as described in https://www.programmersought.com/article/90001188428/
"""
from typing import List

import numpy as np


def process_bboxes(bounding_boxes: np.ndarray) -> np.ndarray:
    pick = nms(bounding_boxes[:, :5], 0.1)
    # First find the NMS of all the scaled bounding boxes
    bounding_boxes = bounding_boxes[pick]

    # Correct/translate the positions of the bounding box
    bounding_boxes = translate_bbox(bounding_boxes[:, :5], bounding_boxes[:, 5:])
    # Convert the bounding boxes into a square form
    bounding_boxes = convert_to_square(bounding_boxes)
    # Round the bounding boxes as integers
    bounding_boxes[:, :4] = np.round(bounding_boxes[:, :4])
    return bounding_boxes


def translate_bbox(bboxes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]

    # Convert from row oriented into column oriented
    width = np.expand_dims(x2 - x1 + 1.0, 1)
    height = np.expand_dims(y2 - y2 + 1.0, 1)

    translation = np.hstack([width, height, width, height]) * offsets
    bboxes[:, :4] = bboxes[:, :4] + translation
    return bboxes


def nms(boxes: np.ndarray, threshold: float) -> List:
    """
    Algorithm from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    :author Dr. Tomasz Malisiewicz (https://twitter.com/quantombone)
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


def generate_bbox(probability: np.ndarray, regression: np.ndarray, scale: float, threshold: float) -> np.ndarray:
    (y, x) = np.where(probability > threshold)

    if y.size == 0:
        return np.array([])

    offsets = np.array([regression[i, y, x] for i in range(4)])

    stride = 2
    cell_size = 12

    bounding_boxes = np.vstack([
        np.round((stride * x + 1.0) / scale),
        np.round((stride * y + 1.0) / scale),
        np.round((stride * x + 1.0 + cell_size) / scale),
        np.round((stride * y + 1.0 + cell_size) / scale),
        probability[y, x],
        offsets
    ])

    return bounding_boxes.T


def convert_to_square(bboxes: np.ndarray) -> np.ndarray:

    squares = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]

    width = x2 - x1 + 1.0
    height = y2 - y1 + 1.0

    max_side = np.maximum(height, width)

    squares[:, 0] = x1 + width * 0.5 - max_side * 0.5
    squares[:, 1] = y1 + height * 0.5 - max_side * 0.5
    squares[:, 2] = squares[:, 0] + max_side - 1.0
    squares[:, 3] = squares[:, 1] + max_side - 1.0

    return squares


def crop_bboxes(bboxes: np.ndarray, width: int, height: int) -> List:
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]

    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0

    x, y, ex, ey = x1, y1, x2, y2

    num_boxes = len(bboxes)

    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))

    end_x = w.copy() - 1
    end_y = h.copy() - 1

    indices = np.where(ex > width - 1)[0]
    end_x[indices] = w[indices] + width - 2.0 - ex[indices]
    ex[indices] = width - 1.0

    indices = np.where(ey > height - 1)[0]
    end_y[indices] = h[indices] + height - 2.0 - ey[indices]
    ey[indices] = height - 1.0

    indices = np.where(x < 0.0)[0]
    dx[indices] = -x[indices]
    x[indices] = 0

    indices = np.where(y < 0.0)[0]
    dy[indices] = -y[indices]
    y[indices] = 0

    return [i.astype('int32') for i in [dy, end_y, dx, end_x, y, ey, x, ex, w, h]]
