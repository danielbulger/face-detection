import numpy as np
import mtcnn.image
from mtcnn import RNet, ONet, PNet
from mtcnn import box

pnet, rnet, onet = PNet(), RNet(), ONet()

tensors, scales = mtcnn.image.get_image_tensors('data/image.jpg', 0.709, 20)


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

    if bounding_boxes:
        return box.process_bboxes(np.vstack(bounding_boxes))

    return []

print(stage1(0.3))
