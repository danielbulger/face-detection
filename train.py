import torch
import mtcnn.image
from mtcnn import RNet, ONet, PNet

pnet = PNet()
rnet = RNet()
onet = ONet()

# Parameters taken from https://arxiv.org/pdf/1910.06261.pdf
tensors = mtcnn.image.get_image_tensors('data/image.jpg', 0.709, 20)

pnet.forward(tensors[0])
rnet.forward(torch.zeros(1, 3, 24, 24))
onet.forward(torch.zeros(1, 3, 48, 48))
