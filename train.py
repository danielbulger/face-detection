import mtcnn.image
from mtcnn import RNet, ONet, PNet

pnet, rnet, onet = PNet(), RNet(), ONet()

tensors, scales = mtcnn.image.get_image_tensors('data/image.jpg', 0.709, 20)

for tensor, scale in zip(tensors, scales):
    print(scale, pnet(tensor))
