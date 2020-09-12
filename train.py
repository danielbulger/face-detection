import mtcnn.image
import mtcnn.pnet

model = mtcnn.pnet.PNet()

# Parameters taken from https://arxiv.org/pdf/1910.06261.pdf
tensors = mtcnn.image.get_image_tensors('data/image.jpg', 0.709, 20)

for tensor in tensors:
    print(model.forward(tensor))
