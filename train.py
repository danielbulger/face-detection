
from PIL import Image

from mtcnn import MTCNN

image = Image.open('data/image.jpg')

mtcnn = MTCNN(0.709, 20, [0.5, 0.5, 0.5])

print(mtcnn.forward(image))
