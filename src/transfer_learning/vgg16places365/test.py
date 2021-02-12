import numpy as np
from PIL import Image
from cv2 import resize

from vgg16_hybrid_places_1365 import VGG16_Hybrid_1365

image = Image.open('test.jpg')
image = np.array(image, dtype=np.uint8)
image = resize(image, (224, 224))
image = np.expand_dims(image, 0)

model = VGG16_Hybrid_1365(weights='places', include_top=False)
features = model.predict(image)
print(features.shape)
