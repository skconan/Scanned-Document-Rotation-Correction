import io
import base64
import cv2 as cv
import numpy as np
from PIL import Image


def to_array(base64_img):
    base64_img = base64_img.replace('data:image/jpeg;base64,', '')
    base64_img = base64_img.replace('data:image/png;base64,', '')
    base64_decoded = base64.b64decode(base64_img)
    image = Image.open(io.BytesIO(base64_decoded))
    image = np.array(image)
    print(image.shape)
    if len(image.shape) >= 3:
        nd_img = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
    else:
        nd_img = image 
    return nd_img