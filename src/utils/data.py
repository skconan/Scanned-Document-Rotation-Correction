import io
import base64
import torch
import cv2 as cv
import numpy as np
from PIL import Image


def padding_to_128(img):
    expected_size = 128
    rows, cols = img.shape
    if rows >= cols:
        ratio = expected_size / rows
    else:
        ratio = expected_size / cols
    img = cv.resize(img, None, fx=ratio, fy=ratio)

    rows, cols = img.shape
    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0
    if rows < 128:
        diff = 128 - rows
        pad_top = diff // 2
        pad_bottom = diff - pad_top
    if cols < 128:
        diff = 128 - cols
        pad_left = diff // 2
        pad_right = diff - pad_left

    img_pad = np.pad(img, ((pad_top, pad_bottom), (pad_left,
                     pad_right)), 'constant', constant_values=0)
    return img_pad


def img_transform(img):
    img_pad = padding_to_128(img)
    img = img_pad / 255.
    img = torch.Tensor(img.reshape(1, 1, 128, 128))
    return img_pad, img.float()


def to_std_angle(angle):
    return angle - 90


def to_array(base64_img):
    base64_img = base64_img.replace('data:image/jpeg;base64,', '')
    base64_img = base64_img.replace('data:image/png;base64,', '')
    base64_decoded = base64.b64decode(base64_img)
    image = Image.open(io.BytesIO(base64_decoded))
    image = np.array(image)
    if len(image.shape) >= 3:
        nd_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        nd_img = image
    return nd_img
