import cv2 as cv
import numpy as np
from src.utils.data import *
from src.utils.file import *
import onnxruntime as ort

onnx_path = 'models/rotation_net.onnx'
ort_session = ort.InferenceSession(onnx_path)


def predict(image):
    img = to_array(image)
    _, img_transformaed = img_transform_onnx(img)
    ort_inputs = {'input': img_transformaed}
    ort_outs = ort_session.run(None, ort_inputs)
    angle = ort_outs[0][0][0]
    angle = to_std_angle(angle)
    return angle
