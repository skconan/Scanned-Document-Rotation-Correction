import os
import imutils
import argparse
import cv2 as cv
import matplotlib.pyplot as plt

import onnxruntime as ort

from utils.file import *
from utils.data import *


def main(img_dir, out_dir, model_path):
    make_dirs(out_dir)
    img_path_list, length = list_files(img_dir)

    print("Load model from %s" % model_path)
    ort_session = ort.InferenceSession(model_path)

    for img_path in img_path_list:
        print(img_path)
        img = cv.imread(img_path, 0)
        name = get_file_name(img_path)

        img_pad, img_transformed = img_transform_onnx(img)
        ort_inputs = {'input': img_transformed}
        ort_outs = ort_session.run(None, ort_inputs)
        angle = ort_outs[0][0][0]
        angle = to_std_angle(angle)

        img_correction = imutils.rotate_bound(img, -angle)

        out_path = os.path.join(out_dir, '%s.png' % name)
        _, axs = plt.subplots(1, 3, figsize=(30, 10))
        plt.title("Angle error: %.2f" % angle)
        axs[0].imshow(img, cmap='gray')
        axs[1].imshow(img_pad, cmap='gray')
        axs[2].imshow(img_correction, cmap='gray')

        plt.savefig(out_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--img_dir', type=str, required=False, help='Scanned Document Image Directory',
                    default='./dataset/testing_set')
    ap.add_argument('-o', '--out_dir', type=str, required=False, help='Output Directory',
                    default='./testing_output')
    ap.add_argument('-m', '--model_path', type=str, required=False, help='Model weight Path',
                    default='./models/model_rotation_net.onnx')

    args = vars(ap.parse_args())

    img_dir = args['img_dir']
    out_dir = args['out_dir']
    model_path = args['model_path']

    main(img_dir, out_dir, model_path)
