import os
import imutils
import argparse
import cv2 as cv
import matplotlib.pyplot as plt

import torch
from torchsummary import summary
from models.rotation_net import RotationNet

from utils.file import *
from utils.data import *


def main(img_dir, out_dir, weight_path, gpu_id):
    make_dirs(out_dir)
    img_path_list, length = list_files(img_dir)

    model = RotationNet(out_channels=64)

    print("Load weight from %s" % weight_path)

    if gpu_id >= 0:
        device = torch.device('cuda', gpu_id)
        checkpoint = torch.load(weight_path, map_location=device)
        model.cuda()
        model.load_state_dict(checkpoint)
        model.eval()
        summary(model, (1, 128, 128))
    else:
        device = torch.device('cpu')
        checkpoint = torch.load(weight_path)
        model.to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        summary(model, (1, 128, 128), device="cpu")

    for img_path in img_path_list:
        print(img_path)
        img = cv.imread(img_path, 0)
        name = get_file_name(img_path)

        img_pad, img_transformed = img_transform(img)
        angle = model(img_transformed.to(device))
        angle = angle.detach().cpu().numpy().item()
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
    ap.add_argument('-w', '--weight_path', type=str, required=False, help='Model weight Path',
                    default='./models/model_rotation_net.pt')
    ap.add_argument('-gpu_id', '--gpu_id', type=int, required=False, help='GPU ID. (-1 for CPU)',
                    default=0)
    args = vars(ap.parse_args())

    gpu_id = args['gpu_id']
    img_dir = args['img_dir']
    out_dir = args['out_dir']
    weight_path = args['weight_path']

    main(img_dir, out_dir, weight_path, gpu_id)
