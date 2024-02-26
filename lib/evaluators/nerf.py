import numpy as np
from lib.config import cfg
import os
import imageio
from lib.utils import img_utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
import torch
import lpips
import imageio
from lib.utils import img_utils
import cv2
import json
import ipdb

class Evaluator:
    def __init__(self, ):
        self.counter = 0
        self.psnrs = []
        self.rgbs = []
        os.system('mkdir -p ' + cfg.result_dir)
        os.system('mkdir -p ' + cfg.result_dir + '/vis')
    def evaluate(self, output, batch):
        rgb = output['rgb_map']
        rgb = rgb.cpu().numpy()
        self.rgbs.append(rgb)
        rgb_8 = (255 * np.clip(rgb, 0, 1)).astype(np.uint8)
        filename = os.path.join(cfg.result_dir, 'rgb_{:04d}.png'.format(self.counter))
        self.counter += 1
        imageio.imwrite(filename, rgb_8)
    def summarize(self):
        pass