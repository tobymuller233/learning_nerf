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
        self.psnrs = []
        os.system('mkdir -p ' + cfg.result_dir)
        os.system('mkdir -p ' + cfg.result_dir + '/vis')
    def evaluate(self, output, batch):
        pass
    def summarize(self):
        pass