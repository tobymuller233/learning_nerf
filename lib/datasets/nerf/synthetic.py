import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2
import ipdb
import torch

def get_rays(H, W, K, c2w):
    # i, j = torch.meshgrid(torch.arange(W, dtype = torch.float32), torch.arange(H, dtype = torch.float32), indexing='xy')
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))      
    i = i.t()
    j = j.t()
    # the direction relative to the camera coordination
    # [400, 400, 3] the world coordination of the pixel
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], axis=-1)
    # rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
    # get the origin of all the rays
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        # view = kwargs['view']
        self.input_ratio = kwargs['input_ratio']
        self.data_root = os.path.join(data_root, scene)
        self.split = split
        self.batch_size = cfg.task_arg.N_rays   # 1024 by default

        metas = {}
        camera_focals = {}
        image_paths = []
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format(split))))
        
        all_images = []
        all_poses = []
        # read in all the images and poses
        for frame in json_info['frames']:
            file_name = os.path.join(self.data_root, frame['file_path'][2:] + '.png')
            all_images.append(imageio.imread(file_name))
            all_poses.append(np.array(frame['transform_matrix']))
        
        # the dataset is split into 3 parts, train, val, test
        # images = np.concatenate(all_images, 0)
        # poses = np.concatenate(all_poses, 0)
        images = (np.array(all_images) / 255.).astype(np.float32)
        poses = (np.array(all_poses)).astype(np.float32)

        if self.split == "train":
            self.input_ratio = cfg.train_dataset.input_ratio
        elif self.split == "test":
            self.input_ratio = cfg.test_dataset.input_ratio
        else:
            self.input_ratio = 1.

        H, W = images[0].shape[:2] # get the width and height of the image
        self.camera_angle = json_info['camera_angle_x']
        self.camera_focal = .5 * W / np.tan(.5 * self.camera_angle) * self.input_ratio
        self.H = int(H * self.input_ratio)
        self.W = int(W * self.input_ratio)
        self.images = np.zeros((images.shape[0], self.H, self.W, 4))
        for i, img in enumerate(images):            # do the crop
            self.images[i] = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        self.poses = poses
        self.N_rand = 1024  # 1024 points in each image

        
        self.K = np.array([self.camera_focal, 0., W/2., 
                            0., self.camera_focal, H/2.,
                            0., 0., 1.]).reshape(3, 3)

    def __getitem__(self, index):
        target = self.images[index]
        pose = self.poses[index]
        rays_o, rays_d = get_rays(self.H, self.W, self.K, torch.Tensor(pose))  # get the rays and the origin points of the image
        coords = torch.stack(torch.meshgrid(torch.linspace(0, self.H - 1, self.H), torch.linspace(0, self.W - 1, self.W)), -1)
        # reshape the tensor
        coords = torch.reshape(coords, [-1, 2]) # [H * W, 2]
        # choose N_rand rays
        select_coords = np.random.choice(coords.shape[0], self.N_rand, replace=False)   # no duplication
        select_coords = coords[select_coords].long() # type transition
        select_rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
        select_rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
        select_targets = target[select_coords[:, 0], select_coords[:, 1]]
        ret = {'rays_o': select_rays_o, 'rays_d': select_rays_d, 'target': select_targets}
        return ret
    
    def __len__(self):
        return self.images.shape[0]
