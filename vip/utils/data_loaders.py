# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import torchvision
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
from PIL import Image

import cv2
import glob
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random


def get_ind(vid, index, ds="ego4d"):
    if ds == "ego4d":
        return torchvision.io.read_image(f"{vid}{index:06}.jpg")
    else:
        try:
            return torchvision.io.read_image(f"{vid}/{index}.jpg")
        except: 
            return torchvision.io.read_image(f"{vid}/{index}.png")

## Data Loader for VIP
class VIPBuffer(IterableDataset):
    def __init__(self, datasource='ego4d', datapath=None, data_type=".png", num_workers=10, doaug = "none", task_type="man"):
        self._num_workers = max(1, num_workers)
        self.datasource = datasource
        self.datapath = datapath
        self.data_type = data_type
        assert(datapath is not None)
        assert(data_type is not None)
        self.doaug = doaug
        self.task_type = task_type
        
        # Augmentations
        self.preprocess = torch.nn.Sequential(
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                )
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            )
        else:
            self.aug = lambda a : a

        # Load Data
        if "ego4d" == self.datasource:
            print("Ego4D")
            self.manifest = pd.read_csv(f"{self.datapath}/manifest.csv")
            print(self.manifest)
            self.ego4dlen = len(self.manifest)

    def _sample(self):        
        # Sample a video from datasource
        if self.datasource == 'ego4d':
            vidid = np.random.randint(0, self.ego4dlen)
            m = self.manifest.iloc[vidid]
            vidlen = m["len"]
            vid = m["path"]
        else: 
            video_paths = glob.glob(f"{self.datapath}/[0-9]*")
            num_vid = len(video_paths)

            video_id = np.random.randint(0, int(num_vid)) 
            vid = f"{video_paths[video_id]}"

            if self.data_type in ['.png', '.jpg']:
                vidlen = len(glob.glob(f'{vid}/*{self.data_type}'))
            elif self.data_type == '.mp4':
                loaded_video, _, _ = torchvision.io.read_video(os.path.join(vid, "trajectory.mp4"), pts_unit='sec', output_format='TCHW')
                loaded_video = loaded_video.to(torch.float32) / 255.0
                vidlen = len(loaded_video)
            else:
                raise ValueError(f'Expected data types are: ".png", ".jpg", or ".mp4" but {self.data_type} given.')

        # Sample (o_t, o_k, o_k+1, o_T) for VIP training
        start_ind = np.random.randint(0, vidlen-2)  
        end_ind = np.random.randint(start_ind+1, vidlen)

        s0_ind_vip = np.random.randint(start_ind, end_ind)
        if s0_ind_vip + 1 == end_ind:
            s1_ind_vip = end_ind
        else:
            s1_ind_vip = np.random.randint(s0_ind_vip + 1, min(s0_ind_vip + 5, end_ind))
        
        # Self-supervised reward
        reward = (sum([(float((s0_ind_vip + (i + 1)) == end_ind) - 1) * (0.98 ** i) for i in range(s1_ind_vip - s0_ind_vip)]), float(start_ind - end_ind))
        
        if self.data_type == '.mp4':
            assert(loaded_video is not None)
            im0 = loaded_video[start_ind] 
            img = loaded_video[end_ind]
            imts0_vip = loaded_video[s0_ind_vip] 
            imts1_vip = loaded_video[s1_ind_vip] 
        else:
            im0 = get_ind(vid, start_ind, self.datasource) 
            img = get_ind(vid, end_ind, self.datasource)
            imts0_vip = get_ind(vid, s0_ind_vip, self.datasource)
            imts1_vip = get_ind(vid, s1_ind_vip, self.datasource)

        if self.doaug == "rctraj":
            ### Encode each image in the video at once the same way
            allims = torch.stack([im0, img, imts0_vip, imts1_vip], 0)
            allims_aug = self.aug(allims / 255.0) * 255.0

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0_vip = allims_aug[2]
            imts1_vip = allims_aug[3]
        else:
            ### Encode each image individually
            im0 = self.aug(im0 / 255.0) * 255.0
            img = self.aug(img / 255.0) * 255.0
            imts0_vip = self.aug(imts0_vip / 255.0) * 255.0
            imts1_vip = self.aug(imts1_vip / 255.0) * 255.0

        im = torch.stack([im0, img, imts0_vip, imts1_vip])
        im = self.preprocess(im)
        return (im, reward)

    def __iter__(self):
        while True:
            yield self._sample()