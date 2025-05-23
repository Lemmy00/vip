# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as T


class DinoDistModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.outdim = 768  
        self.fc = nn.Linear(self.outdim, hidden_dim)

    def forward(self, x):
        features = self.dino(x)
        return self.fc(features)


class VIP(nn.Module):
    def __init__(self, device="cuda", lr=1e-4, hidden_dim=1024, size=50, l2weight=1.0, l1weight=1.0, gamma=0.98, num_negatives=0):
        super().__init__()
        self.device = device
        self.l2weight = l2weight
        self.l1weight = l1weight

        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.size = size # Resnet size
        self.num_negatives = num_negatives

        ## Distances and Metrics
        self.cs = torch.nn.CosineSimilarity(1)
        self.bce = nn.BCELoss(reduce=False)
        self.sigm = Sigmoid()

        params = []
        ######################################################################## Sub Modules
        ## Visual Encoder
        if size == 18:
            self.outdim = 512
            self.convnet = torchvision.models.resnet18(pretrained=False)
        elif size == 34:
            self.outdim = 512
            self.convnet = torchvision.models.resnet34(pretrained=False)
        elif size == 50:
            self.outdim = 2048
            self.convnet = torchvision.models.resnet50(pretrained=False)
        elif size == 0:
            from transformers import AutoConfig
            self.outdim = 768
            assert hidden_dim > 0
            self.convnet = DinoDistModel(self.hidden_dim)

        if hidden_dim  > 0:
            self.convnet.fc = nn.Linear(self.outdim, hidden_dim)
        else:
            self.convnet.fc = Identity()
        self.convnet.train()
        params += list(self.convnet.parameters())        

        ## Optimizer
        self.encoder_opt = torch.optim.Adam(params, lr = lr)

    ## Forward Call (im --> representation)
    def forward(self, obs):
        obs_shape = obs.shape[1:]
        # if not already resized and cropped, then add those in preprocessing
        if obs_shape != (3, 224, 224):
            preprocess = nn.Sequential(
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                )
        else:
            preprocess = nn.Identity()
        
        ## Input must be [0, 255], [3,224,224]
        if torch.max(obs) > 2.0:
            obs = obs.float() /  255.0
        obs_p = preprocess(obs)
        h = self.convnet(obs_p)
        return h

    def sim(self, tensor1, tensor2):
        d = -torch.linalg.norm(tensor1 - tensor2, dim = -1)
        return d
