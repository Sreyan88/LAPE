import re
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import argparse
import sys


class SSMAST(nn.Module):
    """
    SSL pre-training encoder setup for our ICASSP 2023 paper:
    MAST: MULTISCALE AUDIO SPECTROGRAM TRANSFORMERS
    https://arxiv.org/pdf/2211.01515.pdf
    """
    def __init__(self, config, base_encoder):
        super(SSMAST, self).__init__()
        self.encoder = base_encoder(config)
        emb_dim = config["pretrain"]["base_encoder"]["emb_dim"] #768
        fc = OrderedDict([])
        fc['fc1'] = torch.nn.Linear(emb_dim, config["pretrain"]["base_encoder"]["out_dim"]) #256
        self.mlp = torch.nn.Sequential(fc)

    def forward(self, x):
        z = self.encoder(x)
        x = self.mlp(z.float())

        return x
