import re
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

from utils import off_diagonal
from utils import trunc_normal_
from collections import OrderedDict
import argparse
import sys


class SSMAST(nn.Module):
    """
    SSL pre-training encoder setup for our ICASSP 2023 paper:
    MAST: MULTISCALE AUDIO SPECTROGRAM TRANSFORMERS
    https://arxiv.org/pdf/2211.01515.pdf
    """
    def __init__(self, encoder, out_dim, use_bn = False, norm_last_layer=True, n_layers=3, hidden_dim=512, n_mels=64, d=768, output_dim=256):
        super(SSMAST, self).__init__()
        self.encoder = encoder
        emb_dim = d
        fc = OrderedDict([])
        fc['fc1'] = torch.nn.Linear(emb_dim, output_dim)
        self.mlp = torch.nn.Sequential(fc)

    def forward(self, batch, return_before_head=False):

        z = self.encoder(batch, patch_drop=0.0)
        x = self.mlp(z.float())

        return x
