import torch
import torch.nn as nn

from src.encoder.audiontt import AudioNTT2020Task6
from src.encoder.efficientnet import Efficient_Net
from src.encoder.mast import MAST as mast

class MAST(nn.Module):
    """
    Encoder for our ICASSP 2023 paper:
    MAST: MULTISCALE AUDIO SPECTROGRAM TRANSFORMERS
    https://arxiv.org/pdf/2211.01515.pdf
    """

    def __init__(self, label_dim=None, input_tdim=1024, imagenet_pretrain = False, audioset_pretrain = False, model_size='base', return_cls=False, patch_drop=0.1):
        super(MAST, self).__init__()

        self.mast_model = mast(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain = imagenet_pretrain, audioset_pretrain = audioset_pretrain, model_size=model_size, return_cls=return_cls)
        self.patch_drop = patch_drop

    def forward(self, batch):

        z = self.mast_model(batch, patch_drop=self.patch_drop)

        return z
