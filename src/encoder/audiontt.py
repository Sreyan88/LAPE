import re
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class NetworkCommonMixIn():
    """Common mixin for network definition."""

    def load_weight(self, weight_file, device):
        """Utility to load a weight file to a device."""

        state_dict = torch.load(weight_file, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Remove unneeded prefixes from the keys of parameters.
        weights = {}
        for k in state_dict:
            m = re.search(r'(^fc\.|\.fc\.|^features\.|\.features\.)', k)
            if m is None: continue
            new_k = k[m.start():]
            new_k = new_k[1:] if new_k[0] == '.' else new_k
            weights[new_k] = state_dict[k]
        # Load weights and set model to eval().
        self.load_state_dict(weights)
        self.eval()
        logging.info(f'Using audio embbeding network pretrained weight: {Path(weight_file).name}')
        return self

    def set_trainable(self, trainable=False):
        for p in self.parameters():
            p.requires_grad = trainable


class AudioNTT2020Task6(nn.Module, NetworkCommonMixIn):
    """ DCASE2020 Task6 NTT Solution Audio Embedding Network.
        Borrowed from: https://github.com/nttcslab/byol-a/blob/master/byol_a/models.py """

    def __init__(self, n_mels, d, return_all_layers):
        super().__init__()

        self.return_all_layers = return_all_layers

        self.features_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))

        self.features_2 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))

        self.features_3 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
            
        self.fc = nn.Sequential(
            nn.Linear(64 * (n_mels // (2**3)), d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )

        self.d = d

    def forward(self, x):

        x = self.features_1(x)

        x_1 = x.permute(0, 3, 2, 1)
        B, T, D, C = x_1.shape
        x_1 = x_1.reshape((B, T, C*D))
        x_1 = torch.mean(x_1, dim=1)

        x = self.features_2(x)

        x_2 = x.permute(0, 3, 2, 1)
        B, T, D, C = x_2.shape
        x_2 = x_2.reshape((B, T, C*D))
        x_2 = torch.mean(x_2, dim=1)

        x = self.features_3(x) # (batch, ch, mel, time)

        x_3 = x.permute(0, 3, 2, 1)
        B, T, D, C = x_3.shape
        x_3 = x_3.reshape((B, T, C*D))
        x_3 = torch.mean(x_3, dim=1)

        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D))
        
        x = self.fc(x)

        if self.return_all_layers:
            return x_1, x_2, x_3, x

        return x

    def __repr__(self):
        return "AudioNTT2020Task6"