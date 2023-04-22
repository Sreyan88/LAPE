import torch
from torch import nn

class UNFUSED(nn.Module):
    """
    Encoder for our IEEE JSTSP Paper:
    Decorrelating Feature Spaces for Learning General-Purpose Audio Representations
    https://ieeexplore.ieee.org/document/9868132
    """
    
    def __init__(self, config, base_encoder):
        super().__init__()

        self.encoder = base_encoder(config["pretrain"]["input"]["n_mels"], config["pretrain"]["base_encoder"]["output_dim"], config["pretrain"]["base_encoder"]["return_all_layers"])
    def forward(self, x):

        if repr(self.encoder) == "AudioNTT2020Task6":
            x, x_1, x_2, x_3 = self.encoder(x)
        else:
            raise NotImplementedError("Unfused currently supports just AudioNTT2020Task6 encoder")

        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2

        return x, (x_1, x_2, x_3)