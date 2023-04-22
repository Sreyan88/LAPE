import torch
from torch import nn

class SLICER(nn.Module):
    """
    Encoder for our IEEE JSTSP Paper:
    Decorrelating Feature Spaces for Learning General-Purpose Audio Representations
    https://ieeexplore.ieee.org/document/9868132
    """
    
    def __init__(self, config, base_encoder):
        super().__init__()

        self.encoder = base_encoder(config["pretrain"]["input"]["n_mels"], config["pretrain"]["base_encoder"]["output_dim"], config["pretrain"]["base_encoder"]["return_all_layers"])
        self.instance_projector = nn.Linear(config["pretrain"]["base_encoder"]["output_dim"], config["pretrain"]["instance_contrastive_dim"])
        self.cluster_projector = nn.Sequential(
            nn.Linear(config["pretrain"]["base_encoder"]["output_dim"], config["pretrain"]["base_encoder"]["output_dim"]),
            nn.ReLU(),
            nn.Linear(config["pretrain"]["base_encoder"]["output_dim"], config["pretrain"]["cluster_contrastive_dim"]),
            nn.Softmax(dim=1)
        )
    def forward(self, x):

        if repr(self.encoder) == "AudioNTT2020Task6":
            x = self.encoder(x)
        else:
            raise NotImplementedError("SLICER currently supports just AudioNTT2020Task6 encoder")

        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2

        x_instance = self.instance_projector(x)
        x_cluster = self.cluster_projector(x)

        return x_instance, x_cluster