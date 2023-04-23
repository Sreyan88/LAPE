import torch
from torch import nn

class DELORES_M(nn.Module):
    """
    Encoder for our IEEE JSTSP Paper:
    Decorrelating Feature Spaces for Learning General-Purpose Audio Representations
    https://ieeexplore.ieee.org/document/9868132
    """
    
    def __init__(self, config, base_encoder):
        super().__init__()

        self.return_all_layers = config["pretrain"]["base_encoder"]["return_all_layers"]
        self.encoder = base_encoder(config["pretrain"]["input"]["n_mels"], config["pretrain"]["base_encoder"]["output_dim"], self.return_all_layers)
        self.fc = nn.Linear(config["pretrain"]["base_encoder"]["output_dim"], config["pretrain"]["contrastive_dim"])

    def forward(self, x):

        if repr(self.encoder) == "AudioNTT2020Task6":
            x = self.encoder(x)
        else:
            raise NotImplementedError("DELORES_M currently supports just AudioNTT2020Task6 encoder")

        if self.return_all_layers == False:
            raise NotImplementedError("DELORES_M needs return_all_layers = True to be set in the config!")
        else:
            l1, l2, l3, x = x
        
        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2

        x = self.fc(x)

        return x, l1, l2, l3