import torch
from torch import nn

class UNFUSED(nn.Module):
    """
    Encoder for our IEEE ICASSP SASB Workshop Paper:
    UNFUSED: UNsupervised Finetuning Using SElf supervised Distillation
    https://arxiv.org/pdf/2303.05668.pdf
    """
    
    def __init__(self, config, base_encoder):
        super().__init__()

        self.encoder = base_encoder(config["pretrain"]["input"]["n_mels"], config["pretrain"]["base_encoder"]["output_dim"], config["pretrain"]["base_encoder"]["return_all_layers"])
    def forward(self, x):

        if repr(self.encoder) == "AudioNTT2020Task6":
            x = self.encoder(x)
        else:
            raise NotImplementedError("UNFUSED currently supports just AudioNTT2020Task6 encoder")

        if self.return_all_layers == False:
            raise NotImplementedError("UNFUSED needs return_all_layers = True to be set in the config!")
        else:
            l1, l2, l3, x = x

        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2

        return x, (l_1, l_2, l_3)