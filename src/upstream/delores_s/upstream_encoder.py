import torch
from torch import nn

class DELORES_S(nn.Module):
    """
    Encoder for our AAAI 2022 workshop paper:
    Delores: Decorrelating latent spaces for low-resource audio representation learning
    https://arxiv.org/pdf/2203.13628.pdf
    """
    #@Ashish remove all arguments not required
    def __init__(self, config, base_encoder):
        super().__init__()

        self.return_all_layers = config["pretrain"]["base_encoder"]["return_all_layers"]
        self.encoder = base_encoder(config["pretrain"]["input"]["n_mels"], config["pretrain"]["base_encoder"]["output_dim"], self.return_all_layers)

    def forward(self, x):

        if repr(self.encoder) == "AudioNTT2020Task6":
            x = self.encoder(x)
        else:
            raise NotImplementedError("DELORES_S currently supports just AudioNTT2020Task6 encoder")
        
        if self.return_all_layers:
            x = x[-1]

        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2

        return x



        