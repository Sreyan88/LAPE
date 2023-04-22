import re
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

class DownstreamEncoder(nn.Module):
    """
    Encoder for running Downstream Tasks
    """
    
    def __init__(self, config, args, base_encoder, no_of_classes):
        super().__init__()
        self.config = config
        self.output_layer = config["downstream"]['finetune_layer']
        self.return_all_layers = config["downstream"]["base_encoder"]['return_all_layers']
        self.interim_layer_output_shapes = config["downstream"]["base_encoder"]['interim_layer_output_shapes']
        self.output_dim = config["downstream"]["base_encoder"]["output_dim"]

        self.encoder = base_encoder(config["downstream"]["input"]["n_mels"], config["downstream"]["base_encoder"]["output_dim"], self.return_all_layers)

        if self.output_layer == -1:
            self.final = nn.Linear(self.output_dim, no_of_classes)
        else:
            if self.return_all_layers == False:
                raise Exception("Please set return_all_layers=True in config for taking representations from any intermediate layer")
            elif len(self.interim_layer_output_shapes) < self.output_layer:
                raise Exception("Number of layers exceed number of intemediate layers")
            else:
                self.final = nn.Linear(self.interim_layer_output_shapes[self.output_layer], no_of_classes)
    
    def forward(self, x):

        if repr(self.encoder) != "AudioNTT2020Task6":
            raise NotImplementedError("Downstream currently supports just AudioNTT2020Task6 encoder")

        x = self.encoder(x)

        if self.return_all_layers:
            x = torch.mean(x[self.output_layer], dim=1)
        else:
            x = torch.mean(x, dim=1)

        return self.final(x)