import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

class Efficient_Net(nn.Module):

    def __init__(self,args):
        super(AAAI_BARLOW, self).__init__()

        self.args = args
        self.units = args.final_units
        self.model_efficient = EfficientNet.from_name('efficientnet-b0',include_top = False, in_channels = 1, image_size = None)

    def forward(self,batch1, batch2):

        z = self.model_efficient(batch1)
        x = z.flatten(start_dim=1) #1280 (already swished)

        return x

    def __repr__(self):
        return "Efficient_Net"
        