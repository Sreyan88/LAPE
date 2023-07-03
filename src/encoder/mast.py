import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from timm.models.layers import to_2tuple,trunc_normal_
import numpy as np


# override the timm package to relax the input shape constraint.
class Patch_Embed(nn.Module):
    def __init__(self, kernel=(7,7), stride=(7,7), dim_in=3, dim_out=768, img_size = (224, 224), padding=(3, 3)):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(kernel)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_size = patch_size
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=patch_size, stride=patch_size)
        self.num_patches = num_patches

    def forward(self, x):
        # B C H W -> B HW C
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2), x.shape[-2:]

class MAST(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, config):
        
        self.verbose = config["pretrain"]["base_encoder"]["verbose"]
        self.label_dim = config["pretrain"]["base_encoder"]["label_dim"]
        self.fstride = config["pretrain"]["base_encoder"]["fstride"]
        self.tstride = config["pretrain"]["base_encoder"]["tstride"]
        self.input_fdim = config["pretrain"]["base_encoder"]["input_fdim"]
        self.input_tdim = config["pretrain"]["base_encoder"]["input_tdim"]
        self.imagenet_pretrain = config["pretrain"]["base_encoder"]["imagenet_pretrain"]
        self.audioset_pretrain = config["pretrain"]["base_encoder"]["audioset_pretrain"]
        self.model_size = config["pretrain"]["base_encoder"]["model_size"]

        super(MAST, self).__init__()

        if self.verbose == True:
            print('---------------MAST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(self.imagenet_pretrain),str(self.audioset_pretrain)))

        timm.models.mvitv2.PatchEmbed = Patch_Embed
        self.mlp_head = None
        self.has_cls = False

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if self.audioset_pretrain == False:
            if self.model_size == 'large':
                self.v = timm.create_model('mvitv2_large', pretrained=self.imagenet_pretrain)
            elif self.model_size == 'small':
                self.v = timm.create_model('mvitv2_small', pretrained=self.imagenet_pretrain)
            elif self.model_size == 'tiny':
                self.v = timm.create_model('mvitv2_tiny', pretrained=self.imagenet_pretrain)
            elif self.model_size == 'base':
                self.v = timm.create_model('mvitv2_base', pretrained=self.imagenet_pretrain)
            elif self.model_size == 'small_cls':
                self.v = timm.create_model('mvitv2_small_cls', pretrained=self.imagenet_pretrain)
                self.has_cls = True
            else:
                raise Exception('Model size must be one of tiny, small, base, large or small_cls.')
            
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # optional to specify the last MLP layer for a specific class
            if self.label_dim is not None:
                self.mlp_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, self.label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(self.fstride, self.tstride, self.input_fdim, self.input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if self.verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(self.fstride, self.tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16,16), stride=(self.fstride, self.tstride))
            if self.imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if self.imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                # self.v.patch_embed.num_patches + 1 if cls token
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif self.audioset_pretrain == True:
            if self.audioset_pretrain == True and self.imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if self.model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists('../../pretrained_models/audioset_10_10_0.4593.pth') == False:
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='../../pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('../../pretrained_models/audioset_10_10_0.4593.pth', map_location=device)
            audio_model = MAST(config)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            if self.label_dim is not None:
                self.mlp_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, self.label_dim))

            f_dim, t_dim = self.get_shape(self.fstride, self.tstride, self.input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if self.verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(self.fstride, self.tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, self.input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(self.fstride, self.tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x, patch_drop=0, return_cls=False):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        x = x.transpose(2, 3)

        B = x.shape[0]
        x, x_shape = self.v.patch_embed(x)
        
        H, W = x_shape

        if patch_drop > 0:
            patch_keep = 1. - patch_drop
            T_H = int(np.floor((x.shape[1])*patch_keep))
            perm = torch.randperm(x.shape[1])[:T_H]  # keep class token
            idx = torch.tensor(perm,dtype=perm.dtype, device=perm.device)
            x = x[:, idx, :]
                
        thw = [H, W]
        for blk in self.v.stages:
            print(x.shape)
            x, thw = blk(x, thw)
            
        if self.has_cls and return_cls:
            x = self.v.norm(x) # layer norm only if return_cls = False
            x = x[:, 0]
        else:
            x = x.mean(1) # mean if no cls token

        if self.mlp_head is not None:
            x = self.mlp_head(x)

        return x
    

    def __repr__(self):
        return "MAST"