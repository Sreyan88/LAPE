import os
import sys
import torch
import yaml
import argparse
import importlib
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.dataset import BaselineDataModule
from src.augmentations import AugmentationModule

def main(args):


    if args.config is None:
        default_upstream_config = "src/upstream/" + args.upstream + "/config.yaml"
        with open(default_upstream_config, 'r') as duc:
            config = yaml.load(duc, Loader=yaml.FullLoader)
    else:
        with open(args.config, 'r') as duc:
            config = yaml.load(duc, Loader=yaml.FullLoader)
    print(config)

    # load augmentation module
    tfms = AugmentationModule(config, len(pd.read_csv(args.input)))
    
    dm = BaselineDataModule(config, args, tfms, data_csv = args.input, num_workers=config["run"]["num_dataloader_workers"], batch_size=config["run"]["batch_size"]) 
    
    # load upstream expert
    module_path_expert = f'src.upstream.{args.upstream}.upstream_expert'
    expert = getattr(importlib.import_module(module_path_expert), 'Upstream_Expert')

    # load base encoder
    module_path_base_encoder = f'src.encoder'
    base_encoder = getattr(importlib.import_module(module_path_base_encoder), config["pretrain"]["base_encoder"]["type"])
    
    model = expert(config, base_encoder=base_encoder, datamodule=dm)

    # @Ashish do we need lambda for every term, please check this and modularize, 
    # add to conf, also please remove from dir_path, also remove from parser
    # lamb_append_term = '-'.join(np.array(args.lamb_values).astype(str))
    
    checkpoint_callback = ModelCheckpoint(
                                dirpath=config["run"]["save_path"]+'_chkp',
                                filename='{epoch}',
                                monitor="train_loss", 
                                mode="min",
                                save_top_k=1)
        
    if torch.cuda.is_available():
        if args.load_checkpoint:
            trainer = pl.Trainer(gpus=config["run"]["world_size"], callbacks = [checkpoint_callback], accelerator="gpu", strategy="ddp", resume_from_checkpoint=args.load_checkpoint)
        else:
            trainer = pl.Trainer(gpus=config["run"]["world_size"], callbacks = [checkpoint_callback], max_epochs=1)
    else:
        trainer = pl.Trainer(checkpoint_callback = checkpoint_callback,)
    
    trainer.fit(model, dm)
    trainer.save_checkpoint("/speech/ashish/example_test.ckpt")


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Clean the ones not required @Ashish

    # Add data arguments
    parser.add_argument("--input", help="path to data directory", type=str, default='/speech/ashish/test_audio.csv')
    parser.add_argument('--load_checkpoint', type=str, help='load checkpoint', default = None)
    parser.add_argument('-c', '--config', metavar='CONFIG_PATH', help='The yaml file for configuring the whole experiment, except the upstream model', default = None)
    parser.add_argument('--upstream', type=str, help='define the type of upstream', default = 'delores_m')
    # Add model arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)