import os
import time
import json
import torch
import librosa
import pickle
import random
import logging
import argparse
import numpy as np
import torch.nn.functional as F

from os.path import join as path_join
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

import importlib


class MelSpectrogramLibrosa:
    """Mel spectrogram using librosa."""
    def __init__(self, fs=16000, n_fft=1024, shift=160, n_mels=64, fmin=60, fmax=7800):
        self.fs, self.n_fft, self.shift, self.n_mels, self.fmin, self.fmax = fs, n_fft, shift, n_mels, fmin, fmax
        self.mfb = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def __call__(self, audio):
        X = librosa.stft(np.array(audio), n_fft=self.n_fft, hop_length=self.shift)
        return torch.tensor(np.matmul(self.mfb, np.abs(X)**2 + np.finfo(float).eps))


def check_downstream_hf_availability(task):

    task_to_loc = {
        "speech_commands_v1" : "hf",
        "speech_commands_v2" : "hf",
        "speech_commands_v235" : "hf",
    }
    try:
        return task_to_loc[task]
    except:
        return "nhf"

def extract_log_mel_spectrogram(waveform, to_mel_spec):
    """Mel spectrogram using librosa.
    waveform: torch tenspr waveform
    to_mel_spec: object of MelSpectrogramLibrosa class"""

    log_mel_spectrograms = (to_mel_spec(waveform) + torch.finfo().eps).log()
    return log_mel_spectrograms

def compute_features(args, dataloader, model, N): #N is total dataset size
    batch = args.batch_size
    verbose = True
    if verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor) in enumerate(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())
            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * batch: (i + 1) * batch] = aux
            else:
                # special treatment for final batch
                features[i * batch:] = aux

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if verbose and (i % 50) == 0:
                print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

def extract_window(wav, duration=16000, data_size=None):
    """Extract random window of data_size second"""
    if data_size:
        unit_length = int(data_size * 16000)
    else:    
        unit_length = duration
    length_adj = unit_length - len(wav)
    if length_adj > 0:
        half_adj = length_adj // 2
        wav = F.pad(wav, (half_adj, length_adj - half_adj))

    # random crop unit length wave
    length_adj = len(wav) - unit_length
    start = random.randint(0, length_adj) if length_adj > 0 else 0
    wav = wav[start:start + unit_length]

    return wav


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def loss_fn_mse(x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        l = 2 - 2 * (x * y).sum(dim=-1)
        #print(l)
        #print(l.shape)
        return l.mean()

def load_pretrained_encoder(model, args):
    module_path_expert = f'src.upstream.{args.upstream}.upstream_expert'
    expert = getattr(importlib.import_module(module_path_expert), 'Upstream_Expert')
    backbone = expert.load_from_checkpoint(args.checkpoint, strict=False)
    wts = backbone.encoder_q.state_dict()
    mod_missing_keys,mod_unexpected_keys = model.module.load_state_dict(wts,strict=False)
    print('Missing Keys:  ',mod_missing_keys)
    print('Unexpected Keys:  ',mod_unexpected_keys)
    return model


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def freeze_encoder(model):
    logger=logging.getLogger("__main__")
    logger.info("freezing encoder weights")
    for param in model.encoder.parameters():
        param.requires_grad = False


def get_logger(args):
    logger = logging.getLogger(__name__)
    f_handler = logging.FileHandler(os.path.join(args.exp_root,'train.log'))
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)
    logger.setLevel(logging.DEBUG)
    return logger


class Metric(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if isinstance(val, (torch.Tensor)):
            val = val.numpy()
            self.val = val
            self.sum += np.sum(val)
            self.count += np.size(val)
        self.avg = self.sum / self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def select_columns(task):
    task2x = {
        'speech_commands': 'audio'
    }

    task2y = {
        'speech_commands': 'label'
    }

    return task2x[task], task2y[task]
