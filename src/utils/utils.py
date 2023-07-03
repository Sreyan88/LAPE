import os
import time
import json
import torch
import math
import librosa
import pickle
import random
import logging
import importlib
import argparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from os.path import join as path_join
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


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
        "speech_commands_v2_35" : "hf",
    }
    try:
        return task_to_loc[task]
    except:
        return "no_hf"


def map_labels(sample,other_label_id,id_to_keep):
    label = sample["label"]
    sample["label"] = other_label if label in id_to_keep else label
    return sample

def filter_dataset(dataset,task,labels_dict):

    label2id = {i:k for k,i in labels_dict}

    if task == "speech_commands_v2":
        labels_to_keep = ['down', 'go', 'silence', 'on', 'stop', 'left', 'no', 'up', 'yes', 'off', 'right']
        id_to_keep = [label2id[item] for item in label2id if item not in labels_to_keep]
        other_label_id = len(labels_to_keep)
        dataset = dataset.map(map_labels,other_label_id,id_to_keep)
        dataset.cast_column("label", datasets.ClassLabel(names=[i for i in range(other_label_id)]))

    return dataset




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
    backbone = torch.load(args.checkpoint,map_location='cpu')
    wts = backbone['state_dict'].copy()
    for key in list(wts.keys()):
        if 'encoder_q' in key:
            print(key)
            print(key.replace('encoder_q.',''))
            wts[key.replace('encoder_q.','')] = wts[key]

    mod_missing_keys,mod_unexpected_keys = model.load_state_dict(wts,strict=False)
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

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss #+ ne_loss
