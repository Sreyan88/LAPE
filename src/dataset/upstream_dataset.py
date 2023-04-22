import torch
import librosa
import numpy as np
import pandas as pd
import torch.utils.data as data
import torch.nn.functional as f
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

from src.utils import extract_log_mel_spectrogram, extract_window,\
extract_window, MelSpectrogramLibrosa


AUDIO_SR = 16000


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## padd
    batch_1 = [torch.Tensor(t) for t,_ in batch]
    batch_1 = torch.nn.utils.rnn.pad_sequence(batch_1,batch_first = True)
    batch_1 = batch_1.unsqueeze(1)

    batch_2 = [torch.Tensor(t) for _,t in batch]
    batch_2 = torch.nn.utils.rnn.pad_sequence(batch_2,batch_first = True)
    batch_2 = batch_2.unsqueeze(1)

    return batch_1, batch_2


class BaseDataset(Dataset):

    def __init__(self, config, args, data_csv, tfms):

        self.config = config
        #self.audio_files_list = data_dir_list
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms
        self.length = self.config["pretrain"]["input"]["length_wave"]
        self.norm_status = self.config["pretrain"]["normalization"]
        self.sampling_rate = self.config["pretrain"]["input"]["sampling_rate"]
        self.upstream = args.upstream
        self.data = pd.read_csv(data_csv)

    def __getitem__(self, idx):

        audio_file = self.data["files"][idx]
        if self.upstream == "unfused":
            label = self.data["label"][idx]
        wave,sr = librosa.core.load(audio_file, sr=self.sampling_rate)
        wave = torch.tensor(wave)

        if self.config["pretrain"]["input"]["type"] == "raw_wav":
            waveform = extract_window(wave, data_size=self.length) #extract a window

        if self.config["pretrain"]["normalization"] == "l2":
            waveform = f.normalize(waveform,dim=-1,p=2) #l2 normalize

        log_mel_spec = extract_log_mel_spectrogram(waveform, self.to_mel_spec) #convert to logmelspec

        if self.config["pretrain"]["base_encoder"] == "MAST":
            pass #@Ashish please fill this and with rationales beside each line

        # log_mel_spec = log_mel_spec.T

        # n_frames = log_mel_spec.shape[0]

        # p = 1024 - n_frames
        # if p > 0:
        #     m = torch.nn.ZeroPad2d((0, 0, 0, p))
        #     log_mel_spec = m(log_mel_spec)
        # elif p < 0:
        #     log_mel_spec = log_mel_spec[0:1024, :]


        log_mel_spec = log_mel_spec.unsqueeze(0)
        # log_mel_spec = log_mel_spec.permute(0,2,1) #@Ashish if this is particular to MASt please add if condition like above

        if self.tfms:
            lms = self.tfms(log_mel_spec) #do augmentations
        if self.upstream == "unfused":
            return lms, label
        return lms

    def __len__(self):
        return len(self.data)





class BaselineDataModule(pl.LightningDataModule):

    def __init__(self, config, args, tfms, data_csv='./', batch_size=8, num_workers = 8):
        super().__init__()
        self.config = config
        self.args = args
        self.data_dir_train = data_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformation = tfms
        self.dataset_sizes = {}

    def setup(self, stage = None):

        if stage == 'fit' or stage is None:

            self.train_dataset  = BaseDataset(self.config, self.args, self.data_dir_train, self.transformation)
            self.dataset_sizes['train'] = len(self.train_dataset)
            

    def train_dataloader(self):

        return DataLoader(self.train_dataset, 
                          shuffle = True,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          drop_last =True,
                          pin_memory=True)

