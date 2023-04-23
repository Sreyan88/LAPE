import os
import torch
import librosa
import numpy as np
import pandas as pd
import torch.nn.functional as f
from datasets import load_dataset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.utils import extract_log_mel_spectrogram, extract_window, MelSpectrogramLibrosa, select_columns, filter_dataset

class DownstreamDatasetHF(Dataset):

    def __init__(self, args, config, split, tfms=None):

        self.config = config
        if 'speech_commands' in args.task:
            self.version = 'v0.02' if '2' in args.task else 'v0.01'
            self.task = "_".join(args.task.split("_")[:2]) # To accomodate for version in input task
        else:
            self.task = args.task

        try:
            self.dataset = load_dataset(self.task, self.version, split = split)
        except Exception:
            raise Exception('Error in loading {} split of {} dataset. Please check if the dataset name is correct or the speicific split is available in 🤗'.format(self.task,split))

        if split == 'train':
            self.dataset = self.dataset.shuffle(seed=42)

        self.filtered = False
        if ('speech_commands' in args.task) and (self.version == 'v0.02') and ('35' not in args.task):
            labels_dict = self.get_id2label() #Used to filter labels in the next step
            self.dataset = filter_dataset(self.dataset,self.task,labels_dict) #Filter labels
            self.filtered = True
            
        self.sample_rate = self.config["downstream"]["input"]["sampling_rate"]
        self.duration= self.config["run"]["duration"] * self.sample_rate
        self.labels_dict = self.get_id2label()
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.x, self.y = self.select_columns_dataset(self.task)
        self.tfms = tfms

    def get_id2label(self):
        labels = self.dataset.features["label"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        return id2label

    def __len__(self):
        return self.dataset.shape[0]

    def select_columns_dataset(self, task):
        return select_columns(self.task)

    def __getitem__(self, idx):
        wave_audio = self.dataset[idx][self.x]['array']
        wave_audio = torch.tensor(wave_audio) #convert into torch tensor
        wave_audio = extract_window(wave_audio, duration=self.duration) #extract fixes size length
        uttr_melspec = extract_log_mel_spectrogram(wave_audio, self.to_mel_spec) #convert into logmel
        uttr_melspec=uttr_melspec.unsqueeze(0) # unsqueeze it for input to CNN

        if self.tfms:
            uttr_melspec=self.tfms(uttr_melspec) #if tfms present, normalize it

        label = self.dataset[idx][self.y]

        return uttr_melspec, label #return normalized

class DownstreamDataset(Dataset):
    
    def __init__(self, args, config, split, tfms=None, labels_dict=None):
        self.config = config
        self.task = args.task
        self.split = split
        if self.split == 'train':
            self.dataset= pd.read_csv(args.train_csv)
        elif self.split == 'valid':
            self.dataset= pd.read_csv(args.valid_csv)
        elif self.split == 'test':
            self.dataset= pd.read_csv(args.test_csv)
        self.sample_rate = self.config["downstream"]["input"]["sampling_rate"]
        self.duration= self.config["run"]["duration"]
        self.labels_dict = self.get_label2id() if labels_dict is None else labels_dict
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def get_label2id(self):
        unique_labels = set(self.dataset['label'])
        id2label = {i:k for k,i in enumerate(unique_labels)}

        return id2label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx,:]
        wave_audio,sr = librosa.core.load(row['wav'], sr=self.sample_rate) #load file
        wave_audio = torch.tensor(wave_audio) #convert into ttorch tensor
        wave_audio = extract_window(wave_audio,data_size=self.duration) #extract fixes size length
        uttr_melspec = extract_log_mel_spectrogram(wave_audio, self.to_mel_spec) #convert into logmel
        uttr_melspec=uttr_melspec.unsqueeze(0) #unsqueeze it

        if self.tfms:
            uttr_melspec=self.tfms(uttr_melspec) #if tfms present, normalize it

        label = self.labels_dict[row['label']]

        return uttr_melspec, label #return normalized
