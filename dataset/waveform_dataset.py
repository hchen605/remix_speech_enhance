import os

import torch
import librosa
import numpy as np

from torch.utils import data

from util.utils import sample_fixed_length_data_aligned


class Dataset(data.Dataset):
    def __init__(
        self,
        dataset_list,
        limit,
        offset,
        sample_rate,
        is_training,
        sample_length=48000,
        max_length=None,
        do_normalize=True,
    ):
        """
        dataset_list(*.txt):
            <noisy_path> <clean_path>\n
        e.g:
            noisy_1.wav clean_1_1. clean_1_2.wav
            noisy_2.wav clean_2_1. clean_2_2.wav
            ...
            noisy_n.wav clean_n_1. clean_n_2.wav
        """
        super(Dataset, self).__init__()
        self.sample_rate = sample_rate
        self.is_training = is_training

        dataset_list = [line.rstrip("\n") for line in open(dataset_list, "r")]
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.dataset_list = dataset_list
        self.length = len(self.dataset_list)
        self.do_normalize = do_normalize
        self.sample_length = sample_length
        self.max_length = max_length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_path, clean_path_1, clean_path_2 = self.dataset_list[item].split(" ")
        name = os.path.splitext(os.path.basename(noisy_path))[0]


        noisy, _ = librosa.load(noisy_path, sr=self.sample_rate)
        noisy_ori = noisy
        clean_1, _ = librosa.load(clean_path_1, sr=self.sample_rate)
        clean_2, _ = librosa.load(clean_path_2, sr=self.sample_rate)

        if self.do_normalize:
            noisy = self.normalize_wav(noisy)
            clean_1 = self.normalize_wav(clean_1)
            clean_2 = self.normalize_wav(clean_2)

        if self.is_training:
            # The input of model should be fixed-length in the training.
            noisy, clean_1 = sample_fixed_length_data_aligned(
                noisy, clean_1, self.sample_length
            )
            noisy, clean_2 = sample_fixed_length_data_aligned(
                noisy_ori, clean_2, self.sample_length
            )
        elif self.max_length:
            # This is for SaShiMi validation to avoid OOM
            if len(noisy) > self.max_length:
                noisy = noisy[:self.max_length]
                clean_1 = clean_1[:self.max_length]
                clean_2 = clean_2[:self.max_length]

        return noisy, clean_1, clean_2, name

    def normalize_wav(self, wav):
        return wav / np.abs(wav).max()


def collate_fn(batch):
    noisy_list = []
    clean_1_list = []
    clean_2_list = []
    names = []

    for noisy, clean_1, clean_2, name in batch:
        noisy_list.append(torch.tensor(noisy))  # [F, T] => [T, F]
        clean_1_list.append(torch.tensor(clean_1))  # [1, T] => [T, 1]
        clean_2_list.append(torch.tensor(clean_2))  # [1, T] => [T, 1]
        names.append(name)

    noisy_wav = torch.stack(noisy_list, dim=0)
    clean_1_wav = torch.stack(clean_1_list, dim=0)
    clean_2_wav = torch.stack(clean_2_list, dim=0)

    return noisy_wav, clean_1_wav, clean_2_wav, names
