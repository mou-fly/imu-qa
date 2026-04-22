import os

import cv2
import numpy as np
import pandas as pd
from dask.array import indices
from dask.array.overlap import boundaries

from modules.simple_tokenizer import SimpleTokenizer
from utils.h5 import load_h5
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")


class Dataset_img(Dataset):
    def __init__(self, root_path, data_path, label_path, flag='train', caption_len=20, use_patch=False, patch_len=None, stride=None):

        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.use_patch = use_patch
        self.root_path = root_path
        self.data_path = data_path
        self.label_path = label_path
        self.tokenizer = SimpleTokenizer()
        self.caption_max_len = caption_len
        self.patch_len = patch_len
        self.stride = stride
        self.__read_data__()

    def __read_data__(self):
        data_path = os.path.join(self.root_path, self.data_path)
        label_path = os.path.join(self.root_path, self.label_path)

        # h5_file = load_h5(data_path)
        # data = h5_file['images'][:]  # size: (N,30,120,120,3)

        label = pd.read_csv(label_path)
        total_size = len(label)
        text = np.zeros((total_size, self.caption_max_len))
        mask = np.zeros((total_size, self.caption_max_len))
        labels = label[['caption']].values
        ids = label[['id']].values
        index = 0
        for label in labels:
            label = str(label)[2:-2]
            caption = self.tokenizer.encode_word(label)
            caption.insert(0, 1)
            caption.append(2)
            m = [1] * len(caption)
            while len(caption) < self.caption_max_len:
                caption.append(0)
                m.append(0)
            assert len(caption) == self.caption_max_len
            assert len(m) == self.caption_max_len

            text[index] = np.array(caption)
            mask[index] = np.array(m)
            index += 1

        if self.flag == 'test':
            split_indices = [i for i in range(len(label))]
        else:
            if self.flag == 'train':
                split_indices = [i for i in range(len(label)) if i % 10 != 0]
            else:
                split_indices = [i for i in range(len(label)) if i % 10 == 0]

        self.data = ids[split_indices]
        self.text = text[split_indices]
        self.mask = mask[split_indices]

    def __getitem__(self, index):
        return self.data[index], self.text[index], self.mask[index]

    def __len__(self):
        return len(self.data)


class Dataset_raw(Dataset):
    def __init__(
        self,
        root_path,
        data_path,
        label_path,
        flag='train',
        caption_len=20,
        use_patch=False,
        patch_len=None,
        stride=None,
        normalize=False,
        norm_mode='device_channel',
        norm_eps=1e-6,
        norm_ref_data_path=None,
    ):

        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.use_patch = use_patch
        self.root_path = root_path
        self.data_path = data_path
        self.label_path = label_path
        self.tokenizer = SimpleTokenizer()
        self.caption_max_len = caption_len
        self.patch_len = patch_len
        self.stride = stride

        self.normalize = normalize
        self.norm_mode = norm_mode
        self.norm_eps = norm_eps
        self.norm_ref_data_path = norm_ref_data_path

        self.__read_data__()

    def __read_data__(self):
        data_path = os.path.join(self.root_path, self.data_path)
        label_path = os.path.join(self.root_path, self.label_path)

        h5_file = load_h5(data_path)
        data = h5_file['data'][:]  # size: (N,5,1500,6)

        if self.normalize:
            ref_path = data_path
            if self.norm_ref_data_path is not None:
                ref_path = os.path.join(self.root_path, self.norm_ref_data_path)

            if os.path.normcase(ref_path) == os.path.normcase(data_path):
                ref_data = data
            else:
                ref_h5_file = load_h5(ref_path)
                ref_data = ref_h5_file['data'][:]

            mean, std = self.compute_norm_stats(ref_data, mode=self.norm_mode)
            data = self.apply_norm(data, mean, std, eps=self.norm_eps, mode=self.norm_mode)

        if self.use_patch:
            data = self.create_patches(data, self.patch_len, self.stride)

        label = pd.read_csv(label_path)
        total_size = len(data)
        text = np.zeros((total_size, self.caption_max_len))
        mask = np.zeros((total_size, self.caption_max_len))
        labels = label[['caption']].values
        index = 0
        for label in labels:
            label = str(label)[2:-2]
            caption = self.tokenizer.encode_word(label)
            caption.insert(0, 1)
            caption.append(2)
            m = [1] * len(caption)
            while len(caption) < self.caption_max_len:
                caption.append(0)
                m.append(0)
            assert len(caption) == self.caption_max_len
            assert len(m) == self.caption_max_len

            text[index] = np.array(caption)
            mask[index] = np.array(m)
            index += 1

        if self.flag == 'test':
            split_indices = [i for i in range(len(data))]
        else:
            if self.flag == 'train':
                split_indices = [i for i in range(len(data)) if i % 10 != 0]
            else:
                split_indices = [i for i in range(len(data)) if i % 10 == 0]

        self.data = data[split_indices]
        self.text = text[split_indices]
        self.mask = mask[split_indices]

    def __getitem__(self, index):
        return self.data[index], self.text[index], self.mask[index]

    def __len__(self):
        return len(self.data)

    def create_patches(self, data, patch_len=300, stride=100):
        """
        data: np.Array of shape (N, D, T, F)
              N: number of samples
              D: number of devices per sample
              T: time steps
              F: features per time step
        returns:
            patches: shape (N, D, num_patches, patch_len, F)
        """
        N, D, T, F = data.shape
        num_patches = (T - patch_len) // stride + 1
        patch_data = np.zeros((N, D, num_patches, patch_len, F))

        for i in range(num_patches):
            start = i * stride
            end = start + patch_len
            patch_data[:, :, i, :, :] = data[:, :, start:end, :]

        return patch_data

    @staticmethod
    def compute_norm_stats(data, mode='device_channel'):
        """
        data shape: (N, D, T, F)
        """
        if mode == 'device_channel':
            mean = data.mean(axis=(0, 2))  # (D, F)
            std = data.std(axis=(0, 2))    # (D, F)
        elif mode == 'channel':
            mean = data.mean(axis=(0, 1, 2))  # (F,)
            std = data.std(axis=(0, 1, 2))    # (F,)
        elif mode == 'global':
            mean = data.mean()  # scalar
            std = data.std()    # scalar
        else:
            raise ValueError(f"Unknown norm_mode: {mode}")
        return mean, std

    @staticmethod
    def apply_norm(data, mean, std, eps=1e-6, mode='device_channel'):
        if mode == 'device_channel':
            return (data - mean[None, :, None, :]) / (std[None, :, None, :] + eps)
        if mode == 'channel':
            return (data - mean[None, None, None, :]) / (std[None, None, None, :] + eps)
        if mode == 'global':
            return (data - mean) / (std + eps)
        raise ValueError(f"Unknown norm_mode: {mode}")
