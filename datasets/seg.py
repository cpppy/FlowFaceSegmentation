import os
import pickle
import time
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
import logging
from tqdm import tqdm



class SegDataset(Dataset):
    def __init__(self,
                 data_dir='/data/data/train',
                 format_size=(512, 768)
                 ):
        super(SegDataset, self).__init__()
        self.data_dir = data_dir
        self.format_size = format_size

        self.img_fpath_list, self.mask_fpath_list = self._load_data()
        print('n_train: {}'.format(len(self.img_fpath_list)))

    def _load_data(self):
        img_root_dir = os.path.join(self.data_dir, 'jpg')
        mask_root_dir = os.path.join(self.data_dir, 'mask')
        ch_dirs = [
            ch_dir
            for ch_dir in os.listdir(img_root_dir)
            if os.path.isdir(os.path.join(img_root_dir, ch_dir))
        ]#[0:300]
        img_fpath_list = []
        mask_fpath_list = []
        n_chdir = len(ch_dirs)
        print('n_chdir: {}'.format(n_chdir))
        pbar = tqdm(desc='loading', total=n_chdir)
        for ch_dir in ch_dirs:
            img_dir = os.path.join(img_root_dir, ch_dir)
            # print(img_dir)
            mask_dir = os.path.join(mask_root_dir, ch_dir)
            if not os.path.exists(mask_dir):
                print('invalid_identify: {}'.format(ch_dir))
                continue
            for img_fn in os.listdir(img_dir):
                img_fpath = os.path.join(img_dir, img_fn)
                # print(img_fpath)
                # print(os.listdir(mask_dir))
                mask_fpath = os.path.join(mask_dir, img_fn.replace('jpg', 'png'))
                if os.path.exists(mask_fpath):
                    img_fpath_list.append(img_fpath)
                    mask_fpath_list.append(mask_fpath)
            pbar.update(1)
        pbar.close()
        return img_fpath_list, mask_fpath_list

    def __len__(self):
        return len(self.img_fpath_list)

    def _normalize(self, img_cv2):
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        img_data = np.asarray(img_cv2, dtype=np.float32)
        img_data = img_data - mean
        img_data = img_data / std
        img_data = img_data.astype(np.float32)
        return img_data

    def __getitem__(self, idx):
        img_fpath = self.img_fpath_list[idx]
        mask_fpath = self.mask_fpath_list[idx]
        img_cv2 = cv2.imread(img_fpath)
        mask_cv2 = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE)
        # resize
        img_cv2 = cv2.resize(img_cv2, (self.format_size[1], self.format_size[0]))
        mask_cv2 = cv2.resize(mask_cv2, (self.format_size[1], self.format_size[0]))
        # normalize
        img_data = self._normalize(img_cv2)
        img_data = np.transpose(img_data, axes=[2, 0, 1])
        mask_data = np.asarray(mask_cv2, dtype=np.long)
        return img_data, mask_data



if __name__=='__main__':

    seg_dataset = SegDataset()
    for i in range(3):
        sample = seg_dataset.__getitem__(i)
        img, mask = sample
        print(mask.shape)
        print(np.min(mask), np.max(mask))
        print(mask)
