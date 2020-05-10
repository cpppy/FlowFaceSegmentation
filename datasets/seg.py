import os
import pickle
import time
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
import logging




class SegDataset(Dataset):
    def __init__(self,
                 data_dir='/data/data/train',
                 format_size=(768, 512)
                 ):
        super(SegDataset, self).__init__()
        self.data_dir = data_dir
        self.format_size = format_size

    def _load_data(self):
        img_root_dir = os.path.join(self.data_dir, 'jpg')
        mask_root_dir = os.path.join(self.data_dir, 'mask')
        ch_dirs = [
            ch_dir
            for ch_dir in os.listdir(img_root_dir)
            if os.path.isdir(os.path.join(img_root_dir, ch_dir))
        ]
        img_fpath_list = []
        mask_fpath_list = []
        for ch_dir in ch_dirs:
            img_dir = os.path.join(img_root_dir, ch_dir)
            mask_dir = os.path.join(mask_root_dir, ch_dir)
            if not os.path.exists(mask_dir):
                continue
            for img_fn in os.listdir(img_dir):



    def __len__(self):
        return len(self.img_fpath_list)

    def __getitem__(self, idx):



    def _load_source_data(self, data_dir):
        img_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        img_fn_list = os.listdir(img_dir)  # [0:1]
        img_fpath_list = []
        label_fpath_list = []
        for img_fn in img_fn_list:
            img_fpath = os.path.join(img_dir, img_fn)
            label_fpath = os.path.join(label_dir, img_fn)
            if not os.path.exists(label_fpath):
                continue
            img_fpath_list.append(img_fpath)
            label_fpath_list.append(label_fpath)
        return img_fpath_list, label_fpath_list

    def prepare_train_img(self, idx):
        img_fpath = self.img_fpath_list[idx]
        label_fpath = self.label_fpath_list[idx]
        img_info = dict(id=idx, filename=img_fpath)
        mask = cv2.imread(label_fpath, cv2.IMREAD_GRAYSCALE)
        # print('mask_shape: {}'.format(mask.shape))
        mask = np.asarray(mask, np.float32) / 255.0
        ann_info = dict(masks=mask)
        img_info.setdefault('width', mask.shape[1])
        img_info.setdefault('height', mask.shape[0])
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.pipeline is not None:
            results = self.pipeline(results)
        return results

    def prepare_test_img(self, idx):
        img_fpath = self.img_fpath_list[idx]
        label_fpath = self.label_fpath_list[idx]
        img_info = dict(id=idx, filename=img_fpath)
        mask = cv2.imread(label_fpath, cv2.IMREAD_GRAYSCALE)
        # print('mask_shape: {}'.format(mask.shape))
        mask = np.asarray(mask, np.float32) / 255.0
        ann_info = dict(masks=mask)
        img_info.setdefault('width', mask.shape[1])
        img_info.setdefault('height', mask.shape[0])
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.pipeline is not None:
            results = self.pipeline(results)
        return results



if __name__=='__main__':

    mask_dataset = MaskDataset()
    for i in range(3):
        sample = mask_dataset.__getitem__(i)
        print(sample)
