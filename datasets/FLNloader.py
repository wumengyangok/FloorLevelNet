from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import glob

class FLNDataset(Dataset):

    def __init__(self, csv_file, phase, n_class=13, crop=False, flip_rate=0.):

        self.train_data = glob.glob('./train_data/*.png')
        #print(len(self.train_data))
        self.n_class   = n_class
        self.flip_rate = flip_rate
        self.crop      = crop
        self.no_label = False
        self.val = False
        if phase == 'train': 
            self.train_data = glob.glob('./train_data/*[0-9].png')
            print(len(self.train_data))
        if phase == 'val':
            self.train_data = glob.glob('./val_data/*[0-9].png')
            self.val = True
        if phase == 'real':
            self.train_data = glob.glob('./real_data/*.jpg')
            self.no_label = True


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        img_name   = self.train_data[idx]
        img        = cv2.imread(img_name)
        print(img_name)
        if self.no_label:
            img = cv2.resize(img, dsize=(320, 320))
            image = np.copy(img)
            img = img[:, :, ::-1]  # switch to BGR
            img = np.transpose(img, (2, 0, 1)) / 255.
            img = torch.from_numpy(img.copy()).float()
            return {'X': img, 'IM': image, 'PA': self.train_data[idx]}
        label_name = self.train_data[idx][:-4] + '_semantic_label.png'
        label_line_name = self.train_data[idx][:-4] + '_multiple_level.png'
        print(label_name)
        label      = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE) 
        label_line = cv2.imread(label_line_name, cv2.IMREAD_GRAYSCALE)
        img        = cv2.resize(img, dsize=(320, 320))
        label      = cv2.resize(label, dsize=(320, 320))
        label_line = cv2.resize(label_line, dsize=(320, 320))
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()
        label_line = torch.from_numpy(label_line.copy()).long()

        # create one-hot encoding
        h, w = label.size()
        line_target = torch.zeros(13, h, w).long()
        for c in range(13):
            line_target[c][label_line == c] = 1

        target = torch.zeros(7, h, w).long()
        for c in range(7):
            target[c][label == c] = 1

        sample = {'PA': self.train_data[idx], 'X': img, 'Y': target, 'l': label, 'YY': line_target, 'll': label_line}
        return sample

