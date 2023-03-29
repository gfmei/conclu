#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/29/2023 4:49 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : scanobject.py
# @Software: PyCharm
import copy
import os
import sys

import h5py
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from transforms import *


class ScanObjectNNCls(data.Dataset):

    def __init__(self, root, train=True, unsup=True, is_crop=True, n_point=1024, aug='jitter', angle=0.20, p_keep=None):
        super().__init__()
        if p_keep is None:
            p_keep = [0.85, 0.85]
        self.crop = RandomCrop(p_keep=p_keep)
        self.is_crop = is_crop
        if aug == 'jitter':
            self.aug = Jitter(sigma=0.001, clip=0.0025)
        elif aug == 'jiknn':
            self.aug = KNNJitter(sigma=0.001, clip=0.0025, knn=4, metric="euclidean")
        else:
            self.aug = RandomRotation(angle)
        self.n_points = n_point
        self.unsup = unsup
        self.train = train

        if self.train:
            h5 = h5py.File(root + 'training_objectdataset.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            h5 = h5py.File(root + 'test_objectdataset.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()

        print('Successfully load ScanObjectNN with', len(self.labels), 'instances')

    def __getitem__(self, index):
        pt_idxs = np.arange(0, self.points.shape[1])
        if self.train:
            np.random.shuffle(pt_idxs)

        point_set = self.points[index, pt_idxs].copy()
        aug_set = None
        pt_idxs = np.arange(0, self.n_points)  # 2048

        if self.unsup:
            if self.is_crop:
                sample = {'src': point_set, 'ref': point_set}
                pts_dict = self.crop(sample)
                point_set = pts_dict['src']
                aug_set = pts_dict['ref'][pt_idxs]
                aug_set = torch.from_numpy(aug_set)
            else:
                aug_set = copy.deepcopy(point_set)
            np.random.shuffle(pt_idxs)
            # aug_set[:, 0:3] = self.aug(aug_set[:, 0:3])
            aug_set[:, 0:3] = pc_normalize(aug_set[:, 0:3])
        point_set = point_set[pt_idxs]
        point_set[:, 0:3] = self.aug(point_set[:, 0:3])
        point_set = torch.from_numpy(point_set)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if self.unsup:
            return point_set, aug_set, self.labels[index]
        else:
            return point_set, self.labels[index]

    def __len__(self):
        return self.points.shape[0]


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    d_path = '/home/deep/work/data/scanobject/main_split_nobg/'
    data = ScanObjectNNCls(d_path, patches=4, self_supervision=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=6, shuffle=True)
    count = 0
    for point, ids, label in DataLoader:
        count += 1
        print(point.shape)
        print(label.shape)
        print(ids[0].shape)
        if count > 10:
            break
