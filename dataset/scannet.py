#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/29/2023 4:48 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : scannet.py
# @Software: PyCharm
import copy
import os
import sys

import h5py
import numpy as np
import torch
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from transforms import RandomCrop, Jitter, KNNJitter, RandomRotation, pc_normalize


class ScanNetCls(data.Dataset):

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

        def load_h5(name_list):
            out_data = []
            out_label = []
            for name in name_list:
                h5 = h5py.File(root + name, 'r')
                points_train = np.array(h5['data']).astype(np.float32)[:, :, :3]
                labels_train = np.array(h5['label']).astype(int)
                out_data.append(points_train)
                out_label.append(labels_train)
                h5.close()
            points = np.concatenate(out_data, 0)
            labels = np.concatenate(out_label, 0)
            return points, labels

        train_list = [name.strip() for name in open(root + 'train_files.txt', 'r').readlines()]
        test_list = [name.strip() for name in open(root + 'test_files.txt', 'r').readlines()]

        if train:
            points_train, labels_train = load_h5(train_list)
            self.points = points_train
            self.labels = labels_train
        else:
            points_train, labels_train = load_h5(test_list)
            self.points = points_train
            self.labels = labels_train

        print('Successfully load ScanNet with', self.points.shape, 'instances')

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
    d_path = '/home/deep/work/data/scannet/'
    data = ScanNetCls(d_path, knn=4, unsup=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=6, shuffle=True)
    count = 0
    for point, ids, label in DataLoader:
        count += 1
        print(point.shape)
        print(label.shape)
        print(ids[0].shape)
        if count > 10:
            break
