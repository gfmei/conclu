#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/26/2020 4:35 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : modelnet.py
# @Software: PyCharm
import copy
import os
import random
import sys

import h5py
import torch.utils.data as data
from distributed.protocol import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from transforms import *

modelnet10_label = np.array([2, 3, 9, 13, 15, 23, 24, 31, 34, 36]) - 1


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def sampling_subset(points, n_points):
    idx = torch.randperm(n_points)
    return points[idx]


def knn_trans(points, knn=16, metric="euclidean"):
    """
    Args:
      :param knn: default=16
      :param points: Nx3
      :param metric: distance type
    """
    assert (knn > 0)
    kdt = KDTree(points, leaf_size=30, metric=metric)
    # nbs[0]:NN distance,N*17. nbs[1]:NN index,N*17
    dist, idx = kdt.query(points, k=knn + 1, return_distance=True)
    trans_pts = np.take(points, idx[:, 1: knn], axis=0).mean(1)
    return trans_pts


def _load_data_file(name, subset10):
    f = h5py.File(name)
    data = f['data'][:]
    label = f['label'][:].astype(int)
    if subset10:
        label_list = modelnet10_label.tolist()
        valid_list = []
        for i in range(len(label)):
            if label[i] in label_list:
                valid_list.append(i)
                idx = label_list.index(label[i])
                label[i] = idx
        valid_list = np.array(valid_list)
        data = data[valid_list]
        label = label[valid_list]
    return data, label


class ShapeNet(data.Dataset):
    def __init__(self, root, train=True, unsup=True, use_normal=False, is_crop=True,
                 n_point=1024, aug='jitter', angle=0.20, p_keep=None):
        super().__init__()
        if p_keep is None:
            p_keep = [0.80, 0.80]
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

        if train:
            self.points = np.load(root + 'shapenet_2048_train_points.npy')
            self.labels = np.load(root + 'shapenet_2048_train_label.npy')
        else:
            self.points = np.load(root + 'shapenet_2048_test_points.npy')
            self.labels = np.load(root + 'shapenet_2048_test_label.npy')

        if not use_normal:
            self.points = self.points[:, :, :3]

        print('Successfully load ShapeNet with', self.points.shape[0], 'instances')

    def __getitem__(self, index):
        raw_num = self.points.shape[1]
        pt_idxs = np.arange(0, raw_num)
        if self.train:
            np.random.shuffle(pt_idxs)

        point_set = self.points[index, pt_idxs].copy()
        aug_set = None
        if self.unsup:
            if self.is_crop:
                sample = {'src': point_set, 'ref': point_set}
                pts_dict = self.crop(sample)
                point_set = pts_dict['src']
                aug_set = pts_dict['ref']
            else:
                aug_set = copy.deepcopy(point_set)
            if self.n_points != aug_set.shape[0]:
                aug_set = farthest_point_sample(aug_set, self.n_points)
            # np.random.shuffle(pt_idxs)
            aug_set[:, 0:3] = self.aug(aug_set[:, 0:3])
            aug_set = torch.from_numpy(aug_set)
            aug_set[:, 0:3] = pc_normalize(aug_set[:, 0:3])
        if self.n_points != raw_num:
            point_set = farthest_point_sample(point_set, self.n_points)
        if self.train:
            point_set[:, 0:3] = self.aug(point_set[:, 0:3])
        point_set = torch.from_numpy(point_set)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if self.unsup:
            return point_set, aug_set, self.labels[index]
        else:
            return point_set, self.labels[index]

    def __len__(self):
        return self.points.shape[0]


class ModelNet(data.Dataset):
    def __init__(self, root, train=True, unsup=True, subset10=False, use_normal=False,
                 is_crop=True, n_point=1024, aug='jitter', angle=0.0, p_keep=None):
        super(ModelNet, self).__init__()
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

        if subset10:
            if train:
                self.points = np.load(root + 'ModelNet10_normal_2048_train_points.npy')
                self.labels = np.load(root + 'ModelNet10_normal_2048_train_label.npy')
            else:
                self.points = np.load(root + 'ModelNet10_normal_2048_test_points.npy')
                self.labels = np.load(root + 'ModelNet10_normal_2048_test_label.npy')
        else:
            if train:
                self.points = np.load(root + 'ModelNet40_normal_2048_train_points.npy')
                self.labels = np.load(root + 'ModelNet40_normal_2048_train_label.npy')
            else:
                self.points = np.load(root + 'ModelNet40_normal_2048_test_points.npy')
                self.labels = np.load(root + 'ModelNet40_normal_2048_test_label.npy')

        if not use_normal:
            self.points = self.points[:, :, :3]
        if not subset10:
            print('Successfully load ModelNet40 with', self.points.shape[0], 'instances')
        else:
            print('Successfully load ModelNet10 with', self.points.shape[0], 'instances')

        self.num = self.points.shape[0]

    def __getitem__(self, index):
        raw_num = self.points.shape[1]
        pt_idxs = np.arange(0, raw_num)
        if self.train:
            np.random.shuffle(pt_idxs)

        point_set = self.points[index, pt_idxs].copy()
        # point_set[:, 0:3] = pc_normalize_np(point_set[:, 0:3])
        aug_set = None

        if self.unsup:
            if self.is_crop:
                sample = {'src': point_set, 'ref': point_set}
                pts_dict = self.crop(sample)
                point_set = pts_dict['src']
                aug_set = pts_dict['ref']
            else:
                aug_set = copy.deepcopy(point_set)
            if self.n_points < aug_set.shape[0]:
                aug_set = farthest_point_sample(aug_set, self.n_points)
            # np.random.shuffle(pt_idxs)
            aug_set[:, 0:3] = self.aug(aug_set[:, 0:3])
            aug_set = torch.from_numpy(aug_set)
            aug_set[:, 0:3] = pc_normalize(aug_set[:, 0:3])
        if self.n_points < raw_num:
            point_set = farthest_point_sample(point_set, self.n_points)
        if self.train:
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
    # src = np.random.random((100, 3))
    # s = knn_trans(src, p=6, metric="euclidean")
    # d_path = '/data/gmei/data/modelnet40_normal_resampled/dataset/'
    # data = ModelNet(d_path, angle=12)
    d_path = '/data/gmei/data/shapenetcore_partanno_segmentation_benchmark_v0/dataset/'
    data = ShapeNet(d_path, angle=12)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=6, shuffle=True)
    count = 0
    for point, points2, label in DataLoader:
        count += 1
        print(label.shape)
        print(points2.shape)
        if count > 10:
            break
