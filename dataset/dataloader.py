#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/29/2023 4:45 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : dataloader.py
# @Software: PyCharm
import torch

from modelnet import ShapeNet, ModelNet
from scannet import ScanNetCls
from scanobject import ScanObjectNNCls


def data_loader(args):
    train_svmset, test_svmset = None, None
    if args.datatype == 'modelnet':
        train_dataset = ModelNet(root=args.dataset, n_point=args.num_point, train=True, subset10=args.subset10,
                                 use_normal=args.normal, unsup=args.unsup, angle=args.angle, p_keep=args.crop)
        test_dataset = ModelNet(root=args.dataset, n_point=args.num_point, train=False, subset10=args.subset10,
                                use_normal=args.normal, unsup=False, angle=0.0)
    elif args.datatype == 'scannet':
        train_dataset = ScanNetCls(root=args.dataset, n_point=args.num_point, train=True,
                                   unsup=args.unsup, angle=args.angle)
        test_dataset = ScanNetCls(root=args.dataset, n_point=args.num_point, train=False,
                                  unsup=False, angle=args.angle)

    elif args.datatype == 'scanobject':
        train_dataset = ScanObjectNNCls(root=args.dataset, n_point=args.num_point, train=True,
                                        unsup=args.unsup, angle=args.angle)
        test_dataset = ScanObjectNNCls(root=args.dataset, n_point=args.num_point, train=False,
                                       unsup=args.unsup, angle=args.angle)
    elif args.datatype == 'shapenet':
        train_dataset = ShapeNet(root=args.dataset, n_point=args.num_point, train=True,
                                 unsup=args.unsup, angle=args.angle, use_normal=args.normal)
        test_dataset = ShapeNet(root=args.dataset, n_point=args.num_point, train=False,
                                unsup=args.unsup, angle=args.angle, use_normal=args.normal)
    elif args.datatype == 'bothset':
        train_dataset = ShapeNet(root=args.dataset, n_point=args.num_point, train=True,
                                 unsup=args.unsup, angle=args.angle, use_normal=args.normal)
        test_dataset = ShapeNet(root=args.dataset, n_point=args.num_point, train=False,
                                unsup=args.unsup, angle=args.angle, use_normal=args.normal)
        train_svmset = ModelNet(root=args.svmset, n_point=args.num_point, train=True, subset10=args.subset10,
                                use_normal=args.normal, unsup=args.unsup, angle=args.angle, p_keep=args.crop)
        test_svmset = ModelNet(root=args.svmset, n_point=args.num_point, train=False, subset10=args.subset10,
                               use_normal=args.normal, unsup=False, angle=0.0)
    else:
        raise NotImplementedError

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=args.workers)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                   shuffle=False, num_workers=args.workers)
    if args.datatype == 'bothset':
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                        shuffle=True, num_workers=args.workers)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                       shuffle=False, num_workers=args.workers)
        train_svm_loader = torch.utils.data.DataLoader(train_svmset, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=args.workers)
        test_svm_loader = torch.utils.data.DataLoader(test_svmset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=args.workers)
        return train_data_loader, test_data_loader, train_svm_loader, test_svm_loader
    return train_data_loader, test_data_loader


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
