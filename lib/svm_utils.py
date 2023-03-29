#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/23/2020 10:36 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : svm_utils.py
# @Software: PyCharm
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC


def evaluate_svm(train_features, train_labels, test_features, test_labels, md='lsvc'):
    if md == 'sgd':
        clf = SGDClassifier(max_iter=1000, tol=1e-3)
    elif md == 'lsvc':
        clf = LinearSVC()
    else:
        clf = SVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def svm_data(loader, encoder):
    encoder.eval()
    features = list()
    label = list()
    for _, data in enumerate(loader, 0):
        points, target = data[0], data[-1]
        points, target = points.cuda(), target.cuda()
        feature = encoder.backbone(points.transpose(2, 1), True)
        target = target.view(-1)
        features.append(feature.data)
        label.append(target.data)
    features = torch.cat(features, dim=0)
    label = torch.cat(label, dim=0)

    return features, label


def validate(train_loader, test_loader, encoder, best_acc, logger, md='lsvc'):
    # feature extraction
    with torch.no_grad():
        train_features, train_label = svm_data(train_loader, encoder)
        test_features, test_label = svm_data(test_loader, encoder)
    # train svm
    svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(),
                           test_features.data.cpu().numpy(), test_label.data.cpu().numpy(), md)

    if svm_acc > best_acc:
        best_acc = svm_acc

    encoder.train()
    logger.info('Classification results: svm acc=%f,\t best svm acc=%f' % (svm_acc, best_acc))
    print('Classification results: svm acc=', svm_acc, 'best svm acc=', best_acc)
    return svm_acc
