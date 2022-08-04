# -*- coding: utf-8 -*-
# @Time : 2022/6/4 | 10:08
# @Author : YangCheng
# @Email : yangchengyjs@163.com
# @File : case_study.py
# Software: PyCharm
'''

GATMFCDA的case study 设计

选取所有已知的circRNA-disease关联作为正样本和随机选取与正样本等量的负样本做为训练集

保存得分矩阵

'''

import argparse
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from GATCL import GATCNNMF
# from GATMF_mean_att import GATCNNMF
from utils import build_heterograph, sort_matrix, GKL
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, average_precision_score, \
    precision_recall_curve, auc

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-7,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of representations')

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 读取邻接矩阵,625*93,associations:674
    A = np.loadtxt('./data/associationMatrix_625_93.csv', delimiter=',')
    circSimi = np.loadtxt('./data/Integrated_sqe_fun_circRNA_similarity_625.csv', delimiter=',')
    disSimi = np.loadtxt('./data/Integrated_gip_DO_disease_similarity_93.csv', delimiter=',')

    # circ_dis_numpy->tensor
    circSimi_mat = torch.from_numpy(circSimi).to(torch.float32)
    disSimi_mat = torch.from_numpy(disSimi).to(torch.float32)

    circrna_disease_matrix = np.copy(A)
    rna_numbers = circrna_disease_matrix.shape[0]
    dis_number = circrna_disease_matrix.shape[1]

    # # 寻找正样本的索引
    # positive_index_tuple = np.where(circrna_disease_matrix == 1)
    # positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))
    # # 随机打乱
    # random.shuffle(positive_index_list)
    # # 将正样本分为5个数量相等的部分
    # positive_split = math.ceil(len(positive_index_list) / 5)

    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    count = 0

    # 5-fold
    print('starting fivefold cross validation..................')

    new_circrna_disease_matrix = circrna_disease_matrix.copy()

    # relmatrix_to_tensor
    new_circrna_disease_matrix_tensor = torch.from_numpy(new_circrna_disease_matrix).to(device)

    # 标记其余四个没置0的正样本集，“0”表示负样本，“1”表示测试的正样本，“2”表示训练的正样本
    # roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix

    # 异构邻接矩阵
    g = build_heterograph(new_circrna_disease_matrix, circSimi, disSimi).to(device)

    circSimi_mat = circSimi_mat.to(device)
    disSimi_mat = disSimi_mat.to(device)

    model = GATCNNMF(625, 93, 128, 8, 0.1, 0.3, 2778, 1).to(device)
    # 声明参数优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # steps = []
    # loss_value = []

    # 模型训练
    model.train()
    for epoch in range(args.epochs):
        train_predict_result, train_lable = model(g, circSimi_mat, disSimi_mat, new_circrna_disease_matrix_tensor,
                                                  train_model=True)
        loss = F.binary_cross_entropy(train_predict_result, train_lable)
        # steps.append(epoch)
        # loss_value.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | train Loss: %.4f' % (epoch + 1, loss.item()))
    print('到这里啦')
    # 模型评估
    model.eval()
    with torch.no_grad():
        test_predict_result, test_lable = model(g, circSimi_mat, disSimi_mat, new_circrna_disease_matrix_tensor,
                                                train_model=False)
    fpr, tpr, threshold = roc_curve(test_lable, test_predict_result)
    print('AUC value ：', np.trapz(tpr, fpr))
    prediction_matrix = np.zeros(circrna_disease_matrix.shape)
    for num in range(test_predict_result.size()[0]):
        row_num = num // dis_number
        col_num = num % dis_number
        prediction_matrix[row_num, col_num] = test_predict_result[num, 0]

    zero_matrix = np.zeros(prediction_matrix.shape).astype('int64')
    prediction_matrix_temp = prediction_matrix.copy()
    prediction_matrix_temp = prediction_matrix_temp + zero_matrix
    print('saving')
    np.savetxt('predicted_matrix.csv', prediction_matrix_temp, delimiter=',')
    print('saved')
