# -*- coding: utf-8 -*-
# @Time : 2022/5/23 | 20:09
# @Author : YangCheng
# @Email : yangchengyjs@163.com
# @File : GATCL.py
# Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT_layer_v2 import GATv2Conv
from utils import train_features_choose, test_features_choose, build_heterograph

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, embedding_size, drop_rate):
        super(MLP, self).__init__()
        self.embedding_size = embedding_size
        self.drop_rate = drop_rate

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.mlp_prediction = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 2, self.embedding_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 4, self.embedding_size // 6),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 6, 1, bias=False),
            nn.Sigmoid()
        ).to(device)
        self.mlp_prediction.apply(init_weights)

    def forward(self, rd_features_embedding):
        predict_result = self.mlp_prediction(rd_features_embedding)
        return predict_result


class GATCNNMF(nn.Module):
    def __init__(self, in_circfeat_size, in_disfeat_size, outfeature_size, heads, drop_rate, negative_slope,
                 features_embedding_size, negative_times):
        super(GATCNNMF, self).__init__()
        self.in_circfeat_size = in_circfeat_size
        self.in_disfeat_size = in_disfeat_size
        self.outfeature_size = outfeature_size
        self.heads = heads
        self.drop_rate = drop_rate
        self.negative_slope = negative_slope
        self.features_embedding_size = features_embedding_size
        self.negative_times = negative_times

        # 图注意层（多头）
        self.att_layer = GATv2Conv(self.outfeature_size, self.outfeature_size, self.heads, self.drop_rate,
                                   self.drop_rate, self.negative_slope)

        # 定义投影算子
        self.W_rna = nn.Parameter(torch.zeros(size=(self.in_circfeat_size, self.outfeature_size)))
        self.W_dis = nn.Parameter(torch.zeros(size=(self.in_disfeat_size, self.outfeature_size)))
        # 初始化投影算子
        nn.init.xavier_uniform_(self.W_rna.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_dis.data, gain=1.414)

        # 定义卷积层的权重初始化函数
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        # 二维卷积层搭建
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 1), padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 4), padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.cnn_layer16 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 16), padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.cnn_layer32 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 32), padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)
        # 初始化
        self.cnn_layer1.apply(init_weights)
        self.cnn_layer4.apply(init_weights)
        self.cnn_layer16.apply(init_weights)
        self.cnn_layer32.apply(init_weights)

        # MLP
        self.mlp_prediction = MLP(self.features_embedding_size, self.drop_rate)

    def forward(self, graph, circ_feature_tensor, dis_feature_tensor, rel_matrix, train_model):

        circ_circ_f = circ_feature_tensor.mm(self.W_rna)
        dis_dis_f = dis_feature_tensor.mm(self.W_dis)

        N = circ_circ_f.size()[0] + dis_dis_f.size()[0]  # 异构网络的节点个数,num_circ+num_dis

        # 异构网络节点的特征表达矩阵
        h_c_d_feature = torch.cat((circ_circ_f, dis_dis_f), dim=0)

        # 特征聚合
        res = self.att_layer(graph, h_c_d_feature)  # size:[nodes,heads,outfeature_size]
        x = res.view(N, 1, self.heads, -1)

        cnn_embedding1 = self.cnn_layer1(x).view(N, -1)
        cnn_embedding4 = self.cnn_layer4(x).view(N, -1)
        cnn_embedding16 = self.cnn_layer16(x).view(N, -1)
        cnn_embedding32 = self.cnn_layer32(x).view(N, -1)

        cnn_outputs = torch.cat([cnn_embedding1, cnn_embedding4, cnn_embedding16, cnn_embedding32], dim=1)
        print('features_embedding_size:', cnn_outputs.size()[1])

        if train_model:
            train_features_inputs, train_lable = train_features_choose(rel_matrix, cnn_outputs, self.negative_times)
            train_mlp_result = self.mlp_prediction(train_features_inputs)
            return train_mlp_result, train_lable
        else:
            test_features_inputs, test_lable = test_features_choose(rel_matrix, cnn_outputs)
            test_mlp_result = self.mlp_prediction(test_features_inputs)
            return test_mlp_result, test_lable


# import numpy as np

# if __name__ == '__main__':
#     c_d = np.loadtxt('./data/associationMatrix_625_93.csv', delimiter=',')
#     c_c_sim = np.loadtxt('./data/Integrated_sqe_fun_circRNA_similarity_625.csv', delimiter=',')
#     d_d_sim = np.loadtxt('./data/Integrated_gip_DO_disease_similarity_93.csv', delimiter=',')

#     graph = build_heterograph(c_d, c_c_sim, d_d_sim)

#     c_c_tensor = torch.from_numpy(c_c_sim).to(torch.float32)
#     d_d_tensor = torch.from_numpy(d_d_sim).to(torch.float32)
#     c_d_tensor = torch.from_numpy(c_d).to(torch.float32)

#     model = GATCNNMF(625, 93, 128, 4, 0.1, 0.3, 2778, 2)
#     out = model(graph, c_c_tensor, d_d_tensor, c_d_tensor, True)
#     print(out)
