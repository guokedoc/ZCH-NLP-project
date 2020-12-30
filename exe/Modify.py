# coding:utf-8
import time
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import init

sys.path.append("../")
from basic.classify import read_node_label
from basic.graph import MyGraph
from basic.util import node_classification
from model.models import MeanAggregator, EgoEncoder
import warnings

warnings.filterwarnings("ignore")


class Net2Net(nn.Module):    # 定义卷积神经网络
    def __init__(self, global_graph, features, encoder):
        super(Net2Net, self).__init__()
        self.graph = global_graph
        self.node_num = self.graph.node_num
        self.embed_dim = encoder.embed_dim
        self.features = features
        self.encoder = encoder
        self.xent = nn.CrossEntropyLoss()    # 交叉熵损失

        self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim, self.node_num))    # 行数是embed_dim，列数是node_num
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.encoder(nodes)
        scores = embeds.mm(self.weight)    # embeds与weight矩阵相乘
        return scores

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())    # 交叉熵损失函数：第一个参数是input，第二个参数是target

    def evaluate(self, b_list, lab, ratio):
        self.eval()
        hidden = []
        idx = []
        for bat in b_list:
            h = self.encoder(bat)
            hidden.extend(h.detach().cpu().numpy())
            idx.extend(bat)

        f1 = []
        for r in ratio:
            # node_classification是util中函数，使用逻辑斯谛回归方法进行节点分类，得到f1_micro，赋值给f1
            f1.append(node_classification(hidden, np.arange(len(lab)), [lab[i] for i in idx], r))
        return f1, hidden


def features(list_x, nodes):
    batch_x = [list_x[node] for node in nodes]
    content = torch.tensor(batch_x, device=torch.device("cuda:0"), requires_grad=True)
    embed = torch.tanh(content)
    return embed


def main():
    data_dir = '/home/user/zch/data/pubmed/'
    adj_file = 'edges.txt'
    label_file = 'labels.txt'
    con_file = 'content.txt'

    word_emb_dim = 500    # 字嵌入的dimension
    conv_dim = 500    # CNN滤波器维度
    kernel_num = 200    # 200个不同尺寸的filters
    kernel_sizes = [1, 2, 3, 4, 5]    # CNN滤波器的尺寸L，即编码器深度，L=2时最佳
    conv_drop = 0.2    # dropout层进一步应用于CNN学习的内容嵌入，其中dropout概率p＝0.2
    enc_dim = 500    # 编码器的隐含层维度k，随着k值增大，模型性能增强，但当k值达到500时，性能开始下降
    batch_size = 32
    epoch_num = 100
    l_rate = 1e-4
    class_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]    # 节点标记率

    gpu_id = 0
    gpu = torch.device('cuda', gpu_id)

    start = time.time()
    graph = MyGraph(data_dir + adj_file)    # 读取边信息,...

    _, labels = read_node_label(data_dir + label_file)    # 读取标签信息,classify中函数

    f = open(data_dir + con_file, 'rb')
    list_x = []
    for line in f:
        list_x.append(list(map(float, line.strip().split())))

    agg1 = MeanAggregator(lambda nodes: features(list_x, nodes), gpu)
    enc1 = EgoEncoder(lambda nodes: features(list_x, nodes), conv_dim, enc_dim, graph, agg1)

    agg2 = MeanAggregator(lambda nodes: enc1(nodes), gpu)
    enc2 = EgoEncoder(lambda nodes: enc1(nodes), enc1.embed_dim, enc_dim, graph, agg2, base_model=enc1)

    c2n = Net2Net(graph, None, enc2)
    c2n.cuda(gpu)    # 在gpu上训练    gpu = "cuda:0"

    # Adam最优化参数迭代算法：RMSprop和Momentum算法的结合，考虑前一时刻运动，即考虑了变化趋势；对于出现频率不同的方向采取不同的学习率；偏置矫正后迭代十分平稳
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, c2n.parameters()), lr=l_rate)

    # 遍历数据迭代器，然后将输入馈送到网络并进行优化
    for e in range(epoch_num):
        avg_loss = []    # 使损失平滑
        c2n.train()
        batch_list = graph.get_batches(batch_size)
        for batch in batch_list:
            optimizer.zero_grad()    # 初始化梯度缓冲区
            loss = c2n.loss(batch, torch.tensor(batch, dtype=torch.int64, device=gpu))    # 目标与输出间的损失，公式9
            loss.backward()
            optimizer.step()    # 使用Adam算法，更新权重
            avg_loss.append(loss.item())

        # 节点分类结果
        f1_micro, embedding = c2n.evaluate(batch_list, labels, class_ratio)
        minute = np.around((time.time() - start) / 60)
        ls = np.mean(avg_loss)
        print('Epoch:', e, 'loss:', ls, 'mi-F1:', np.around(f1_micro, 3), 'time:', minute, 'mins.')
        if e == 99:
            embedding = np.array(embedding)
            np.savetxt('/home/user/zch/data/pubmed/embedding.txt', embedding)
        avg_loss.clear()




if __name__ == "__main__":
    main()


