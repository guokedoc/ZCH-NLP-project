import time
import numpy as np
import torch
from torch import nn
from torch.nn import init
from basic.classify import read_node_label
from basic.graph import MyGraph
from basic.util import read_word_code, node_classification, fetch
from model.models import MeanAggregator, EgoEncoder, ContentCNN
import warnings
import torch.nn.functional as F
import random

warnings.filterwarnings("ignore")


class Net2Net(nn.Module):
    def __init__(self, global_graph, features, encoder, num_class, lam, sem=True):
        super(Net2Net, self).__init__()
        self.graph = global_graph
        self.node_num = self.graph.node_num
        self.embed_dim = encoder.embed_dim
        self.features = features
        self.encoder = encoder
        self.xent = nn.CrossEntropyLoss()
        self.sem = sem

        if sem:   # semi-supervised learning
            # fully connected. The input is embed_dim, output is num_class
            self.fc_1 = nn.Linear(self.embed_dim, num_class)
            self.lam = lam    # to balance the weight between supervised and unsupervised loss

        self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim, self.node_num))
        init.xavier_uniform_(self.weight)

    def loss(self, nodes, nodes_id, nodes_label, mask):
        embeds = self.encoder(nodes)
        # unsupervised loss, reconstruct the graph
        scores = embeds.mm(self.weight)
        unsupervised_loss = self.xent(scores, nodes_id.squeeze())

        if self.sem:
            # supervised loss, map the node embeddings into the label space by using a fc layer
            pred_labels = F.log_softmax(self.fc_1(embeds), dim=-1).index_select(0, mask)
            nodes_label = nodes_label.index_select(0, mask).view(-1)  # choose nodes_label which has a mask
            supervised_loss = F.nll_loss(pred_labels, nodes_label)
            return unsupervised_loss + self.lam * supervised_loss
        else:
            return unsupervised_loss

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
        return f1


def main():
    semi = True
    ratio = 0.5  # node labeled ratio for semi-supervised learning
    lamda = 0.3
    print("Semi-supervised? [{}], labeled nodes ratio: [{}]".format(semi, ratio))

    data_dir = '/home/rnn/zch/data/Citeseer/'
    adj_file = 'edges.txt'
    label_file = 'labels.txt'
    con_file = 'title.txt'
    voca_file = 'voc.txt'
    word_num = 5523
    max_doc_len = 34
    labelclass = 10

    # data_dir = '../../data/DBLP/'
    # adj_file = 'edges.txt'
    # label_file = 'labels.txt'
    # con_file = 'title.txt'
    # voca_file = 'voc.txt'
    # word_num = 8501
    # max_doc_len = 27
    # labelclass = 4

    # data_dir = '../../data/Cora/'
    # adj_file = 'edges.txt'
    # label_file = 'labels.txt'
    # con_file = 'abstract.txt'
    # voca_file = 'voc.txt'
    # word_num = 12619
    # max_doc_len = 100
    # labelclass = 7

    word_emb_dim = 500
    conv_dim = 500
    kernel_num = 200
    kernel_sizes = [1, 2, 3, 4, 5]
    conv_drop = 0.2
    enc_dim = 500
    batch_size = 32
    epoch_num = 100
    l_rate = 1e-4
    class_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]

    gpu_id = 0
    gpu = torch.device('cuda', gpu_id)

    start = time.time()
    graph = MyGraph(data_dir + adj_file)

    _, labels = read_node_label(data_dir + label_file)

    node_content, pad_code = read_word_code(data_dir + con_file, data_dir + voca_file)

    features = ContentCNN(word_num, word_emb_dim, conv_dim, kernel_num, kernel_sizes, conv_drop, gpu)

    agg1 = MeanAggregator(lambda nodes: features(fetch(node_content, nodes, max_doc_len, pad_code)), gpu)
    enc1 = EgoEncoder(lambda nodes: features(fetch(node_content, nodes, max_doc_len, pad_code)), conv_dim, enc_dim,
                      graph, agg1)

    agg2 = MeanAggregator(lambda nodes: enc1(nodes), gpu)
    enc2 = EgoEncoder(lambda nodes: enc1(nodes), enc1.embed_dim, enc_dim, graph, agg2, base_model=enc1)

    c2n = Net2Net(graph, features, enc2, labelclass, lamda, semi)
    param_count = 0
    for param in c2n.parameters():
        param_count += param.view(-1).size()[0]
    print('total number of parameters: %d\n' % param_count)

    c2n.cuda(gpu)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, c2n.parameters()), lr=l_rate)

    for e in range(epoch_num):
        avg_loss = []
        c2n.train()
        batch_list = graph.get_batches(e, batch_size)
        for batch in batch_list:

            optimizer.zero_grad()  # 初始化梯度缓冲区
            label_batch = [labels[i] for i in batch]
            label_mask = [idx for idx in range(len(batch)) if random.uniform(0, 1) < ratio]
            if len(label_mask) == 0:
                label_mask = [0]
            label_batch = torch.LongTensor(label_batch).to(gpu)
            label_mask = torch.LongTensor(label_mask).to(gpu)
            loss = c2n.loss(batch, torch.tensor(batch, dtype=torch.int64, device=gpu), label_batch, label_mask)
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())

        # node classification results
        f1_micro = c2n.evaluate(batch_list, labels, class_ratio)
        minute = np.around((time.time() - start) / 60)
        ls = np.mean(avg_loss)
        print('Epoch:', e, 'loss:', ls, 'mi-F1:', np.around(f1_micro, 3), 'time:', minute, 'mins.')
        avg_loss.clear()


if __name__ == "__main__":
    main()
