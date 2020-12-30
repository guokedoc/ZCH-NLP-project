import numpy as np
# import networkx as nx

from basic.util import exclusive_combine


class MyGraph(object):
    def __init__(self, path, edgelist=True):
        self.neighbor_dict = {}  # 邻居节点字典：将每个节点视作key，把它的邻居节点赋给value
        self.edges_list = []
        if edgelist:
            fin = open(path, 'r')
            for l in fin.readlines():
                self.edges_list.append(list(map(int, l.strip().split())))  # edges_list是去掉空格、换行的数值型列表
                e = l.split()  # e是去掉换行的str型列表
                i, j = int(e[0]), int(e[1])  # 将相邻的两个节点id给到i, j
                self.update_edge(i, j)  # 更新邻居节点字典
                self.update_edge(j, i)
            fin.close()

        for key in self.neighbor_dict.keys():
            self.neighbor_dict[key] = list(self.neighbor_dict[key])  # 将字典的value生成list给到每个key中，也就是节点的邻居信息是list

        self.node_list = list(self.neighbor_dict.keys())  # 将所有节点id存为list
        self.node_list.sort()  # 排序后所有节点id的int型
        self.node_num = len(self.node_list)

    def update_edge(self, i, j):
        if i in self.neighbor_dict:
            self.neighbor_dict[i].add(j)
        else:
            self.neighbor_dict[i] = {j}

        if j in self.neighbor_dict:
            self.neighbor_dict[j].add(i)
        else:
            self.neighbor_dict[j] = {i}

    def get_batches(self, epoch_n, batch_size):
        # np.random.seed(1)
        if epoch_n != 99:
            np.random.shuffle(self.node_list)
        num_batches = self.node_num // batch_size
        batch_list = []
        # 生成batch_list:num_batches行、batch_size列的二维列表
        for n in range(num_batches):
            batch_list.append(self.node_list[n * batch_size: (n + 1) * batch_size])

        if self.node_num > num_batches * batch_size:
            batch_list.append(self.node_list[num_batches * batch_size:])

        self.node_list.sort()
        return batch_list

    def get_neighbors(self, in_list):
        neighbors = [self.neighbor_dict[i] for i in in_list]
        return exclusive_combine(neighbors)

    def diffuse(self, step, nodes):
        cur_list = nodes
        scale_list = [cur_list]
        for s in range(step):
            neighbors = self.get_neighbors(cur_list)
            cur_list = exclusive_combine([cur_list, neighbors])
            scale_list.append(cur_list)
        return scale_list  # From now to the past

    def statistic(self):
        neigh_num = []
        for n in self.node_list:
            neigh_num.append(len(self.neighbor_dict[n]))

        return np.max(neigh_num), np.min(neigh_num), np.mean(neigh_num)


if __name__ == '__main__':
    path = '/home/rnn/zch/data/Citeseer/edges.txt'
    graph = MyGraph(path)

    print(graph.statistic())
