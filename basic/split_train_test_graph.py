import random
import argparse
import os
import numpy as np
# import pandas

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', default=r"C:\Users\lenovo\Documents\GitHub\data\Citeseer")
# parser.add_argument('--gpu', '-g')
args = parser.parse_args()


ratios = ["0.15", "0.25", "0.35", "0.45", "0.55", "0.65", "0.75", "0.85", "0.95"]

number = 0
ratio = ratios[number]
os.makedirs('%s/%s/' %(args.dataset, ratio), exist_ok=True)
f = open('%s/edges.txt' % (args.dataset), 'rb')
edges = [i for i in f]
# 训练集边数据selected
selected = int(len(edges) * float(ratio))
# 随机采样selected长度的边数据
selected = random.sample(edges, selected)
remain = [i for i in edges if i not in selected]

'''写train和test文件'''
# 'wb'格式为以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件
fw1 = open('%s/%s/train_graph_before.txt' % (args.dataset, ratio), 'wb')
fw2 = open('%s/%s/test_graph.txt' % (args.dataset, ratio), 'wb')
for i in selected:
    fw1.write(i)
for i in remain:
    fw2.write(i)

'''重排train的数据'''
s = []
for select in selected:
    s.append(list(map(int, select.strip().split())))
s0 = s
s = [i2 for i1 in s for i2 in i1]
# 去重
s1 = []
for i in s:
    if i not in s1:
        s1.append(i)
s1.sort()  # 排好序的节点id
max_id = len(s1)-1
new_s1 = []
for i in range(max_id):
    new_s1.append(i)
# 生成reference字典
refer_dict = dict(zip(s1, new_s1))
new_line = []
new_train = []
for line in s0:
    for i in line:
        if i in refer_dict.keys():
            i = refer_dict[i]
        new_line.append(i)
    new_train.append(new_line)
    new_line = []
fw_new_train = open('%s/%s/train_graph_after.txt' % (args.dataset, ratio), 'wb')
for i in new_train:
    for j in i:
        j = str(j) + ' '
        j = j.encode()
        fw_new_train.write(j)
    n = '\n'
    n = n.encode()
    fw_new_train.write(n)

'''划分content'''
f_con = open('%s/title.txt' % (args.dataset), 'r', encoding='utf-8')
content = [line for line in f_con]
content_train = []
for i in s1:
    content_train.append(content[i])
fw3 = open('%s/%s/content_train.txt' % (args.dataset, ratio), 'wb')
for i in content_train:
    i = i.encode()
    fw3.write(i)

'''生成新的voc'''
voc0 = []
for line in content_train:
    line = line.strip().split()
    for i in line:
        voc0.append(i)
# 去重
voc = []
for i in voc0:
    if i not in voc:
        voc.append(i)
voc.append('<eos>')
voc = sorted(voc)
punc = ['#', '#NUM#', '$', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<eos>', '=', '?', '[', ']', '_', '\\']
voc_max_len = []
for i in voc:
    if i not in punc:
        voc_max_len.append(i)
max_doc = 'a'
for i in voc_max_len:
    if len(max_doc) < len(i):
        max_doc = i
max_doc_len = len(max_doc)
print('max_doc_len:', max_doc_len)
fw4 = open('%s/%s/voc_new.txt' % (args.dataset, ratio), 'wb')
for i in voc:
    i = i.encode()
    fw4.write(i)
    n = '\n'
    n = n.encode()
    fw4.write(n)
