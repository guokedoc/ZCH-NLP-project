# coding:utf-8
# nk iiplab可视化实验代码 2019.5.1

import numpy as np
from sklearn.manifold import TSNE
from time import time
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

sns.set(style='white', color_codes=True)
# plt.rcParams['font.family'] = ['sans-serif']  # 设置字体样式
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文

import argparse  # 用来处理命令行参数

parser = argparse.ArgumentParser()  # 创建一个解析对象。ArgumentParser对象保存了所有必要的信息，用以将命令行参数解析为相应的python数据类型
parser.add_argument('--dataset', '-d', default='Citeseer')  # 向该对象中添加关注的命令行参数和选项
parser.add_argument('--algorithm', '-a', default='ours')
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
args = parser.parse_args()  # 进行解析：parse_args()方法将命令行参数字符串转换为相应对象并赋值给Namespace对象的相应属性，默认返回一个Namespace对象


class visualize(object):
	"""
	使用tsne对embedding降维，之后进行可视化任务。可能出现embedding数量与标签数量不匹配的情况：
	1.有的对象有标签但无表示；2.有的对象有表示但无标签（例如半监督环境下）3.有的对象无表示无标签
	这些情况我们选择将这些对象剔除
	"""
	def __init__(self, embedding_path, label_path, class_num, class_name, fig_path=None, title=None, header=False):
		"""
		embedding_path: 要进行可视化任务的embedding的路径与文件名；
		label_path: 每个embedding对应的标签;
		class_num: 总共有多少个类;
		class_name: dict, key---数字标签(str) value---实际标签；
		fig_path: 图片保存的路径，如果不给定，默认保存在当前目录下;
		title: 图片标题
		header: 要可视化的embedding是带标头的吗，默认不是"""
		self.embedding_path = embedding_path
		self.label_path = label_path
		self.class_num = class_num
		self.class_name = class_name
		self.header = header
		if fig_path is None:
			t0 = time()
			self.fig_path = './figure_%s_%s' % (args.dataset, args.algorithm)
		else:
			self.fig_path = fig_path
		self.title = title

	def _get_embedding(self, header):
		r_f = open(self.embedding_path, "r")
		if not header:  # 无标头，直接读
			embedding = [list(map(float, embed.strip().split(' '))) 
				if embed != '\n' else embed.strip() for embed in r_f.readlines()]
		
		else:   # 有标头，需要处理一下
			embedding = list()
			node2vec = dict()
			for line in r_f.readlines()[1:]:
				node2vec[int(line.strip().split(" ")[0])] = list(map(float, line.strip().split(" ")[1:]))

			print(len(node2vec))

			for line in range(len(node2vec)):
				embedding.append(node2vec[line])
				
		return embedding

	def _format_label(self):
		"""有的label格式不符合要求，用此函数修改格式，使每一行仅保留标签名（实际标签名）"""
		r_f = open(self.label_path, "r")
		labels = [l.strip().split(" ") if l != '\n' else l.strip() for l in r_f.readlines()]
		try:
			labels_ = [self.class_name[label[1]] for label in labels]
		except ValueError:
			pass
		return labels_

	def _get_label(self):
		label = self._format_label()
		return label

	def _get_tsne(self, embeddings):
		tsne_embed = TSNE(n_components=2, random_state=0).fit_transform(embeddings)  # np.array
		np.save("./tsne_embed_%s_%s" % (args.dataset, args.algorithm), tsne_embed)
		return tsne_embed

	def plot_embedding(self):
		embedding = self._get_embedding(header=self.header)
		label = self._get_label()
		embeddings, labels = [], []
		print(len(embedding))
		print(len(label))
		# assert是断言，为False时直接报错，不会让程序出错崩溃
		assert len(embedding) == len(label)

		for i in range(len(embedding)):   # 遍历整个embedding
			if (not isinstance(embedding[i], str)) and np.sum(np.square(np.array(embedding[i]))) != 0.0 and label[i] != '':
				# 保证每个点都有embedding和label
				embeddings.append(embedding[i])
				labels.append(label[i])
		if os.path.exists("./tsne_embed_%s_%s.npy" % (args.dataset, args.algorithm)):
			print("you already have run tsne, so we just reload tsne embedding")
			x_embeddings = np.load("./tsne_embed_%s_%s.npy" % (args.dataset, args.algorithm))
		else:
			print("now we run tsne dimension reduction")
			x_embeddings = self._get_tsne(embeddings)

		embedding = pd.DataFrame(x_embeddings, columns=["x", "y"])
		embedding["label"] = labels

		# estimator = KMeans(n_clusters=7)  # 构造聚类数为n_clusters的聚类器
		# estimator.fit(embedding)  # 聚类
		# label = estimator.labels_ # 获取聚类标签
		# embedding["label"] = label

		plaette = sns.color_palette("husl", self.class_num)  # 自动选择颜色的调色板，有class_num个颜色
		# hue指颜色表示标签
		g = sns.FacetGrid(embedding, hue='label', palette=plaette, legend_out=True, size=5)  # hue分类显示
		g = g.map(plt.scatter, 'x', 'y', s=10, edgecolor='white', linewidths=0)   
		
		plt.xlabel('The first dimension', color='gray')
		plt.ylabel('The second dimension', color='gray')
		g.add_legend(title="CLASS")
		# leg = plt.legend(loc='right', title="CLASS", frameon=False, fontsize=13)  # 显示图例
		# for text in leg.get_texts():
		# 	plt.setp(text, color='gray')
		# plt.savefig(self.fig_path + ".eps", bbox_inches='tight')
		# plt.savefig(self.fig_path + ".png", bbox_inches='tight')
		# plt.savefig(self.fig_path + ".svg", bbox_inches='tight')
		plt.show()

if __name__ == "__main__":
	class_name = {
	# Citeseer
	"0" : "0",
	"1" : "1",
	"2" : "2",
	"3" : "3",
	"4" : "4",
	"5" : "5",
	"6" : "6",
	"7" : "7",
	"8" : "8",
	"9" : "9"

	# Cora
	# "0" : "0",
	# "1" : "1",
	# "2" : "2",
	# "3" : "3",
	# "4" : "4",
	# "5" : "5",
	# "6" : "6",
	# "7" : "7"

	# DBLP
	# "0": "0",
	# "1": "1",
	# "2": "2",
	# "3": "3",

	# pubmed
	# "0": "0",
	# "1": "1",
	# "2": "2"
	}
	visualize_task = visualize(embedding_path="C:/Users/62488/Documents/GitHub/data/Citeseer/embedding.txt",
								# embedding_path="C:/Users/62488/Documents/GitHub/data/pubmed/embedding.txt",
								label_path="C:/Users/62488/Documents/GitHub/data/Citeseer/labels.txt",
								class_name=class_name,
								class_num=3,
								header=False)
	visualize_task.plot_embedding()
