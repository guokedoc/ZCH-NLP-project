# ZCH-NLP-project
Net2Net-NE code and its extension

basic文件夹中包含用于分类的代码classify.py、用于定义图的graph.py、用于切分数据集的split_train_test_graph.py、用于存放各类功能函数的util.py；
data文件夹中包含以上提到的四个数据集的数据，每个数据集文件夹中存放着内容、标签、边、字典和嵌入数据；
model文件夹中存放着Net2Net-NE模型主体结构中的各个组件文件models.py，组件包含内容嵌入组件（ContentCNN）、自我网络编码器（EgoEncoder）和聚合器（MeanAggregator）；
exe文件夹中则存放着主运行文件Net2Net-NE.py和Modify.py，main函数的开头可以选择是否开启半监督学习，以及引入训练的标签节点比率和监督学习权重，同时，可以选择数据集；
Node Visualize文件夹中存放着用来进行节点可视化的visualize.py文件。需要注意的一点是Pubmed数据集因其结构与其他数据集不同，因此运行需要用到Modify.py文件。

运行时只打开Net2Net.py文件或Modify.py文件即可，选择半监督标签节点比率、监督学习权重和数据集，即可迭代训练，训练完成后通过逻辑斯谛回归分类器输出F1-score数值，用于评测运行结果。
运行代码得到的embedding数据可以通过visualize.py文件生成节点可视化的节点图。
