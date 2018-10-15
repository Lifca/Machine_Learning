# 第九章：启动与运行 Tensorflow

*TensorFlow* 是一个用于数值计算的强大的开源软件库，尤其适用于大规模机器学习的微调。它的基本原理很简单：先用 Python 定义要执行的计算图（例如图 9-1），之后 TensorFlow 会使用优化的 C++ 代码高效运行该图。

![1](./images/chap09/9-1.png)

最重要的是，可以将图分为多块，在多个 CPU 或 GPU 上并行运行（见图 9-2 ）。 TensorFlow 也支持分布式计算，所以你可以通过在数百台服务器上分割计算，在合理的时间内用庞大的数据集训练大型神经网络（见第十二章）。 Tensorflow 能在由数十亿实例组成、每个实例有数百万特征的训练集上训练一个带有数百万参数的网络。这没什么好惊讶的，因为 Tensorflow 是由 Google	Brain 团队开发的，它支持许多谷歌的大规模服务器，比如 Google Cloud Speech， Google Photos 和 Google	Search 。

![2](./images/chap09/9-2.png)

当　Tensorflow 在 2015 年 11 月开源时，在深度学习上已经有许多流行的开源库了（表 9-1 列举了一些），公平地说，大部分 Tensorflow 的功能已经存在于其他一些库中。尽管如此， Tensorflow 清晰的设计、可扩展性、灵活性和优秀的文档（更不用说 Google 的名号）飞速让它荣登榜首。简而言之， Tensorflow 被设计成灵活的、可扩展的、生产就绪的，现有的三种框架中只讨论到了两种。
