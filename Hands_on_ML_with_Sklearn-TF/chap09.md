# 第九章：启动与运行 Tensorflow

*TensorFlow* 是一个用于数值计算的强大的开源软件库，尤其适用于大规模机器学习的微调。它的基本原理很简单：先用 Python 定义要执行的计算图（例如图 9-1），之后 TensorFlow 会使用优化的 C++ 代码高效运行该图。

![1](./images/chap09/9-1.png)

最重要的是，可以将图分为多块，在多个 CPU 或 GPU 上并行运行（见图 9-2 ）。 TensorFlow 也支持分布式计算，所以你可以通过在数百台服务器上分割计算，在合理的时间内用庞大的数据集训练大型神经网络（见第十二章）。 Tensorflow 能在由数十亿实例组成、每个实例有数百万特征的训练集上训练一个带有数百万参数的网络。这没什么好惊讶的，因为 Tensorflow 是由 Google Brain 团队开发的，它支持许多谷歌的大规模服务器，比如 Google Cloud Speech， Google Photos 和 Google	Search 。

![2](./images/chap09/9-2.png)

当 Tensorflow 在 2015 年 11 月开源时，在深度学习上已经有许多流行的开源库了（表 9-1 列举了一些），公平地说，大部分 Tensorflow 的功能已经存在于其他一些库中。尽管如此， Tensorflow 清晰的设计、可扩展性、灵活性和优秀的文档（更不用说 Google 的名声了）飞速让它荣登榜首。简而言之， Tensorflow 被设计成灵活的、可扩展的、生产就绪的，现有的三种框架中只讨论到了两种。下面是一些 Tensorflow 的亮点：

- 它不仅能在 Windows 、 Linux 和 MacOS 上运行，还能在移动设备上运行，包括 Android 系统和 iOS 系统。
- 它提供了非常简单的 Python API ，称为 *TF.Learn* （`tensorflow.contrib.learn`），兼容 Scikit-Learn 。接下来你会看到，只需几行代码，你就能用它训练各种神经网络。它之前是一个称为 *Scikit Flow* （或 *skflow* ）的独立项目。
- 它也提供另一个简单的 API ，称为 *TF-slim* （`tensorflow.contrib.slim`），来简化构建、训练和评估神经网络。
- 其他一些高级的 API 已经在 Tensorflow 之上独立构建，比如 [Keras](http://keras.io) （现在可以通过`tensorflow.contrib.keras`来使用） 和 [Pretty Tensor](https://github.com/google/prettytensor/) 。
- 它的核心 Python API 提供了更多的灵活性（以更高的复杂性为代价）来创建各种计算，包括任何你能想到的神经网络架构。
- 它包括了许多机器学习操作的高效 C++ 实现，特别是那些需要构建神经网络的。还有 C++ API ，能定义自己的高性能操作。
- 它提供了一些高级优化节点，用于搜寻最小化损失函数的参数。它们都很易于使用，因为 Tensorflow 会自动处理你定义函数的梯度。这被称为**自动微分**（ *automatic differentiating*，或 *autodiff* ）。
- 它也有一个优秀的可视化工具，称为 *TensorBoard* ，你可以浏览计算图，观察学习曲线等等。
- Google 还推出了 [用于运行 Tensorflow 图的的云服务](https://cloud.google.com/ml) 。
- 最后，它还有一支热忱的团队，和一群热情而乐于助人的开发者，以及一个致力于不断改善的成长中的社区。它是 GitHub 上最流行的开源项目之一，有越来越多的优秀项目正在基于它进行构建（例如，在 [https://www.tensorflow.org/](https://www.tensorflow.org/) 或 [https://github.com/jtoy/awesome-tensorflow](https://github.com/jtoy/awesome-tensorflow) 上查看资源页面）。如果要提问技术性问题，你应该使用 [http://stackoverflow.com/](http://stackoverflow.com/) ，将你的问题标记为 “`tensorflow`” 。你也可以通过 GitHub 提交 bug 和功能需求。对于一般讨论，可以加入 [Google 小组](http://goo.gl/N7kRF9) 。

在本章中，我们会介绍 Tensorflow 的基础，从安装到创建、运行、保存和可视化简单的计算图。在你创建自己的第一个神经网络前（下一章我们会进行），熟练掌握这些基础很重要。

![table](./images/chap09/9-table.png)

## 安装

来开始吧！假设你已经在第二章跟着安装说明安装了 Jupyter 和 Scikit-Learn ，你可以简单地使用 pip 来安装 Tensorflow 。如果你使用 virtualenv 构建了独立环境，你首先需要激活它：

```python
$ cd $ML_PATH   # 你的机器学习工作路径(e.g., $HOME/ml)
$ source env/bin/activate
```

接下来，安装 Tensorflow ：

```python
pip3 install --upgrade tensorflow
```

> **笔记**
> 为了 GPU 支持，你需要安装`tensorflow-gpu`而不是`tensorflow`。详见第十二章。

输入以下命令，来测试你的安装。它应该输出你安装的 Tensorflow 版本。

```python
$ python3 -c 'import tensorflow; print(tensorflow.__version__)'
1.0.0
```

## 创建你的第一个图，在会话中运行

下面的代码会创建图 9-1 中的图：

```python
import tensorflow as tf
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
```

这样就完成了！要理解的最重要的事是代码
