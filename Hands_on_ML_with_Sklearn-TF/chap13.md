# 第十三章：卷积神经网络

尽管在 1996 年， IBM 的超级计算机深蓝击败了世界国际象棋冠军 Garry	Kasparov ，但是直到现在计算机都不能可靠地解决一些琐碎复杂的任务，比如检测照片中的小狗，或是语音识别。为什么这些任务对于人类而言不费吹灰之力？答案是感知发生在我们意识到的领域之外，在专门化的视觉、听觉以及大脑中的其他感知模块中。当感知信息达到意识层时，它已经被高级特征所装饰。例如，当你看一张可爱的小狗照片时，你不能选择*不*去看这只小狗，或者*不*去注意它的可爱。你也不能解释你是*如何*认出这是一只可爱的小狗，这对你而言是显而易见的。因此，我们不能相信自己的主观经验：感知并不是无关紧要的，为了理解它，我们必须研究感知模块是如何工作的。

卷积神经网络（CNN）是从大脑视觉皮层的研究中出现的，自从 20 世纪 80 年代以来，它们一直被用于图像识别。在近几年，多亏计算能力的进步、可用训练数据的数量的增加以及训练深度网络的技巧的提升， CNN 致力于在一些复杂视觉任务上获得超人的表现。它们强化了图像搜索服务、无人驾驶汽车、自动视频分类等领域。此外， CNN 并不局限于视觉感知：它们在其他任务中也很成功，比如语音识别或自然语言处理（*natural	language	processing*，NLP）。不过，我们现在专注于视觉应用。

在本章中，我们会介绍 CNN 的由来，构建模块的外观，以及如何用 Tensorflow 实现它们。之后我们会展示一些最佳的 CNN 架构。

## 视觉皮层的架构

在 1958 年 和 1959 年， David H. Hubel 和 Torsten Wiesel 进行了一系列对猫的实验（几年后是对猴子的实验），在视觉皮层的结构上给出了重要的见解（作者因此获得了 1981 年的诺贝尔生理和医学奖）。特别地，他们展示了视觉皮层中许多神经元都有一个微小的**局部感受野**（*local receptive field*），意味着它们只对视野中有限区域内的视觉刺激有反应（见图 13-1 ， 5 个神经元的局部感受野用虚线圆圈表示）。不同神经元的感受野可能会重叠，它们一起铺满了整个视觉域。此外，作者也展示了一些神经元只对水平方向的图像有反应，而另一些只对不同方向上的图像有反应（两个神经元也许会有相同的感受野，但是对不同方向的图像做出反应）。他们也注意到，有些神经元有更大的感受野，它们会对更复杂的图案——低级图案的组合——做出反应。这些观测结果引发了猜想，高级神经元基于邻近低级神经元的输出（在图 13-1 中，注意每个神经元只从前一层上连接了部分神经元）。这个强大的架构可以在视野中任何地方检测各种复杂模图案。

![1](./images/chap13/13-1.png)

这些视觉皮层的研究启发了 1980 年的 [新认知机](http://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf)  ，逐渐演变为我们现在称呼的**卷积神经网络**（*convolutional neural networks*）。一个重要的里程碑是 1998 年 由 Yann	LeCun,	Léon	Bottou,	Yoshua	Bengio,	and	Patrick	Haffner 发表的 [论文](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) ，提出了著名的 *LeNet-5* 架构，被广泛运用于手写支票号码识别。这个架构有一些你已知的构建模块，比如全连接层和 sigmoid 激励函数，不过也介绍了两个全新的构建模块：**卷积层**（*convolutional layers*）和**池化层**（*pooling layers*）。现在让我们来看一看。

> **笔记**
> 对于图像识别任务，为什么不简单使用一个拥有全连接层的常规深度神经网络？不幸的是，尽管这样在小型图片（比如 MNIST ）上运行良好，但是，它在大型图片上就会由于需求参数过多而崩溃。例如，一张 100×100 的图片有 10000 个像素，如果第一层有 1000 个神经元（这已经严重限制了传输到下一层的信息量），这就意味着一共有 1000 万条连接。这还只是第一层。 CNN 使用局部连接层解决这个问题。

## 卷积层

CNN 中最重要的构建模块就是卷积层（*convolutional layer*）：第一卷积层的神经元不是连接到输入图像的每个像素上（就像上一章），而是只连接到感受野上的像素（见图 13-2 ）。依次，第二卷积层中的每个神经元也只连接位于第一卷积层小矩形内的神经元。这种架构允许网络专注于第一隐藏层中的低级特征，然后将它们组合为下一隐藏层中的高级特征，以此类推。这种层次结构在现实世界的图像中很常见，这也是 CNN 在图像识别上表现优异的原因。

![2](./images/chap13/13-2.png)

> **笔记**
> 到目前为止，所有我们看到的多层神经网络都含有由一长串神经元组成的层，所以将输入图像传递给神经网络之前，我们不得不将它们压缩到一维。现在，每一层都是二维的形式，神经元与相关输入的匹配变得更加容易。

在给定的某层中，一个位于第 ![i](http://latex.codecogs.com/gif.latex?i) 行第 ![j](http://latex.codecogs.com/gif.latex?j) 列的神经元连接前一层中位于第 ![i](http://latex.codecogs.com/gif.latex?i) 行到 ![i+f_h-1](http://latex.codecogs.com/gif.latex?i&plus;f_h-1) 行、第 ![j](http://latex.codecogs.com/gif.latex?j) 列到 ![j+f_w-1](http://latex.codecogs.com/gif.latex?j&plus;f_w-1) 列的神经元的输出，其中 ![f_h](http://latex.codecogs.com/gif.latex?f_h) 和 ![f_w](http://latex.codecogs.com/gif.latex?f_w) 分别是感受野的高度和宽度（见图 13-3 ）。为了使该层的高度和宽度与前一层保持一致，通常会在输入周围添加零，如图所示。这被称为**零填充**（*zero padding*）。

![3](./images/chap13/13-3.png)

通过将感受野隔开，还可以将较大的输入层与较小的层相连接，如图 13-4 所示。两个连续感受野之间的距离被称为**步幅**（*stride*）。在图中，一个 5×7 的输入层（添加了零填充）与一个 3×4 的层相连接，使用了 3×3 的感受野，步幅为 2 （本例中各个方向的步幅都相同，但并不总是这样的）。上层中位于第 ![i](http://latex.codecogs.com/gif.latex?i) 行第 ![j](http://latex.codecogs.com/gif.latex?j) 列的神经元与前一层中位于第 ![i\times s_h](http://latex.codecogs.com/gif.latex?i%5Ctimes%20s_h) 行到 ![i\times s_h+f_h-1](http://latex.codecogs.com/gif.latex?i%5Ctimes%20s_h&plus;f_h-1) 行、第 ![j\times s_w+f_w-1](http://latex.codecogs.com/gif.latex?j%5Ctimes%20s_w&plus;f_w-1) 列的神经元的输出层相连接，其中 ![s_h](http://latex.codecogs.com/gif.latex?s_h) 和 ![s_w](http://latex.codecogs.com/gif.latex?s_w) 分别是垂直和水平方向上的步幅。

![4](./images/chap13/13-4.png)

### 过滤器

神经元的权重可以表示为感受野大小的小图像。例如，图 13-5 展示了两种可能的权重集合，称为**过滤器**（*filters*）或**卷积核**（*convolution kernels*）。第一种表示为中央有一条垂直白线的黑色矩形（它是一个 7×7 的矩阵，除了中间一列都是 1 ，其余都是 0 ）；使用了这些权重的神经元会忽视感受野中除了中央垂直线以外的一切事物（因为所有的输入都会乘上零，除了中央垂直线的那一列）。第二种过滤器是中央有一条水平白线的黑色矩形。同样地，使用了这些权重的神经元会忽视感受野中除了中央水平线以外的一切事物。

现在如果某一层中的神经元使用相同的垂直线过滤器（且具有相同的偏置项），你将图 13-5 中底部所示的输入图像传递给网络，会得到左上角的输出图像。注意，垂直白线被增强，而其余的则变模糊。类似地，右上角的图像是使用水平线过滤器的神经元得到的图像；注意，水平白线被增强，而其余的则变模糊。因此，使用同一过滤器的神经元层会给你**特征映射**（*feature map*），它会突出图像中和过滤器最相似的区域。在训练过程中， CNN 会找到对任务最有利的过滤器，它会学习将它们组合为更复杂的特征（例如，十字是图像中垂直过滤器和水平过滤器都激活的区域）。

![5](./images/chap13/13-5.png)

### 堆叠的多特征映射

到目前为止，简单起见，我们将每个卷积层都表现为一个薄的二维层，但是实际上它是由几组规模相同的特征映射组成的，所以用三维图来表示更准确（见图 13-6 ）。在特征映射中，所有的神经元共享相同的参数（权重和偏置项），但是不同的特征映射可能有不同的参数。神经元的感受野与之前描述相同，不过它会扩展到之前所有层的特征映射上。简而言之，卷积层同时在输入中应用多个过滤器，可以在输入中的任何位置检测多种特征。

> **笔记**
> 事实上，特征映射中所有神经元共享相同参数会大幅降低模型中的参数数量，不过最重要的是，这意味着一旦 CNN 学会了识别某个位置的特征，它就能在任何其他位置识别它。相反地，一旦常规的 DNN 学会了识别某个位置的特征，它就只能在该特定的位置识别它。

此外，输入图像也由多个子层组成：每个**颜色通道**（*color channel*）都有一个。典型的三种是：红，绿，蓝（ RGB ）。灰度图只有一个通道，但是有些图像可能有更多——比如，捕捉额外光频（比如红外线）的卫星图像。

![6](./images/chap13/13-6.png)

具体地，位于卷积层 ![l](http://latex.codecogs.com/gif.latex?l) 中特征映射 ![k](http://latex.codecogs.com/gif.latex?k) 的第 ![i](http://latex.codecogs.com/gif.latex?i) 行第 ![j](http://latex.codecogs.com/gif.latex?j) 列的神经元与前一层 ![l-1](http://latex.codecogs.com/gif.latex?l-1)中位于第 ![i\times s_h](http://latex.codecogs.com/gif.latex?i%5Ctimes%20s_h) 行到 ![i\times s_h+f_h-1](http://latex.codecogs.com/gif.latex?i%5Ctimes%20s_h&plus;f_h-1) 行、第 ![j\times s_w](http://latex.codecogs.com/gif.latex?j%5Ctimes%20s_w) 列到 ![j\times s_w+f_w-1](http://latex.codecogs.com/gif.latex?j%5Ctimes%20s_w&plus;f_w-1) 列的神经元的输出相连接，遍布所有特征映射（在 ![l-1](http://latex.codecogs.com/gif.latex?l-1) 层中）。所有位于同一行同一列、但在不同的特征映射中的神经元与上一层中完全相同的神经元的输出相连接。

公式 13-1 用一个数学公式概括了之前的解释：它展示了如何计算卷积层中给定神经元的输出。因为索引不同，所以它有点儿丑，不过它所做的是计算所有输入的权重总和，加上偏置项。

![z_{i,j,k}=b_k+\sum_{u=0}^{f_h-1}\sum_{v=0}^{f_w-1}\sum_{k'=0}^{f_{n'}-1}x_{i',j',k'}\cdot w_{u,v,k',k}\;\;\;\mathrm{with}\begin{cases} 
i'=i\times s_h+u\\
j'=j\times s_w+v
\end{cases}](http://latex.codecogs.com/gif.latex?z_%7Bi%2Cj%2Ck%7D%3Db_k&plus;%5Csum_%7Bu%3D0%7D%5E%7Bf_h-1%7D%5Csum_%7Bv%3D0%7D%5E%7Bf_w-1%7D%5Csum_%7Bk%27%3D0%7D%5E%7Bf_%7Bn%27%7D-1%7Dx_%7Bi%27%2Cj%27%2Ck%27%7D%5Ccdot%20w_%7Bu%2Cv%2Ck%27%2Ck%7D%5C%3B%5C%3B%5C%3B%5Cmathrm%7Bwith%7D%5Cbegin%7Bcases%7D%20i%27%3Di%5Ctimes%20s_h&plus;u%5C%5C%20j%27%3Dj%5Ctimes%20s_w&plus;v%20%5Cend%7Bcases%7D)

- ![z_{i,j,k}](http://latex.codecogs.com/gif.latex?z_%7Bi%2Cj%2Ck%7D) 是卷积层（ ![l](http://latex.codecogs.com/gif.latex?l) 层）特征映射 ![k](http://latex.codecogs.com/gif.latex?k) 的位于第 ![i](http://latex.codecogs.com/gif.latex?i) 行第 ![j](http://latex.codecogs.com/gif.latex?j) 列的神经元的输出。
- ![s_h](http://latex.codecogs.com/gif.latex?s_h) 和 ![s_w](http://latex.codecogs.com/gif.latex?s_w) 分别是垂直和水平方向上的步幅， ![f_h](http://latex.codecogs.com/gif.latex?f_h) 和 ![f_w](http://latex.codecogs.com/gif.latex?f_w) 是感受野的高度和宽度， ![f_{n'}](http://latex.codecogs.com/gif.latex?f_%7Bn%27%7D) 是上一层（ ![l-1](http://latex.codecogs.com/gif.latex?l-1) 层）中特征映射的数量。
- ![x_{i',j',k'}](http://latex.codecogs.com/gif.latex?x_%7Bi%27%2Cj%27%2Ck%27%7D) 是卷积层 ![l-1](http://latex.codecogs.com/gif.latex?l-1) 中位于第 ![i'](http://latex.codecogs.com/gif.latex?i') 行第 ![j'](http://latex.codecogs.com/gif.latex?j') 列特征映射 ![k'](http://latex.codecogs.com/gif.latex?k') 的神经元的输出。
- ![b_k](http://latex.codecogs.com/gif.latex?b_k) 是特征映射 ![k'](http://latex.codecogs.com/gif.latex?k') （![l](http://latex.codecogs.com/gif.latex?l) 层）的偏置项。你可以把它看作是调整特征映射 ![k](http://latex.codecogs.com/gif.latex?k) 整体亮度的旋钮。
- ![w_{u,v,k',k}](http://latex.codecogs.com/gif.latex?w_%7Bu%2Cv%2Ck%27%2Ck%7D) 是卷积层 ![l](http://latex.codecogs.com/gif.latex?l) 特征映射 ![k](http://latex.codecogs.com/gif.latex?k) 中任意两个神经元之间的连接权重，它的输入位于第 ![u](http://latex.codecogs.com/gif.latex?u) 行第 ![v](http://latex.codecogs.com/gif.latex?v) 列（相对于神经元的感受野），特征映射为 ![k'](http://latex.codecogs.com/gif.latex?k') 。

### Tensorflow 实现

在 Tensorflow 中，每张输入图像通常表示为 3D 张量`shape [height, width, channels]`。小批量表示为 4D 张量`shape	[mini-batch size, height, width, channels]`。卷积层的权重表示为 4D 张量 ![f_h,f_w,f_{n'},f_n](http://latex.codecogs.com/gif.latex?f_h%2Cf_w%2Cf_%7Bn%27%7D%2Cf_n) 。卷积层的偏置项简单表示为 1D 张量`shape	[fn]`。

来看一个简单的例子。下面的代码使用了 Scikit-Learn 的`load_sample_images()`，加载了两张简单图像（两张彩图，一张是中国寺庙，另一张是一朵花）。之后它创建了两个 7×7 的过滤器（一个是垂直线过滤器，一个是水平线过滤器），将它们应用到两张图像上，使用一个由 Tensorflow 的`tf.nn.conv2d()`函数创建的卷积层（有零填充，步幅为 2 ）。最后，它绘制出其中一张图像的结果特征映射图（和图 13-5 右上角的图类似）。

```python
import numpy as np
from sklearn.datasets import load_sample_images
import numpy as np
import tensorflow as tf

# Load sample images
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape

# Create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line

# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME")

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})

plt.imshow(output[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
plt.show()
```

大部分的代码都是一目了然的，不过`tf.nn.conv2d()`这一行需要一些解释：

- `X`是输入的小批量（也是个 4D 张量，如前所述）。
- `filters`是应用的过滤器种类（也是个 4D 张量，如前所述）。
- `strides`是一个有四个元素的 1D 数组，中间两个元素是垂直和水平方向的步幅（ ![s_h](http://latex.codecogs.com/gif.latex?s_h) 和 ![s_w](http://latex.codecogs.com/gif.latex?s_w) ）。第一个和最后一个元素现在必须为 1 。以后可能会被用来指定批量的步幅（跳过一些实例）和频道步幅（跳过一些前一层的特征映射或频道）。
- `padding`必须为`"VALID"`或者`"SAME"`：
  - 如果设置为`"VALID"`，卷积层就*不*使用零填充，也许会忽视底部的一些行列，以及输入图像的右侧，取决于步幅，如图 13-7 所示（简单起见，这里只展示了水平维度，不过事实上垂直维度的逻辑应用也一样）。
  - 如果设置为`"SAME"`，卷积层会在必要时使用零填充。本例中，输出神经元的数量等于输入神经元的数量除以步幅，向上取整（本例中， ![\mathrm{ceil}(13/5)=3](http://latex.codecogs.com/gif.latex?%5Cmathrm%7Bceil%7D%2813/5%29%3D3) ）。然后在输入周围尽可能均匀地添加零。

![7](./images/chap13/13-7.png)

在这个简单例子中，我们手动创建了过滤器，但是在真正的 CNN 中，你会让训练算法自动找到最优过滤器。 Tensorflow 有一个`tf.layers.conv2d()`函数，它会创建各种过滤器（称为核（*kernel*）），并随机初始化它们。例如，下面的代码会创建一个输入占位符，它由拥有两个 7×7 特征映射的卷积层组合，使用了 2×2 的步幅（注意该函数只需要垂直和水平的步幅），以及`padding`为`"SAME"`：

```python
X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
conv = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2,2], padding="SAME")
```

不幸的是，卷积层有许多超参数：你必须选择过滤器的数量，它们的高度与宽度，步幅，和填充类型。你可以和往常一样使用交叉验证来找到最佳的超参数值，但会非常费时。我们稍后会讨论常见的 CNN 架构，能为你提供思路，找到在实践中表现最佳的超参数。

### 内存需求

CNN 的另一个问题是卷积层需要大量的 RAM ，尤其是在训练的时候，因为反向传播的反向传递需要正向传播中所有计算的中间值。

例如，考虑这样一个卷积层，有 5×5 的过滤器，输出 200 个规模为 150×100 的特征映射，步幅为 1 ，填充为`SAME`。如果输入是一张 150×150 的 RGB 图像（有三个通道），那么参数的数量就是 ![(5\times5\times3+1)\times200 =15,200](http://latex.codecogs.com/gif.latex?%285%5Ctimes5%5Ctimes3&plus;1%29%5Ctimes200%20%3D15%2C200) （ +1 对应偏置项），和整个全连接层相比还是很小的数量。然而，200 个特征映射每个都包含 150×100 个神经元，每个神经元都需要计算 ![5\times5\times3=75](http://latex.codecogs.com/gif.latex?5%5Ctimes5%5Ctimes3%3D75) 个输入的权重总和：总计就是 2.25 亿的浮点乘法运算。虽然不像全连接层那么糟糕，但是计算仍然很复杂。此外，如果特征映射是用 32 位浮点数表示的，那么卷积层的输出会占据 RAM 的 ![200\times150\times100\times32=96](http://latex.codecogs.com/gif.latex?200%5Ctimes150%5Ctimes100%5Ctimes32%3D96)  百万位（大约 11.4 MB ）。这只是一个实例！如果一个训练批量包含 100 个实例，那么该层会占用超过 1G 的 RAM ！

在推断（即对一个新的实例做出预测）中，一旦下一层的计算完成，被该层占用的 RAM 就会被释放，所以你只需要两个连续层所需的 RAM 。不过在训练期间，正向传播中计算的一切数据都需要被保留用于反向传递，所以所需 RAM 的数量（至少）为所有层所需 RAM 的总量。

> **提示**
> 如果训练因为内存不足错误而导致崩溃，你可以尝试减少小批量的大小。另外，你也可以尝试使用步幅来减少维度，或者删除一些层。抑或你可以尝试使用 16 位的浮点，舍弃 32 位的。除此之外，你还可以在多设备间分布 CNN 。

现在我们来看 CNN 的第二种常见构建模块：池化层。

## 池化层

一旦你理解了卷积层的工作原理，池化层就简单多了。它们的目标是对输入图像进行二次采样（即收缩），而减少计算负担、内存用量和参数数量（从而降低过拟合的风险）。减小输入图像的大小也可以使神经网络容忍轻微的图像偏移（位置不变）。

就像卷积层一样，池化层中的每个神经元都与前一层有限数量的神经元相连接，位于小的矩形感受野中。你必须定义它的大小、步幅、填充类型，和之前一样。不过，池化神经元没有权重，它所做的是使用聚集函数（比如最大值或平均值）来聚合输入。图 13-8 展示了一个**最大池化层**（*max pooling layer*），它是最普通的一种池化层。在这个例子中，我们使用一个 2×2 的池化核，步幅为 2 ，无填充。注意，只有每个核中的最大输入值才会进入下一层。其他的输入都会被舍弃。

![8](./images/chap13/13-8.png)

这显然是种非常具有破坏性的层：即便只有一个 2×2 的核和数值为 2 的步幅，输出也会在各个方向上缩小为原来的两倍（所以面积总共会缩小四倍），一下舍弃了 75% 的输入值。

池化层通常在每个输入频道上独立运作，所以输出宽度和输入宽度相同。你也许会池化深度维度，此时图像的空间维度（高度和宽度）仍未改变，但是频道的数量减少了。

在 Tensorflow 中实现最大池化层非常简单。下面的代码创建了一个最大池化层，使用了 2×2 的核，步幅为 2 ，无填充，之后将其应用到数据集中的所有的图像上：

```python
[...] # load the image dataset, just like above
# Create a graph with input X plus a max pooling layer
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")

with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})

plt.imshow(output[0].astype(np.uint8))  # plot the output for the 1st image
plt.show()
```

`ksize`参数包含了包含了核的形状，四种输入张量的维度：`[batch size, height, width, channels]`。Tensorflow 现在不支持池化多样例，所以`ksize`的第一个元素一定是 1 。此外，它也不支持同时池化空间维度（高度和宽度）和深度维度，所以`ksize[1]`和`ksize[2]`也必须是 1 ，否则`ksize[3]`一定是 1 。

为了创建一个**平均池化层**（*average pooling layer*），只需将`max_pool()`函数换为`avg_pool()`函数。

现在你已经了解了所有用于创建卷积神经网络的构建模块。让我们来看看如何组合它们。

## CNN 架构

典型的 CNN 架构由这些堆叠而成：一些卷积层（每一个都有一个 ReLU 层），然后是一个池化层，之后是另一些卷积层（+ ReLU ），然后是另一个池化层，以此类推。随着图像在神经网络中进展，它变得越来越小，不过由于卷积层的存在也变得越来越深（即有更多的特征映射）（见图 13-9 ）。在堆栈的顶部加入了常规的前馈神经网络，它由一些全连接层（+ ReLUs）组成，最后一层会输出预测结果（比如，一个输出估计类别概率的 softmax 层）。

![9](./images/chap13/13-9.png)

> **提示**
> 一种常见的错误是使用过大的卷积核。你能通过将两个 3×3 的核堆叠起来而获得和一个 9×9 的核相同的效果，也减少了计算量。

这些年来，这种基础架构已经发展出了许多变种，极大地导致了该领域的进步。这种进步的衡量标准就是比赛中的错误率，比如 ILSVRC [ImageNet challenge](http://image-net.org/)。在该比赛中，图像分类的 top-5 错误率在五年里从 26% 降到了 3%。 top-5 错误率是系统前五名不包含正确答案的预测中测试图像的数量。图像很大（ 256 像素），共有 1000 种类别，其中有一些非常不明显（比如区分 120 种狗的品种）。浏览获胜作品的演变过程是了解 CNN 运作原理的好办法。

我们首先来看经典的 LeNet-5 架构（ 1998 ），然后是 ILSVRC 挑战的三位胜者： AlexNet （ 2012 ）， GoogLeNet （ 2014 ）和 ResNet （ 2015 ）。

> **其他的可视化任务**
>   

### LeNet-5

LeNet-5 架构可能是最著名的 CNN 架构。之前提到过，它是由 Yann	LeCun 在 1998 年创造的，被广泛用于手写数字识别（ MNIST ）。它由表 13-1 所示的层组成。

![lenet5](./images/chap13/13-lenet5.png)

有一些额外的需要注意：

- MNIST 的图像是 28×28 像素，不过在传递给网络之前会被零填充扩展到 32×32 像素，并进行归一化。剩下的网络不使用任何填充，所以图像在网络中进展时大小一直在缩小。
- 平均池化层比往常要稍微复杂一些：每个神经元计算输入的平均值，再乘上一个学习系数（每个映射都有一个），再加上一个学习偏置项（也是每个映射都有一个），最后再应用于激励函数。
- C3 映射中大部分神经元只与三个或四个 S2 映射的神经元相连接（而不是六个）。详见原始论文中的表 1 。 
- 输出层有点特殊：每个神经元并不计算输入的点积和权重向量，而是输出它们输入向量与权重向量之间欧氏距离的平方。每个输出测量图像属于某一特定数字类别的程度。交叉验证损失函数现在是首选，因为它能更多地惩罚错误的预测，生成更大的梯度，也收敛得更快。

Yann LeCun 的 [网站](http://yann.lecun.com/) （ “LENET” 部分）展示了 LeNet-5 分类数字的优秀 demo 。

### AlexNet

[AlexNet CNN 架构](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 以巨大的优势赢得了 2012 年的 ImageNet ILSVRC 挑战：它达到了 17% 的 top-5 错误率，而第二名只有 26% ！它由 Alex Krizhevsky（也是它名字的由来）、 Ilya Sutskever 、 Geoffrey Hinton 开发。它和 LeNet-5 很相似，只是更大且更深，也是第一个直接将卷积层堆叠在一起的，而不是在卷积层上堆叠一个池化层。表 13-2 展示了该架构。

![alexnet](./images/chap13/13-alexnet.png)

为了减少过拟合，作者使用了两种我们先前讨论过的正则技术：首先在训练期间将丢失率（ 50% 的丢失率）应用于 F8 和 F9 的输出。其次，他们通过随机对图像进行各种偏移，水平翻转和改变光照条件，实现了数据增强。

AlexNet 也在 C1 层和 C3 层的 ReLU 之后立即使用竞争标准化步骤，被称为**局部响应标准化**（*local response normalization*）。这种标准化的形式使最强激活的神经元抑制同一位置上但特征映射相邻的神经元（这种竞争激活在生物神经元中已被观测到）。这就鼓励了特化不同的特征映射，强制它们分开，并让它们探索范围更广的特征，最终提高泛化能力。公式 13-2 展示了 LRN 的应用方法。

![b_i=a_i(k+\alpha\sum_{j=j_{\mathrm{low}}}^{j_{\mathrm{high}}}a_j^2)^{-\beta}\;\;\; \mathrm{with}\begin{cases} j_\mathrm{high}=\min(i+\frac{r}{2},f_n-1)\\ j_\mathrm{low}=\max(0,i-\frac{r}{2}) \end{cases}](http://latex.codecogs.com/gif.latex?b_i%3Da_i%28k&plus;%5Calpha%5Csum_%7Bj%3Dj_%7B%5Cmathrm%7Blow%7D%7D%7D%5E%7Bj_%7B%5Cmathrm%7Bhigh%7D%7D%7Da_j%5E2%29%5E%7B-%5Cbeta%7D%5C%3B%5C%3B%5C%3B%20%5Cmathrm%7Bwith%7D%5Cbegin%7Bcases%7D%20j_%5Cmathrm%7Bhigh%7D%3D%5Cmin%28i&plus;%5Cfrac%7Br%7D%7B2%7D%2Cf_n-1%29%5C%5C%20j_%5Cmathrm%7Blow%7D%3D%5Cmax%280%2Ci-%5Cfrac%7Br%7D%7B2%7D%29%20%5Cend%7Bcases%7D)

- ![b_i](http://latex.codecogs.com/gif.latex?b_i) 是位于某行 ![u](http://latex.codecogs.com/gif.latex?u) 某列 ![v](http://latex.codecogs.com/gif.latex?v) 、特征映射 ![i](http://latex.codecogs.com/gif.latex?i) 的神经元的标准化输出（注意在该公式中我们只考虑在该行该列的神经元，所以 ![u](http://latex.codecogs.com/gif.latex?u) 和 ![v](http://latex.codecogs.com/gif.latex?v) 不显示）。
- ![a_i](http://latex.codecogs.com/gif.latex?a_i) 是在 ReLU 步骤之后、标准化之前的神经元的激励。
- ![k,\alpha,\beta,r](http://latex.codecogs.com/gif.latex?k%2C%5Calpha%2C%5Cbeta%2Cr) 是超参数。 ![k](http://latex.codecogs.com/gif.latex?k) 被称为偏置项， ![r](http://latex.codecogs.com/gif.latex?r) 被称为深度半径。
- ![f_n](http://latex.codecogs.com/gif.latex?f_n) 是特征映射的数量。

例如，如果 ![r=2](http://latex.codecogs.com/gif.latex?r%3D2) ，且神经元有很强的激励，那么它会抑制在它上下的特征映射的神经元的激活。

在 AlexNet 中，超参数如下设置： ![r=2,\alpha=0.00002,\beta=0.75,k=1](http://latex.codecogs.com/gif.latex?r%3D2%2C%5Calpha%3D0.00002%2C%5Cbeta%3D0.75%2Ck%3D1) 。这一步可以通过 Tensorflow 的`tf.nn.local_response_normalization()`操作来实现。

AlexNet 的一种变种称为 *ZF	Net* 由 Matthew Zeiler 和 Rob Fergus 开发，获得了 2013 年 ILSVRC 挑战的胜利。它本质上是超参数（特征映射的数量、核的大小、步幅等等）经过微调的 AlexNet 。

### GoogLeNet

[GoogLeNet 架构](http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) 由 Google Research 的 Christian Szegedy 等人开发的，它获得了 2014 年 ILSVRC 挑战的胜利，使 top-5 错误率降到了低于 7% 。它优秀的表现很大程度上是因为网络比先前的 CNN 都要深得多（见图 13-11 ）。这是通过被称为**起始模块**（*inception modules*）的子网络实现的，它允许 GoogLeNet 使用比之前的架构更高效的参数：事实上 GoogLeNet 比 AlexNet 少了十倍的参数（ AlexNet 大约 6000 万， GoogLeNet 大约 600 万）。

图 13-10 展示了一个起始模块的架构。“3 × 3 + 2(S)”的记号代表该层使用了 3 × 3 的核，步幅为 2 ，填充为 SAME 。输入信号首先被复制传递到四种不同的层。所有的卷积层都使用了 ReLU 激励函数。注意第二组卷积层使用了不同的核大小（ 1 × 1 ，3 × 3 和 5 × 5 ），允许它们在不同的比例下获取图案。另外也要注意，每个独立的层都使用了步幅为 1 的零填充（即便是最大池化层），所以它们的输出都有和输入一样的高度和宽度。这样就可以在最后的深度连接层（*depth concat layer*）上沿着深度维度连接所有的输出（即从四个顶部卷积层将所有的特征映射堆叠起来）。这个连接层可以通过 Tensorflow 的`tf.concat()`操作，设置`axis=3`（ axis 3 是深度）来实现。

![10](./images/chap13/13-10.png)

你也许想知道为什么起始模块有一个 1 × 1 核的卷积层。当然这些层不能获取任何特征，因为它们一次只看一个像素？事实上，这些层为以下两个目的服务：

- 首先，它们被设置为输出比输入少得多的特征映射，所以它们作为**瓶颈层**（*bottleneck layers*），意味着
