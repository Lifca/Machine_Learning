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

这样就完成了！要理解的最重要的事是代码并不执行任何计算，即使它看起来做了（尤其是最后一行）。它只创建了一个计算图。事实上，甚至连变量都没有初始化。为了对图求值，你需要打开一个 Tensorflow 的会话（*session*），并用它初始化变量和求`f`值。 Tensorflow 的会话负责处理在 CPU 或 GPU 设备上的操作并运行它们，它会保留所有的变量值。下面的代码会创建一个会话，初始化变量，求出`f`的值，之后关闭会话（释放资源）。

```python
>>> sess = tf.Session()
>>> sess.run(x.initializer)
>>> sess.run(y.initializer)
>>> result = sess.run(f)
>>> print(result)
42
>>> sess.close()
```

每次都不得不重复`sess.run()`有点麻烦，不过幸运的是有更好的方法：

```python
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
```

在`with`块代码中，会话被设置为默认会话。调用`x.initializer.run()`和调用`tf.get_default_session().run(x.initializer)`是等效的，类似地，调用`f.eval()`和调用`tf.get_default_session().run(f)`也是等效的。这样就增加了代码的可读性。此外，会话会在块代码的最后自动关闭。

你可以使用`global_variables_initializer()`函数，无需手动初始化每个单独变量。注意，它并不会立即执行初始化，而是在图中创建一个节点，它在运行时会初始化所有变量：

```python
init = tf.global_variables_initializer()  # 准备初始化节点
with tf.Session() as sess:
    init.run()	# 事实上初始化了所有变量
    result = f.eval()
```

在 Jupyter 或者 Python Shell 中，你可能会偏于创建一个`InteractiveSession`。它与常规`Session`的唯一区别是当`InteractiveSession`被创建时，它会自动被设为默认会话，所以你不需要再写一个`with`块了（但是当你完事后，你需要手动关闭会话）。

```python
>>> sess = tf.InteractiveSession()
>>> init.run()
>>> result = f.eval()
>>> print(result)
42
>>> sess.close()
```

一个 Tensorflow 程序通常被分为两部分：第一部分会构建一个计算图（被称为构造阶段（*construction phase*）），第二部分会运行它（这是执行阶段（*execution phase*））。构造阶段通常建立计算图，代表机器学习模型以及所需用于训练的计算。执行阶段通常进行循环，重复对训练步骤求值（例如，每步一个小批量），逐渐改善模型的参数。我们稍后会有一个样例。

## 管理图

任何你所创建的节点都会被自动加入默认图中：

```python
>>> x1 = tf.Variable(1)
>>> x1.graph is tf.get_default_graph()
True
```

大多数情况下它运行没问题，但是有时你也许想要管理多个独立图。你可以创建新的`Graph`，暂时在`with`块中将它设为默认图，像这样：

```python
>>> graph = tf.Graph()
>>> with graph.as_default():
...     x2 = tf.Variable(2)
...
>>> x2.graph is graph
True
>>> x2.graph is tf.get_default_graph()
False
```

> **提示**
> 在 Jupyter （或 Python Shell ）中，在实验时通常会多次运行同一个命令。因此，你可能会得到有许多重复节点的默认图。一种解决方法是重启 Jupyter 内核（或 Python Shell ），不过更便捷的方法是运行`tf.reset_default_graph()`，重置默认图。

## 节点值的生命周期

当你求一个节点的值时， Tensorflow 会自动确定依赖的节点集，并优先求出这些节点的值。例如，考虑下面的代码：

```python
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:
    print(y.eval())	  # 10
    print(z.eval())	  # 15
```

首先，代码定义了一个非常简单的图。之后它启动了会话，运行图来对`y`求值： Tensorflow 自动检测到`y`依赖于`x`，而`x`依赖于`w`，所以它会优先计算`w`，然后是`x`，再是`y`，之后返回`y`的值。最后，代码运行图来求`z`的值。再次强调， Tensorflow 检测到它必须先对`w`和`x`求值。有一点很重要，之前求得的`w`和`x`不会被重用。简而言之，之前的代码会计算`w`和`x`两次。

所有的节点值都会在图运行间被删除，除了被会话跨图保持的变量值（队列和读写器也会保持一些状态，详见第十二章）。变量在初始化时开始生命周期，在会话关闭时结束生命周期。

如果你想要高效地计算`y`和`z`，而不用在之前的代码中求两次`w`和`x`，你必须让 Tensorflow 在一张图运行时同时计算`y`和`z`，就像下面的代码一样：

```python
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)    # 10
    print(z_val)    # 15
```

> **警告**
> 在单进程 Tensorflow 中，多会话并不共享状态，即便它们重用了同一张图（每个会话都有属于自己的每个变量的备份）。在分布式 Tensorflow 中（见第十二章），变量状态储存在服务器上，不在会话上，所以多会话也能共享同变量。

## Tensorflow 实现线性回归

Tensorflow 操作（也简称为 *ops* ）可以接收任意数量的输入，生成任意数量的输出。例如，加法和乘法操作接收两个输入，生成一个输出。常量和变量无需输入（被称为源操作（*source ops*））。输入与输出是多维数组，被称为**张量**（*tensors*，也是名字 “tensor flow” 的由来）。就像 NumPy 的数组，张量也有类型和形状。事实上，在 Python API 中，张量可以简单地用 NumPy 数组来表示。它们通常包含浮点数，不过你也可以让它们来承载字符串（任意字节的数组）。

迄今为止的样例中，张量只包含了单个标量值，不过你当然可以对任何形状的数组进行计算。例如，下面的代码操作二维数组，在加利福尼亚房价数据集上实现了线性回归（在第二章介绍过）。它从获取数据开始，之后在所有训练实例上添加一个额外的偏置输入特征（ ![x_0=1](http://latex.codecogs.com/gif.latex?x_0%3D1) ）（这一步使用了 NumPy ，所以能立即运行）；再之后它会创建两个 Tensorflow 常量节点`X`和`y`来保存数据和目标，并用一部分 Tensorflow 提供的矩阵操作定义`theta`。这些矩阵函数——`transpose()`，`matmul()`和`matrix_inverse()`——都是不言自明的，不过和往常一样，它们并不会立即执行计算。相反地，它们在图中创建节点，在图运行的时候节点就会执行。你也许意识到了`theta`的定义对应于正规方程（见第四章）。最后，这些节点创建一个会话，用它来计算`theta`。

```python
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
```

如果你有 GPU 的话，和直接用 NumPy 计算正规方程相比，这段代码的主要优点是 Tensorflow 会自动在你的 GPU 上运行（如果你安装了有 GPU 支持的 Tensorflow ，当然，详见第十二章）。

## 实现梯度下降

让我们来试试使用批量梯度下降（在第四章介绍过），而不用正规方程。首先，我们会手动计算梯度，之后我们会使用 Tensorflow 的 autodiff 功能来使 Tensorflow 自动计算梯度，最后我们会使用一些 Tensorflow 现成的优化器。

> **警告**
> 当你使用梯度下降时，记住首先标准化输入特征向量是很重要的，否则训练可能会很慢。你可以通过 Tensorflow 、 NumPy 、 Scikit-Learn 的`StandardScaler`，或者其他你更偏好的解决方法来完成这一步。下面的代码假设标准化已经完成。

### 手动计算梯度

下面的代码应该相当容易看懂，除了一些新的元素：

- `random_uniform()`函数在图中创建一个新的节点，会生成一个包含了随机值的张量，给定形状和值域，类似 NumPy 的`rand()`函数。
- `assign()`函数创建一个节点，它会赋给变量新值。在本例中，它实现了批量梯度下降的步骤 ![\theta^{(\mathrm{next\;step})}=\theta-\eta\nabla_{\theta}\mathrm{MSE}(\theta)](http://latex.codecogs.com/gif.latex?%5Ctheta%5E%7B%28%5Cmathrm%7Bnext%5C%3Bstep%7D%29%7D%3D%5Ctheta-%5Ceta%5Cnabla_%7B%5Ctheta%7D%5Cmathrm%7BMSE%7D%28%5Ctheta%29) 。
- 主要的循环会一遍遍执行训练步骤（共`n_epochs`次），每 100 次循环都会打印当前的均方误差（`mse`）。你应该看到每次迭代均方误差都在下降。

```python
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
        
    best_theta = theta.eval()
```

### 使用自动微分

之前的代码运行没问题，但是它需要从损失函数（ MSE ）中利用数学推导梯度。在线性回归的例子中，这相当简单，但是如果你在深度神经网络也这么做，你会有点儿头疼：这将会是乏味而易错的。你可以使用符号微分（*symbolic differentiation*）来自动找到偏导数的方程，不过生成的代码并不一定非常有效。

为了理解原因，考虑函数 f(x)=exp(exp(exp(x))) 。如果你知道微积分，你能求出它的导数 f′(x)=exp(x) × exp(exp(x)) × exp(exp(exp(x))) 。如果你将 f(x) 和 f'(x) 分别按照它们的样子编写代码，那么你的代码不会太有效。更有效的解决方法是写一个函数，先计算 exp(x) ，之后是 exp(exp(x)) ，再是 exp(exp(exp(x))) ，并返回这三个值。这样可以直接得到 f(x) （第三项），如果你需要导数，将三项全部相乘即可。用传统方法你不得不调用`exp`函数九次来计算 f(x) 和 f'(x) ，而用这种方法你只需要调用三次。

当你的函数被任意代码定义时，它会变得更糟。你能找到计算下面函数偏导数的方程（或代码）吗？提示：别试。

```python
def my_func(a, b):
    z = 0
    for i in range(100):
        z = a * np.cos(z + i) + z * np.sin(b - i)
    return z
```

幸运的是， Tensorflow 的自动微分功能可以帮上忙：它会自动而高效地为你计算梯度。只需简单地将上一节梯度下降代码中`gradients = ...`这一行替换为下面的一行代码，它会继续正常运行。

```python
gradients = tf.gradients(mse, [theta])[0]
```

`gradients()`函数获取了 op （本例中是`mse`）和一个变量列表（本例中只有`theta`），它为每个变量创建了 op 列表，来计算 op 关于每个变量的梯度。所以`gradients`节点会计算均方误差对于`theta`的梯度向量。

有四种自动计算梯度的主要方法。表 9-2 进行了总结。 Tensorflow 使用反向模式自动微分（*reverse-mode autodiff*），当输入很多、输出很少时它的表现很完美（高效而准确），比如在神经网络中。它只需遍历图 ![n_{\mathrm{outputs}}+1](http://latex.codecogs.com/gif.latex?n_%7B%5Cmathrm%7Boutputs%7D%7D&plus;1) 次就能计算所有输出关于输入的偏导数。

![table](./images/chap09/9-table.png)

如果你对它的工作原理感兴趣，请参阅附录 D 。

### 使用优化器

Tensorflow 会为你计算梯度。不过可以更简单：它也提供了许多现成的优化器，包括梯度下降优化器。你可以简单地把之前的`gradients = ...`和`training_op = ...`替换为下面的代码，一切都会正常工作：

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
```
