# 第六章 决策树

就像 SVM ，决策树也是一种多功能机器学习算法，既可以实现分类任务，又能实现回归任务，甚至还能处理多输出任务。它们是很强大的算法，可以拟合复杂的数据集。例如，在第二章中，你在加利福尼亚房价数据集上训练了`DecisionTreeRegressor`模型，拟合效果很好（实际上都过拟合了）。

决策树也是随机森林的基本组成（见第七章），而随机森林是当今最强大的机器学习算法之一。

在本章中，我们会首先讨论如何用决策树训练、可视化以及进行预测。然后我们会使用 Scikit-Learn 来学习 CART 训练算法，讨论如何正则化树并将它们用于回归任务。最后，我们会讨论一些决策树的局限。

## 训练和可视化决策树

为了理解决策树，首先创建一个决策树，看看它是如何进行预测的。下面的代码在鸢尾花数据集（见第四章）上训练了`DecisionTreeClassifier`：

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:] # petal length and width 
y = iris.target
   
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
```

你可以可视化训练好的决策树，通过`export_graphviz()`方法来输出图像定义文件，命名为*iris_tree.dot*。

```python
from sklearn.tree import export_graphviz

export_graphviz(
		tree_clf,
		out_file=image_path("iris_tree.dot"),
		feature_names=iris.feature_names[2:],
		class_names=iris.target_names,
		rounded=True,
		filled=True
				)
```

然后你可以使用`graphviz`包里的`dot`命令行工具，将 *.dot* 文件转换为 PDF 或 PNG 等多种格式。下面的命令行会将 .dot 文件转换为 .png 图像文件：

```
$ dot -Tpng iris_tree.dot -o iris_tree.png
```

第一个决策树如图 6-1 。

![1](./images/chap6/6-1.png)

## 进行预测

来看看图 6-1 中的决策树是如何进行预测的。假设你发现了一朵鸢尾花，想要将它分类。你从根节点（*root node*）（深度为 0 ，在顶端）开始：该节点询问花瓣长度是否小于 2.45 厘米
