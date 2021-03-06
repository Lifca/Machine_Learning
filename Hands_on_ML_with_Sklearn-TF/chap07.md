# 第七章：集成学习与随机森林

假设你向数千人询问一个复杂的问题，然后把他们的答案合并起来。在很多时候，你会发现这个合并的答案要优于专家的答案。这被称为**群众智慧**（*wisdom of the crowd*）。类似地，如果你把一组预测器（比如分类器或回归器）的预测结果合并起来，你通常会得到比最佳单一预测器更好的预测结果。这一组预测器就称为集成，因此，这种技术被称为**集成学习**（*Ensemble Learning*），集成学习算法被称为**集成方法**（*Ensemble method*）。

例如，你能在训练集的不同的随机子集上训练一组决策树分类器。为了进行预测，你得到了所有树的预测结果，将票数最多的类作为预测结果（见第六章最后的练习）。这样一种决策树的集成被称为**随机森林**（*Random Forest*），它尽管很简单，却是当今最强大的机器学习算法之一。

此外，我们在第二章讨论过，在一个项目的最后你会经常使用集成方法，一旦你建立了一些不错的预测器，将它们组合为一个更好的预测器。事实上，机器学习竞赛中的获胜算法经常包含了一些集成方法（最知名的在 [Netflix Prize competition](http://netflixprize.com/)）。

在本章中，我们会讨论最流行的集成方法，包括 bagging， boosting， stacking 还有一些其他的算法。我们也会探索随机森林。

## 投票分类器

假设你已经训练了一些分类器，每个都有 80% 的精度。你可能有一个逻辑回归分类器， SVM 分类器，随机森林分类器，K 近邻分类器等等（见图 7-1 ）。

![1](./images/chap7/7-1.png)

一种创建更优分类器的简单方法是合并每一个分类器，预测类别为票数最多的类。这种多数票分类器被称为硬投票分类器（见图 7-2 ）。

![2](./images/chap7/7-2.png)

令人惊讶的是，这个投票分类器通常会比集成学习中最好的分类器有更高的精度。事实上，即便每个学习器都是**弱学习器**（*weak learner*）（意味着它只比胡乱猜测好一点），集成仍然是一个**强分类器**（*strong learner*）（精度更高），假如弱学习器的数量足够多，它们就会足够多样性。

这怎么可能？下面的类比能帮你解开谜团。假设你有一个有偏差的硬币，有 51% 的概率是正面， 49% 的概率是背面。如果你抛 1000 次硬币，大约会得到 510 次正面和 490 次背面，因此大多数是正面。如果你用数学计算，你会发现在 1000 次抛硬币后获得多数票为正面的概率接近 75% 。你抛硬币的次数越多，这个概率就会越高（例如，你抛 10000 次硬币，概率会升到 97% ）。这是因为**大数定律**（*law of large numbers*）：随着你持续抛硬币，正面的比例会不断接近抛出正面的概率（ 51% ）。图 7-3 展示了 10 种有偏差的硬币抛掷。你能看到，随着抛硬币次数增加，正面的比例越来越接近 51% 。最终这十种都很接近 51% ，被认为大于 50% 。

![3](./images/chap7/7-3.png)

类似地，假设你创建了一个集成，拥有 1000 个单独运行时正确率只有 51% 的分类器（只比胡乱猜测好一点）。如果你预测多数投票类，精度可能会提高到 75% ！不过，这只有当所有的分类器都完全独立、误差互不关联时才成立，显然本例中不可能实现，因为它们是在相同的数据集上训练的。它们很可能会犯同一种错，所以多数票可能会投给错误的类别，从而降低集成的精度。

> **提示**
> 集成方法在预测器彼此独立时工作得最好。一种得到多样化分类器的方法是用不同的算法训练它们。这样就增加了它们犯不同的错误的概率，提高集成的精度。

下面的代码创建并训练了 Scikit-Learn 中的投票分类器，它由三个不同的分类器组成（训练集为卫星数据集，在第五章中有介绍）：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
log_clf	= LogisticRegression()
rnd_clf	= RandomForestClassifier()
svm_clf	= SVC()
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
voting_clf.fit(X_train,	y_train)
```

 来看看每个分类器在测试集上的精度：
 
 ```python
>>> from sklearn.metrics import accuracy_score
>>> for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
...	clf.fit(X_train, y_train)
...	y_pred = clf.predict(X_test)
...	print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
...
LogisticRegression 0.864
RandomForestClassifier 0.872
SVC 0.888
VotingClassifier 0.896
```

现在你明白了！投票分类器的表现优于所有单独的分类器。

如果所有的分类器都能评估类别概率（即都有`predict_proba()`方法），你就可以让 Scikit-Learn 将类预测为概率最高的类，平均在所有单独分类器上。这被称为**软投票**（*soft voting*）。它通常能比硬投票表现得更好，因为它给予高置信投票更多的权重。你只需将`voting="hard"`换为`voting="soft"`，确保所有的分类器都能评估类别概率。这不是`SVC`类默认的选择，所以你需要把它的超参数`probability`设置为`True`（这样`SVC`类就会使用交叉验证来评估类别概率，降低训练速度，增加`predict_proba()`方法）。如果你把之前的代码改为软投票，你会发现投票分类器的精度到达了 91% ！

## Bagging 和 Pasting

一种得到多样化分类器的方法是用不同的训练算法，正如之前讨论的。另一种方法是给每个预测器都使用相同的训练算法，但在训练集的随机子集上训练它们。有放回采样的方法被称为 [bagging](http://statistics.berkeley.edu/sites/default/files/tech-reports/421.pdf) （*bootstrap aggregating*的缩写）。无放回采样的方法被称为 [pasting](https://link.springer.com/article/10.1023/A:1007563306331) 。

换言之， bagging 和 pasting 都允许训练实例在多个预测器中被多次采样，但是只有 bagging 允许在同一个预测器中多次采样训练实例。这种采样和训练过程如图 7-4 所示。

![4](./images/chap7/7-4.png)

一旦所有的预测器都完成训练，集成就能简单合并所有预测，为新实例做出预测。聚合函数对于分类通常是统计模式（*statistical mode*）（就像硬投票分类器一样，最频繁的预测），对于回归则是平均数。每个单独的预测器都比在原始训练集上训练有更高的偏差，但是聚合函数同时降低了偏差和方差。网络的结果是，在同一原始训练上训练的集合和单独的预测器相比，偏差相似，但有更低的方差。

在图 7-4 中可以看到，预测器能通过不同的 CPU 或服务器并行训练。类似地，预测也能并行进行。这就是 bagging 和 pasting 如此流行的原因之一：它们扩展得很好。

## Scikit-Learn 中的 Bagging 和 Pasting

Scikit-Learn 为 Bagging 和 Pasting 提供了简单的 API `BaggingClassifier`类（或`BaggingRegressor`类）。下面的代码训练了 500 棵决策树分类器组成的集成，每棵决策树都在 100 个有放回随机抽样的训练实例上进行了训练（这是 bagging 的一个例子，如果你想使用 pasting，只需设置`bootstrap=False`）。`n_jobs`参数告诉 Scikit-Learn 用于训练和预测的CPU核心数量（ -1 就是使用全部核心）。

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
	DecisionTreeClassifier(), n_estimators=500,
	max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```

> **笔记**
> 如果基分类器能评估类别概率（即有`predict_proba()`方法），`BaggingClassifier`会自动应用软投票而不是硬投票，决策树分类器就是如此。

图 7-5 比较了单棵决策树和 500 棵决策树组成的集成的决策边界，两者都在卫星数据集上进行训练。如你所见，集成的预测可能会比单棵决策树泛化得好：集成有可比的偏差，但是偏差更小（它在训练集上的错误数量大致相同，但决策边界更有规律）。

![5](./images/chap7/7-5.png)

自助法
