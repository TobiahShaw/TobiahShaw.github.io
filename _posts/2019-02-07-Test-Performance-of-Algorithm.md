---
layout: post
title: How to test the performance of algorithm
tag: ML
---

# 测试算法效率

在这一节，我们需要实现一种方法，使得我们可以测试我们算法的效果。

为了达到测试的效果，我们需要使用训练中从未使用的数据来检查模型对未知的数据，预测结果的好坏。

由此，我们可以将我们所有的数据数据随机的分为两部分，一部分用于训练，一部分用于测试，即

`train test split`

先加载我们需要的，用于测试的鸢尾花数据集

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target
```

通过 shape 查看数据的形状

```python
X.shape
```

```terminal
(150, 4)
```

```python
y.shape
```

```terminal
(150,)
```

## 实现 train test split 算法

首先我们将数据打乱，因为样本 X 和标签 y 之间根据索引一一对应，我们不能将两个数据各自打乱，一面产生错误数据。

我们可以先生成乱序的索引，然后将 X 和 y 按照这个乱序的索引排序。

- 生成随机索引

```python
shuffle_indexes = np.random.permutation(len(X))
```

- 确定测试数据集的比例，分出测试数据集索引

```python
test_ratio = 0.2
test_size = int(len(X) * test_ratio)
```

可以查看一下测试集的大小

```python
test_size
```

```terminal
30
```

- 生成训练集和测试集的索引

```python
test_indexes = shuffle_indexes[:test_size]
train_indexes = shuffle_indexes[test_size:]
```

- 生成训练数据集

```python
X_train = X[train_indexes]
y_train = y[train_indexes]
```

- 生成测试数据集

```python
X_test = X[test_indexes]
y_test = y[test_indexes]
```

查看一下训练集样本和标签的大小

```python
print(X_train.shape)
print(y_train.shape)
```

```terminal
(120, 4)
(120,)
```

查看一下测试集样本和标签的大小

```python
print(X_test.shape)
print(y_test.shape)
```

```terminal
(30, 4)
(30,)
```

## 封装算法

```python
def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0],\
        "the size of X must equal to the size of x"
    assert 0.0 <= test_ratio <= 1.0,\
        "test_ratio must valid"
    if seed:
        np.random.seed(seed)
    shuffle_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test ,y_train, y_test
```

## 使用封装好的算法

```python
X_train, X_test ,y_train, y_test = train_test_split(X,y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

```terminal
(120, 4)
(120,)
(30, 4)
(30,)
```

## 将结果用于上一节 kNN 算法

```python
my_kNN_classifier = kNNClassifier(3)
my_kNN_classifier.fit(X_train=X_train, y_train=y_train)
```

```terminal
kNN(k=3)
```

```python
y_predict = my_kNN_classifier.predict(X_predict=X_test)

y_predict
```

```terminal
array([2, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 0, 0, 2, 1, 0, 0, 2, 0, 1, 2, 1,
           0, 1, 2, 0, 0, 2, 1, 2])
```

```python
y_test
```

```terminal
array([2, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 0, 0, 2, 1, 0, 0, 2, 0, 1, 2, 2,
           0, 1, 2, 0, 0, 2, 1, 2])
```

通过计算正确预测的比例来衡量效率

```python
accuracy = sum(y_predict == y_test) / len(y_test)

accuracy
```

```terminal
0.9666666666666667
```

## sklearn 中的 train_test_split

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

```terminal
(120, 4)
(120,)
(30, 4)
(30,)
```