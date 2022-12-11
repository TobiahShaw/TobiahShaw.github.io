---
layout: post
title: The Accuracy of Classified
tag: ML
---

# 分类准确性

既然我们有了分类算法，我们就需要一个指标来衡量我们算法训练出来模型的好坏。

这里我们引入分类准确度的概念，即分类正确样本数占总样本数的比例，是一个0到1之间的值。

在本节我们顺便学习一下，如何使用scikit-learn中内置数据集的使用方法。

## 实现自己的 accuracy

```python
# 首先导入我们所需要的包
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets

# 手写数字数据集读取
digits = datasets.load_digits()
# 查看数据级的内容
digits.keys()
```

```terminal
dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])
```

```python
# 查看数据集的描述
print(digits.DESCR)
```

```terminal
    Optical Recognition of Handwritten Digits Data Set
    ===================================================

    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 5620
        :Number of Attributes: 64
        :Attribute Information: 8x8 image of integer pixels in the range 0..16.
        :Missing Attribute Values: None
        :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
        :Date: July; 1998

    This is a copy of the test set of the UCI ML hand-written digits datasets
    http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

    The data set contains images of hand-written digits: 10 classes where
    each class refers to a digit.

    Preprocessing programs made available by NIST were used to extract
    normalized bitmaps of handwritten digits from a preprinted form. From a
    total of 43 people, 30 contributed to the training set and different 13
    to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of
    4x4 and the number of on pixels are counted in each block. This generates
    an input matrix of 8x8 where each element is an integer in the range
    0..16. This reduces dimensionality and gives invariance to small
    distortions.

    For info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G.
    T. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.
    L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,
    1994.

    References
    ----------
      - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their
        Applications to Handwritten Digit Recognition, MSc Thesis, Institute of
        Graduate Studies in Science and Engineering, Bogazici University.
      - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.
      - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.
        Linear dimensionalityreduction using relevance weighted LDA. School of
        Electrical and Electronic Engineering Nanyang Technological University.
        2005.
      - Claudio Gentile. A New Approximate Maximal Margin Classification
        Algorithm. NIPS. 2000.
```

```python
# 数据集中的data是数据的样本
X = digits.data
X.shape
```

```terminal
(1797, 64)
```

```python
# 数据集中的target是数据的标签
y = digits.target
y.shape
```

```terminal
(1797,)
```

```python
# 数据标签的值，这里是手写数字，所以分别有0到9，10个数字为标签
digits.target_names
```

```terminal
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

```python
# 查看一下前一百个数据的标签
y[:100]
```

```terminal
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
        2, 3, 4, 5, 6, 7, 8, 9, 0, 9, 5, 5, 6, 5, 0, 9, 8, 9, 8, 4, 1, 7,
        7, 3, 5, 1, 0, 0, 2, 2, 7, 8, 2, 0, 1, 2, 6, 3, 3, 7, 3, 3, 4, 6,
        6, 6, 4, 9, 1, 5, 0, 9, 5, 2, 8, 2, 0, 0, 1, 7, 6, 3, 2, 1, 7, 4,
        6, 3, 1, 3, 9, 1, 7, 6, 8, 4, 3, 1])
```

```python
# 查看前十个数据的样本，可以看到每个数据样本是个长度为64向量，其实是8*8举证拉平得到的，
# 每个值代表点的灰阶度，0到16之间
X[:10]
```

```terminal
    array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,
            15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,
            12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,
             0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,
            10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.],
           [ 0.,  0.,  0., 12., 13.,  5.,  0.,  0.,  0.,  0.,  0., 11., 16.,
             9.,  0.,  0.,  0.,  0.,  3., 15., 16.,  6.,  0.,  0.,  0.,  7.,
            15., 16., 16.,  2.,  0.,  0.,  0.,  0.,  1., 16., 16.,  3.,  0.,
             0.,  0.,  0.,  1., 16., 16.,  6.,  0.,  0.,  0.,  0.,  1., 16.,
            16.,  6.,  0.,  0.,  0.,  0.,  0., 11., 16., 10.,  0.,  0.],
           [ 0.,  0.,  0.,  4., 15., 12.,  0.,  0.,  0.,  0.,  3., 16., 15.,
            14.,  0.,  0.,  0.,  0.,  8., 13.,  8., 16.,  0.,  0.,  0.,  0.,
             1.,  6., 15., 11.,  0.,  0.,  0.,  1.,  8., 13., 15.,  1.,  0.,
             0.,  0.,  9., 16., 16.,  5.,  0.,  0.,  0.,  0.,  3., 13., 16.,
            16., 11.,  5.,  0.,  0.,  0.,  0.,  3., 11., 16.,  9.,  0.],
           [ 0.,  0.,  7., 15., 13.,  1.,  0.,  0.,  0.,  8., 13.,  6., 15.,
             4.,  0.,  0.,  0.,  2.,  1., 13., 13.,  0.,  0.,  0.,  0.,  0.,
             2., 15., 11.,  1.,  0.,  0.,  0.,  0.,  0.,  1., 12., 12.,  1.,
             0.,  0.,  0.,  0.,  0.,  1., 10.,  8.,  0.,  0.,  0.,  8.,  4.,
             5., 14.,  9.,  0.,  0.,  0.,  7., 13., 13.,  9.,  0.,  0.],
           [ 0.,  0.,  0.,  1., 11.,  0.,  0.,  0.,  0.,  0.,  0.,  7.,  8.,
             0.,  0.,  0.,  0.,  0.,  1., 13.,  6.,  2.,  2.,  0.,  0.,  0.,
             7., 15.,  0.,  9.,  8.,  0.,  0.,  5., 16., 10.,  0., 16.,  6.,
             0.,  0.,  4., 15., 16., 13., 16.,  1.,  0.,  0.,  0.,  0.,  3.,
            15., 10.,  0.,  0.,  0.,  0.,  0.,  2., 16.,  4.,  0.,  0.],
           [ 0.,  0., 12., 10.,  0.,  0.,  0.,  0.,  0.,  0., 14., 16., 16.,
            14.,  0.,  0.,  0.,  0., 13., 16., 15., 10.,  1.,  0.,  0.,  0.,
            11., 16., 16.,  7.,  0.,  0.,  0.,  0.,  0.,  4.,  7., 16.,  7.,
             0.,  0.,  0.,  0.,  0.,  4., 16.,  9.,  0.,  0.,  0.,  5.,  4.,
            12., 16.,  4.,  0.,  0.,  0.,  9., 16., 16., 10.,  0.,  0.],
           [ 0.,  0.,  0., 12., 13.,  0.,  0.,  0.,  0.,  0.,  5., 16.,  8.,
             0.,  0.,  0.,  0.,  0., 13., 16.,  3.,  0.,  0.,  0.,  0.,  0.,
            14., 13.,  0.,  0.,  0.,  0.,  0.,  0., 15., 12.,  7.,  2.,  0.,
             0.,  0.,  0., 13., 16., 13., 16.,  3.,  0.,  0.,  0.,  7., 16.,
            11., 15.,  8.,  0.,  0.,  0.,  1.,  9., 15., 11.,  3.,  0.],
           [ 0.,  0.,  7.,  8., 13., 16., 15.,  1.,  0.,  0.,  7.,  7.,  4.,
            11., 12.,  0.,  0.,  0.,  0.,  0.,  8., 13.,  1.,  0.,  0.,  4.,
             8.,  8., 15., 15.,  6.,  0.,  0.,  2., 11., 15., 15.,  4.,  0.,
             0.,  0.,  0.,  0., 16.,  5.,  0.,  0.,  0.,  0.,  0.,  9., 15.,
             1.,  0.,  0.,  0.,  0.,  0., 13.,  5.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  9., 14.,  8.,  1.,  0.,  0.,  0.,  0., 12., 14., 14.,
            12.,  0.,  0.,  0.,  0.,  9., 10.,  0., 15.,  4.,  0.,  0.,  0.,
             3., 16., 12., 14.,  2.,  0.,  0.,  0.,  4., 16., 16.,  2.,  0.,
             0.,  0.,  3., 16.,  8., 10., 13.,  2.,  0.,  0.,  1., 15.,  1.,
             3., 16.,  8.,  0.,  0.,  0., 11., 16., 15., 11.,  1.,  0.],
           [ 0.,  0., 11., 12.,  0.,  0.,  0.,  0.,  0.,  2., 16., 16., 16.,
            13.,  0.,  0.,  0.,  3., 16., 12., 10., 14.,  0.,  0.,  0.,  1.,
            16.,  1., 12., 15.,  0.,  0.,  0.,  0., 13., 16.,  9., 15.,  2.,
             0.,  0.,  0.,  0.,  3.,  0.,  9., 11.,  0.,  0.,  0.,  0.,  0.,
             9., 15.,  4.,  0.,  0.,  0.,  9., 12., 13.,  3.,  0.,  0.]])
```

```python
# 随便去个阉割版拿出来
some_digits = X[666]
y[666]
```

```terminal
0
```

```python
# 将向量以灰度图的形式打印出来
some_digits_image = some_digits.reshape(8,8)
plt.imshow(some_digits_image, cmap = matplotlib.cm.binary)
plt.show()
```

![png](../assets/img/kNN/output_10_0.png)

在这里我们加载上两个课上封装的kNN和train test split，进行训练

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
my_kNN_clf = kNNClassifier(10)
my_kNN_clf.fit(X_train=X_train, y_train=y_train)
```

```terminal
kNN(k=10)
```

使用训练好的样本预测测试集的结果，并根据测试集的样本算出准确度

```python
y_predict = my_kNN_clf.predict(X_predict=X_test)
accuracy = sum(y_predict == y_test) / len(y_test)
accuracy
```

```terminal
0.9860724233983287
```

我们把这个方法封装一下：

```python
def accuracy_score(y_true, y_predict):
    assert y_true.shape == y_predict.shape,\
        "the size of y_true must be equal to the size of y_predict"
    return (sum(y_true == y_predict) / len(y_true))
```

使用封装的方法试一下

```python
accuracy = accuracy_score(y_true=y_test, y_predict=y_predict)
accuracy
```

```terminal
0.9860724233983287
```

完善kNN算法，增加score方法

```python
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_true=y_test,y_predict=y_predict)
```

使用封装好的 kNN score 方法

```python
my_kNN_clf.score(X_test=X_test, y_test=y_test)
```

```terminal
0.9860724233983287
```

## scikit-lean 中的 accuracy_score

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
```

```terminal
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=3, p=2,
               weights='uniform')
```

```python
y_predict = knn_clf.predict(X=X_test)
```

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_true=y_test, y_pred=y_predict)
```

```terminal
0.9972222222222222
```

```python
knn_clf.score(X=X_test, y=y_test)
```

```terminal
0.9972222222222222
```

可以看到我们的实现，其实是仿照的scikit-learn的风格来的，这样做的话其实是有一些好处的，可以让我们的封装兼容scikit-learn的一些方法，以后我们也会这么做。
