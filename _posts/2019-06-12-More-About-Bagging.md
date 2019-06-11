
# 更多和 Bagging 相关的讨论

## OOB Out-of-Bag

放回取样导致一部分样本很有可能没有取到。

平均大约有 37% 的样本没有取到（这部分样本被称为 OOB）。

不使用 train_test_split 而使用这部分没有取到的样本做测试或验证（oob_score_）。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
```

```python
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
```

```python
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```

![png](..\assets\img\EnsembleLearning\4_output_3_0.png)

### 使用 OOB

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
```

```python
bagging_clf = BaggingClassifier(DecisionTreeClassifier()
                                , n_estimators=500, max_samples=100, bootstrap=True
                                , oob_score=True)
bagging_clf.fit(X, y)
```

BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'),
             bootstrap=True, bootstrap_features=False, max_features=1.0,
             max_samples=100, n_estimators=500, n_jobs=1, oob_score=True,
             random_state=None, verbose=0, warm_start=False)

```python
bagging_clf.oob_score_
```

0.914

## 并行化处理

Bagging 的思路极易进行并行化处理，因为独立的训练子模型

使用参数 n_jobs

### 使用n_jobs

```python
%%time
bagging_clf = BaggingClassifier(DecisionTreeClassifier()
                                , n_estimators=5000, max_samples=100, bootstrap=True
                                , oob_score=True)
bagging_clf.fit(X, y)
```

Wall time: 5.06 s

```python
%%time
bagging_clf = BaggingClassifier(DecisionTreeClassifier()
                                , n_estimators=5000, max_samples=100, bootstrap=True
                                , oob_score=True, n_jobs=-1)
bagging_clf.fit(X, y)
```

Wall time: 3.26 s

## 子模型产生差异化的方式

- 使用更小的样本数据集，即针对样本进行随机取样
- 针对特征进行随机取样（样本数据特征多） （random subspaces）
- 同事针对样本，有针对特征进行随机取样 （random patches）

### random subspaces

```python
random_subpaces_clf = BaggingClassifier(DecisionTreeClassifier()
                                , n_estimators=500, max_samples=500, bootstrap=True
                                , oob_score=True, max_features=1
                                , bootstrap_features=True)
random_subpaces_clf.fit(X, y)
random_subpaces_clf.oob_score_
```

0.836

### random patches

```python
random_patches_clf = BaggingClassifier(DecisionTreeClassifier()
                                , n_estimators=500, max_samples=100, bootstrap=True
                                , oob_score=True, max_features=1
                                , bootstrap_features=True)
random_patches_clf.fit(X, y)
random_patches_clf.oob_score_
```

0.858
