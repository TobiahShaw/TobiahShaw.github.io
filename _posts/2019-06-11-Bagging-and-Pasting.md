
# 集成学习中的取样

之前的做法中，是使用不同的机器学习方法进行集成。

虽然有很多机器学习方法，但是从投票的角度看，任然不够多。

所以，我们要创建更多的子模型！集成更多的子模型的意见。

子模型之间不能一致！子模型之间要有差异性。

**如何创造差异性？**

**每个子模型只看样本数据的一部分。**

每一个子模型不需要太高的准确率。

取样方式：

- 放回取样 (Bagging)

- 不放回取样 (Pasting)

其中 Bagging 更常用，因为

- 可获得更多样本

- 不强烈的依赖随机

Bagging 在统计学中也被称为 bootstrap

在集成学习中，我们经常使用**决策树**的方法，因为决策树更容易出现差异化的模型，这是集成学习希望看到的。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
```

```python
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
```

```python
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()
```

![png](..\assets\img\EnsembleLearning\3_output_3_0.png)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

## 使用bagging

### 使用决策树

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

dt_bagging_clf = BaggingClassifier(DecisionTreeClassifier()
                                , n_estimators=500, max_samples=100, bootstrap=True)

dt_bagging_clf.fit(X_train, y_train)
dt_bagging_clf.score(X_test, y_test)
```

0.912

```python
dt_bagging_clf2 = BaggingClassifier(DecisionTreeClassifier()
                                , n_estimators=5000, max_samples=100, bootstrap=True)

dt_bagging_clf2.fit(X_train, y_train)
dt_bagging_clf2.score(X_test, y_test)
```

0.92

### 使用支撑向量机

```python
from sklearn.svm import SVC

svc_bagging_clf = BaggingClassifier(SVC(gamma=1.0)
                                , n_estimators=500, max_samples=100, bootstrap=True)

svc_bagging_clf.fit(X_train, y_train)
svc_bagging_clf.score(X_test, y_test)
```

0.888

### 使用逻辑回归

```python
from sklearn.linear_model import LogisticRegression

log_bagging_clf = BaggingClassifier(LogisticRegression(solver='lbfgs')
                                , n_estimators=500, max_samples=100, bootstrap=True)

log_bagging_clf.fit(X_train, y_train)
log_bagging_clf.score(X_test, y_test)
```

0.848

### 使用kNN

```python
from sklearn.neighbors import KNeighborsClassifier

knn_bagging_clf = BaggingClassifier(KNeighborsClassifier()
                                , n_estimators=500, max_samples=100, bootstrap=True)

knn_bagging_clf.fit(X_train, y_train)
knn_bagging_clf.score(X_test, y_test)
```

0.88
