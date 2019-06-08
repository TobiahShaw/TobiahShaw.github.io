
# 什么是集成学习

集成学习是使用一系列学习器进行学习，并使用某种规则把各个学习结果进行整合从而获得比单个学习器更好的学习效果的一种机器学习方法。

类似于生活中的专家会诊等。

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

![png](..\assets\img\EnsembleLearning\1_output_3_0.png)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

## 三种算法分别训练

```python
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(solver='lbfgs')
log_clf.fit(X_train, y_train)
log_clf.score(X_test, y_test)
```

0.864

```python
from sklearn.svm import SVC

svc_clf = SVC(gamma=1.0)
svc_clf.fit(X_train, y_train)
svc_clf.score(X_test, y_test)
```

0.888

```python
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_clf.score(X_test, y_test)
```

0.888

## 进行投票

```python
y_log_predict = log_clf.predict(X_test)
y_svc_predict = svc_clf.predict(X_test)
y_dt_predict  = dt_clf.predict(X_test)
```

```python
y_predict = np.array((y_log_predict + y_svc_predict + y_dt_predict)>=2, dtype=int)
```

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict)
```

0.912

## scikit-learn 中的 VotingClassifier

```python
from sklearn.ensemble import VotingClassifier
```

```python
v_clf = VotingClassifier(estimators=[
    ("log", LogisticRegression(solver='lbfgs')),
    ("svc", SVC(gamma=1.0)),
    ("dt", DecisionTreeClassifier())
], voting='hard')
v_clf.fit(X_train, y_train)
v_clf.score(X_test, y_test)
```

0.912
