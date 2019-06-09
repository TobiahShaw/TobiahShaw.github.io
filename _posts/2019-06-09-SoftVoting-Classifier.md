
# Soft Voting Classifier

不同于少数服从多数的 hard voting，更合理的投票，应该更有权值。

假设有五个模型进行投票：

模型1：A:99%,B:1%;

模型2：A:49%,B:51%;

模型3：A:40%,B:60%;

模型4：A:90%,B:10%;

模型5：A:30%,B:70%;

则按照上一小节，hard voting 的方式 A：两票， B：三票，则**应该分为 B 类**，但是可以看到模型中投票给 A 的模型，都比较确定分为 A 类，而投给 B 类的三个模型中，模型2和模型3的分类都是不太确定的。

我们考虑分类概率的情况：

A: (0.99 + 0.49 + 0.40 + 0.90 + 0.30) / 5 = 0.616
B: (0.01 + 0.51 + 0.60 + 0.10 + 0.70) / 5 = 0.384

**则此时应分为 A 类**

后者计算方式就叫做 soft voting

**前提**

**要求集合中的每一个模型都能估计概率（predict_proba）**

可用算法：逻辑回归（基于概率）、kNN（加权值投票）、决策树（类似kNN，在叶子节点加权值投票）、支撑向量机（使用另外的算法，会占用更多计算资源，牺牲计算时间）

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

![png](..\assets\img\EnsembleLearning\2_output_3_0.png)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
```

## hard voting

```python
hv_clf = VotingClassifier(estimators=[
    ("log", LogisticRegression(solver='lbfgs')),
    ("svc", SVC(gamma=1.0)),
    ("dt", DecisionTreeClassifier(random_state=666))
], voting='hard')
```

```python
hv_clf.fit(X_train, y_train)
hv_clf.score(X_test, y_test)
```

0.896

## soft voting

```python
sv_clf = VotingClassifier(estimators=[
    ("log", LogisticRegression(solver='lbfgs')),
    ("svc", SVC(gamma=1.0,probability=True)),
    ("dt", DecisionTreeClassifier(random_state=666))
], voting='soft')
```

```python
sv_clf.fit(X_train, y_train)
sv_clf.score(X_test, y_test)
```

0.904
