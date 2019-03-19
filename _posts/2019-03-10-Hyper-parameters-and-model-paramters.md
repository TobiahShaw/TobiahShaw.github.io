<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# 超参数和模型参数

上个小节我们讲了网格搜索和超参数，这一节我们就研究一下看kNN具体的超参数的意义。

同时在这一节中我们不适用网格搜索，来感受一下网格搜索内部到底做了什么。

- 超参数：在算法运行前需要决定的参数

- 模型参数：算法过程中学习的参数

kNN算法没有模型参数

kNN算法中的k是典型的超参数

```python
# 导包
import numpy as np
from sklearn import datasets

# 读数据
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 训练测试分离
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=666)

# kNN模型训练以及score

from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(X_train, y_train)
kNN.score(X_test, y_test)
```

0.9888888888888889

## 寻找最好的k

```python
best_score = 0.0
best_k = -1
for k in range(1, 11):
    kNN = KNeighborsClassifier(n_neighbors=k)
    kNN.fit(X_train, y_train)
    score = kNN.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_k = k
print("best_k =", best_k)
print("best_score =", best_score)
```

best_k = 4
best_score = 0.9916666666666667

## 考虑距离？不考虑距离？

```python
best_method = ""
best_score = 0.0
best_k = -1
for method in ["uniform", "distance"]:
    for k in range(1, 11):
        kNN = KNeighborsClassifier(n_neighbors=k, weights=method)
        kNN.fit(X_train, y_train)
        score = kNN.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_k = k
            best_method = method

print("best_method =", best_method)
print("best_k =", best_k)
print("best_score =", best_score)
```

best_method = uniform
best_k = 4
best_score = 0.9916666666666667

## 距离的定义

- 曼哈顿距离

$$\sum_{i=1}^{n}\ |X_i^{(a)} - X_i^{(b)}|$$
或者表示为
$$(\sum_{i=1}^{n}\ |X_i^{(a)} - X_i^{(b)}|^1)^{\frac{1}{1}}$$

- 欧拉距离

$$\sqrt{\sum_{i=1}^{n}\ (X_i^{(a)} - X_i^{(b)})^2}$$
或者表示为
$$(\sum_{i=1}^{n}\ |X_i^{(a)} - X_i^{(b)}|^2)^{\frac{1}{2}}$$

- 明可夫斯基距离

$$(\sum_{i=1}^{n}\ |X_i^{(a)} - X_i^{(b)}|^p)^{\frac{1}{p}}$$

此时：

曼哈顿距离可看做 p=1 时的明可夫斯基距离

欧拉距离可看做 p=2 时的明可夫斯基距离

此时我们获得了一个超参数p

## 是否考虑明可夫斯基距离

```python
%%time
best_p = -1
best_method = ""
best_score = 0.0
best_k = -1
for method in ["uniform", "distance"]:
    for p in range(1, 5):
        for k in range(1, 11):
            kNN = KNeighborsClassifier(n_neighbors=k, weights=method, p=p)
            kNN.fit(X_train, y_train)
            score = kNN.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_k = k
                best_method = method
                best_p = p

print("best_p =", best_p)
print("best_method =", best_method)
print("best_k =", best_k)
print("best_score =", best_score)
```

best_p = 2
best_method = uniform
best_k = 4
best_score = 0.9916666666666667
Wall time: 29.7 s
