
# CART

Classification And Regression Tree

**根据某一个维度d和某一个阈值v进行二分**

**scikit-learn 的决策树实现是 CART**

其他实现方式：ID3，C4.5，C5.0

预测复杂度：O(log m)

训练复杂度：O(n * m * log m)

缺点：非常容易产生过拟合（事实上非参数学习都是这样）

解决方案：

减枝：降低复杂度，解决过拟合

# 决策树的超参数

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
```

```python
X, y = datasets.make_moons(noise=0.25, random_state=666)
```

```python
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```

![png](..\assets\img\DecisionTree\6_output_4_0.png)

```python
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X, y)
```

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')

```python
def plot_decision_boundary(model, axis):

    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)
```

```python
plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.show()
```

![png](..\assets\img\DecisionTree\6_output_7_0.png)

```python
dt_clf2 = DecisionTreeClassifier(max_depth=2)
dt_clf2.fit(X, y)
```

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')

```python
plot_decision_boundary(dt_clf2, axis=[-1.5, 2.5, -1, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.show()
```

![png](..\assets\img\DecisionTree\6_output_9_0.png)

```python
dt_clf3 = DecisionTreeClassifier(min_samples_split=10)
dt_clf3.fit(X, y)

plot_decision_boundary(dt_clf3, axis=[-1.5, 2.5, -1, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.show()
```

![png](..\assets\img\DecisionTree\6_output_10_0.png)

```python
dt_clf4 = DecisionTreeClassifier(min_samples_leaf=6)
dt_clf4.fit(X, y)

plot_decision_boundary(dt_clf4, axis=[-1.5, 2.5, -1, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.show()
```

![png](..\assets\img\DecisionTree\6_output_11_0.png)

```python
dt_clf5 = DecisionTreeClassifier(max_leaf_nodes=4)
dt_clf5.fit(X, y)

plot_decision_boundary(dt_clf5, axis=[-1.5, 2.5, -1, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
plt.show()
```

![png](..\assets\img\DecisionTree\6_output_12_0.png)

## 常用超参数

min_samples_split

min_samples_leaf

min_weight_fraction_leaf

max_depth

max_leaf_nodes

min_features
