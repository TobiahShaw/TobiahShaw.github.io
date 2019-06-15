
# 随机森林和 Extra-Trees

## 随机森林

我们对使用决策树随机取样的集成学习有个形象的名字--随机森林。

scikit-learn 中封装的随机森林，在决策树的节点划分上，在**随机的特征子集**上寻找最优划分特征。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
```

```python
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=666)
```

```python
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()
```

![png](..\assets\img\EnsembleLearning\5_output_3_0.png)

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, random_state=666, oob_score=True)
rf_clf.fit(X, y)
```

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,
                oob_score=True, random_state=666, verbose=0, warm_start=False)

```python
rf_clf.oob_score_
```

0.892

**自定义决策树某些参数**

```python
rf_clf2 = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16
                                , random_state=666, oob_score=True)
rf_clf2.fit(X, y)
rf_clf2.oob_score_
```

0.906

## Extra-Trees

在决策树的节点划分上，使用随机的特征和随机的阈值。

随机性更加极端。

提供了额外的随机性，一直过拟合，但增大了 bias 。

更快的训练速度。

```python
from sklearn.ensemble import ExtraTreesClassifier

et_clf = ExtraTreesClassifier(n_estimators=500, bootstrap=True
                              , random_state=666, oob_score=True)
et_clf.fit(X, y)
```

ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='gini',
               max_depth=None, max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,
               oob_score=True, random_state=666, verbose=0, warm_start=False)

```python
et_clf.oob_score_
```

0.892

## 集成学习解决回归问题

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
```
