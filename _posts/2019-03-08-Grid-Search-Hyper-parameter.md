
# 网格搜索

上一小节，我们介绍了分类准确度的概念，既然有了衡量尺度，我们就要尽可能优化模型，已最大化分类准确度。

对于提升模型性能，我们通常的做法是调整超参数。

在kNN算法中，我们常用的超参数就是k近邻的k值。

还有一个就是是否使用权重，即不同的样本点对计算距离时存在权重。

另外还有一个就是距离的定义：

- 曼哈顿距离
- 欧拉距离
- 明可夫斯基距离

我们先随便调整一下超参数看下

```python
import numpy as np
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target
```

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
```

```python
from sklearn.neighbors import KNeighborsClassifier

sk_kNN_clf = KNeighborsClassifier(n_neighbors=4, weights="uniform")
sk_kNN_clf.fit(X_train, y_train)
sk_kNN_clf.score(X_test, y_test)
```

0.9916666666666667

## Grid Search

显而易见的是一个个调整超参数是个非常繁琐的工作，这里我们引入网格搜索的概念，使用网格搜索对我们指定的参数进行训练，并且得到个最佳模型。

```python
# 定义搜索的参数
param_grid = [
    {
        "weights": ["uniform"],
        "n_neighbors": [i for i in range(1, 11)]
    },
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1, 11)],
        "p": [i for i in range(1, 6)]
    }
]
```

```python
knn_clf = KNeighborsClassifier()
```

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(knn_clf, param_grid)
```

```python
%%time
grid_search.fit(X_train, y_train)
```

输出：

    Wall time: 2min 51s

    GridSearchCV(cv=None, error_score='raise',
           estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform'),
           fit_params=None, iid=True, n_jobs=1,
           param_grid=[{'weights': ['uniform'], 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, {'weights': ['distance'], 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'p': [1, 2, 3, 4, 5]}],
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)


最好的模型

```python
grid_search.best_estimator_
```

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=3, p=3,
               weights='distance')


最好的模型的score（CV - 交叉验证）

```python
grid_search.best_score_
```

0.9853862212943633


最好的超参数

```python
grid_search.best_params_
```

{'n_neighbors': 3, 'p': 3, 'weights': 'distance'}

```python
knn_clf = grid_search.best_estimator_
knn_clf.score(X_test, y_test)
```

0.9833333333333333


多核操作

```python
grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=2, verbose=3)
```

```python
%%time
grid_search.fit(X_train, y_train)
```

    Fitting 3 folds for each of 60 candidates, totalling 180 fits

    [Parallel(n_jobs=2)]: Done  28 tasks      | elapsed:   27.6s
    [Parallel(n_jobs=2)]: Done 124 tasks      | elapsed:  1.3min


    Wall time: 1min 58s


    [Parallel(n_jobs=2)]: Done 180 out of 180 | elapsed:  2.0min finished


    GridSearchCV(cv=None, error_score='raise',
           estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=3, p=3,
               weights='distance'),
           fit_params=None, iid=True, n_jobs=2,
           param_grid=[{'weights': ['uniform'], 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, {'weights': ['distance'], 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'p': [1, 2, 3, 4, 5]}],
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=3)

## 更多距离的定义

- 向量空间余弦相似度 Cosine Similarity
- 调整余弦相似度 Adjust Cosine Similarity
- 皮尔森相关系数 Pearson Correlation Coefficient
- Jaccard 相似系数 Jaccard Coefficient
