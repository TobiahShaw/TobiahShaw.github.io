
# MNIST

```python
import numpy as np
from sklearn.datasets import fetch_mldata
```

```python
mnist = fetch_mldata("MNIST original", data_home="./datasets")
```

```python
mnist
```

{'DESCR': 'mldata.org dataset: mnist-original',
     'COL_NAMES': ['label', 'data'],
     'target': array([0., 0., 0., ..., 9., 9., 9.]),
     'data': array([[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)}

```python
X, y = mnist.data, mnist.target
```

```python
X.shape
```

(70000, 784)

```python
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)
```

```python
X_train.shape
```

(60000, 784)

```python
y_train.shape
```

(60000,)

```python
X_test.shape
```

(10000, 784)

```python
y_test.shape
```

(10000,)

```python
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
%time knn_clf.fit(X_train, y_train)
```

Wall time: 29.4 s

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')

```python
%time knn_clf.score(X_test, y_test)
```

Wall time: 10min

0.9688

## PCA 降维

```python
from sklearn.decomposition import PCA

pca = PCA(0.9)
pca.fit(X_train)
X_train_reductopn = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
```

```python
X_train_reductopn.shape
```

(60000, 87)

```python
knn_clf2 = KNeighborsClassifier()
%time knn_clf2.fit(X_train_reductopn, y_train)
```

Wall time: 484 ms

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')

```python
%time knn_clf2.score(X_test_reduction, y_test)
```

Wall time: 1min 16s

0.9728

可以看到，经过PCA处理后，训练和预测所需的时间大量减少，并且准确的得到了上升，这是因为PCA在降维的同时，进行了降噪。不做归一化的原因是因为数据中点都代表了像素点的亮度，在同一量纲下，无需进行数据归一化
