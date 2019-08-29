# 向量化运算和梯度下降中归一化

$$\nabla J(\theta) = 
\frac{2}{m} \cdot \begin{pmatrix}
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \\\\
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_1 \\\\
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_2 \\\\
\ldots \\\\
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_n
\end{pmatrix} =
\frac{2}{m} \cdot \begin{pmatrix}
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_0\\\\
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_1 \\\\
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_2 \\\\
\ldots \\\\
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_n
\end{pmatrix}$$

$$X^{(i)}_0  \equiv 1$$

$$\nabla J(\theta) = \frac{2}{m}\cdot((X_b \theta - y)^T \cdot X_b)^T = \frac{2}{m}\cdot X_b^T \cdot (X_b \theta - y)$$

## 使用向量化后的梯度下降

```python
import numpy as np
from sklearn import datasets
```

```python
boston = datasets.load_boston()
```

```python
X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]
```

```python
%run ../util/model_selection.py
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
```

```python
%run ../LinearRegression/LinearRegression.py
```

```python
lr1 = LinearRegression()
%time lr1.fit_normal(X_train, y_train)
lr1.score(X_test, y_test)
```

Wall time: 14 ms

0.8129802602658359

```python
lr2 = LinearRegression()
%time lr2.fit_gd(X_train, y_train, eta=0.000001, n_iters=1e6)
lr2.score(X_test, y_test)
```

Wall time: 45.6 s

0.7541852353980764

## 梯度下降和数据归一化

在使用梯度下降法之前，做好进行数据归一化,归一化后学习率不需要设置太小，增加训练时间

```python
from sklearn.preprocessing import StandardScaler
```

```python
stdslr = StandardScaler()
stdslr.fit(X_train)
```

StandardScaler(copy=True, with_mean=True, with_std=True)

```python
X_train_std = stdslr.transform(X_train)
X_test_std = stdslr.transform(X_test)
```

```python
lr3 = LinearRegression()
%time lr3.fit_gd(X_train_std, y_train)
lr3.score(X_test_std, y_test)
```

Wall time: 190 ms

0.8129880620122235

## 梯度下降的优势

特征数多时，梯度下降法训练耗时比正规方程解要少

```python
m = 1000
n = 8000
big_X = np.random.normal(size=(m, n))
true_theta = np.random.uniform(0.0, 100, size=n+1)
big_y = big_X.dot(true_theta[1:]) + true_theta[0] + np.random.normal(0.0, 10.0, size=m)
```

```python
lr4 = LinearRegression()
%time lr4.fit_normal(big_X, big_y)
```

Wall time: 14.8 s

LinearRegression()

```python
lr5 = LinearRegression()
%time lr5.fit_gd(big_X, big_y)
```

Wall time: 4.61 s

LinearRegression()
