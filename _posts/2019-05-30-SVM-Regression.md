
# SVM思路结局回归问题

- 在margin两条线内尽可能包含最多的点，所决定的中间的那个直线，就是拟合的模型
- 思想和分类问题相反，在分类问题中margin要求点尽可能少，甚至Hard Margin SVM里，margin之间没有点

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
```

```python
boston = datasets.load_boston()
X = boston.data
y = boston.target
```

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
```

```python
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def StandardLinearSVR(epsilon=0.1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("linearSVR", LinearSVR(epsilon=epsilon))
    ])
```

```python
svr = StandardLinearSVR()
svr.fit(X_train, y_train)
svr.score(X_test, y_test)
```

0.6356660559722602

```python
def RBFKernelSVR(epsilon=0.1, gamma=1.0):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("SVR", SVR(epsilon=epsilon, kernel='rbf', gamma=gamma))
    ])
```

```python
rbf_svr = RBFKernelSVR(0.1, 0.045)
rbf_svr.fit(X_train, y_train)
rbf_svr.score(X_test, y_test)
```

0.6835003387550918

```python
from sklearn.preprocessing import PolynomialFeatures

def PolySVR(degree=3, epsilon=0.1):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("linearSVR", LinearSVR(epsilon=epsilon))
    ])
```

```python
poly_svr = PolySVR()
poly_svr.fit(X_train, y_train)
poly_svr.score(X_test, y_test)
```

0.7590544115659419