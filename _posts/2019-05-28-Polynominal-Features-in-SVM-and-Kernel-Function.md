
# SVM 使用多项式特性

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
```

```python
X, y = datasets.make_moons()
```

```python
X.shape
```

(100, 2)

```python
y.shape
```

(100,)

```python
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```

![png](..\assets\img\SVM\4_output_5_0.png)

```python
X, y = datasets.make_moons(noise=0.15, random_state=666)
```

```python
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```

![png](..\assets\img\SVM\4_output_7_0.png)

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def PolynomialSVC(degree, C=1.0):
    return Pipeline([
        ("poly", PolynomialFeatures(degree)),
        ("std_scaler", StandardScaler()),
        ("svc", LinearSVC(C=C))
    ])
```

```python
polySVC = PolynomialSVC(3)
polySVC.fit(X, y)
```

Pipeline(memory=None,
         steps=[('poly', PolynomialFeatures(degree=3, include_bias=True, interaction_only=False)), ('std_scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svc', LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
         verbose=0))])

```python
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100 )).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100 )).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    custom_camp = ListedColormap(['#EF9A9A', '#FFF59F', '#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_camp)
```

```python
plot_decision_boundary(polySVC, [-1.5,2.5,-1,1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```

![png](..\assets\img\SVM\4_output_11_0.png)

## 使用多项式核函数的SVM

```python
from sklearn.svm import SVC

def PolynomialKernelSVC(degree, C=1.0):
    return Pipeline([
        ("std_scaler",StandardScaler()),
        ("svc", SVC(kernel="poly", degree=degree, C=C))
    ])
```

```python
poly_kernel_svc = PolynomialKernelSVC(degree=3)
poly_kernel_svc.fit(X, y)
```

Pipeline(memory=None,
         steps=[('std_scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svc', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='poly', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False))])

```python
plot_decision_boundary(poly_kernel_svc, [-1.5,2.5,-1,1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```

![png](..\assets\img\SVM\4_output_15_0.png)
