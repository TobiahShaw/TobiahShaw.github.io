
# 逻辑回归使用多项式特征

```python
import numpy as np
import matplotlib.pyplot as plt
```

```python
np.random.seed(666)
```

```python
X = np.random.normal(0, 1, size=(200,2))
y = np.array(X[:,0] ** 2 + X[:,1]**2 < 1.5, dtype=int)
```

```python
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```

![png](..\assets\img\LogisticRegression\4_output_4_0.png)

## 使用逻辑回归

```python
%run LogisticRegression.py
```

```python
log_reg = LogisticRegression()
log_reg.fit(X, y)
log_reg.score(X, y)
```

0.605

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
plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```

![png](..\assets\img\LogisticRegression\4_output_9_0.png)

## 添加多项式项

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

def PolynomialLogisticRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("log_reg", LogisticRegression())
    ])
```

```python
poly2_log_reg = PolynomialLogisticRegression(2)
poly2_log_reg.fit(X, y)
```

Pipeline(memory=None,
         steps=[('poly', PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)), ('std_scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('log_reg', LogisticRegression())])

```python
poly2_log_reg.score(X, y)
```

0.95

```python
plot_decision_boundary(poly2_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```

![png](..\assets\img\LogisticRegression\4_output_14_0.png)


```python
# overfitting
poly20_log_reg = PolynomialLogisticRegression(20)
poly20_log_reg.fit(X, y)
```

Pipeline(memory=None,
         steps=[('poly', PolynomialFeatures(degree=20, include_bias=True, interaction_only=False)), ('std_scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('log_reg', LogisticRegression())])

```python
poly20_log_reg.score(X, y)
```

0.955

```python
plot_decision_boundary(poly20_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```

![png](..\assets\img\LogisticRegression\4_output_17_0.png)