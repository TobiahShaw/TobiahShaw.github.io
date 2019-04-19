
# 多项式回归

- 多项式回归是线性回归的一种
- 多项式回归添加了原来线性回归中自变量的高次项
- 多项式回归问题可以通过变量转换化为多元线性回归问题来解决

```python
import numpy as np
import matplotlib.pyplot as plt
```

```python
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
```

```python
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
```

```python
plt.scatter(x, y)
plt.show()
```

![png](..\assets\img\PolynomialRegression\output_4_0.png)

## 使用线性回归拟合

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
```

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

```python
y_predict = lin_reg.predict(X)
```

```python
plt.scatter(x, y)
plt.plot(x, y_predict, color='r')
plt.show()
```

![png](..\assets\img\PolynomialRegression\output_8_0.png)

## 解决方案，添加一个特征

```python
X2 = np.hstack([X, X**2])
X2.shape
```

(100, 2)

```python
lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)
```

```python
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.show()
```

![png](..\assets\img\PolynomialRegression\output_12_0.png)

```python
lin_reg2.coef_
```

array([1.02286659, 0.51340548])

```python
lin_reg2.intercept_
```

1.9901912014704577