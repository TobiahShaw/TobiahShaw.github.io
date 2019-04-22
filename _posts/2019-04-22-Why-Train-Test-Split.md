
# 为什么要训练数据集和测试数据集

## 模型的泛化能力

- 泛化能力即由此及彼的能力
- 模型在面对新的数据样本时的预测能力
- 过拟合、欠拟合泛化能力差

## 如何提高模型的泛化能力

- 训练数据集、测试数据集分离
- 如果使用训练数据集得到的模型对测试数据集预测能力较好的话，我们就说模型泛化能力不错
- 如果模型泛化能力差，多半可能遭遇了过拟合

```python
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(666)
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, size=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
```

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def PolynomialRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])
```

### 线性回归

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_predict = lin_reg.predict(X_test)
lin_reg.score(X_test, y_test)
```

0.5437074558433677

```python
mean_squared_error(y_test, y_predict)
```

2.2199965269396573

### 多项式回归

#### degree=2

```python
poly2_reg = PolynomialRegression(2)
poly2_reg.fit(X_train, y_train)
y_predict_poly2 = poly2_reg.predict(X_test)
poly2_reg.score(X_test, y_test)
```

0.8348374397431063

```python
mean_squared_error(y_test, y_predict_poly2)
```

0.8035641056297901

#### degree=10

```python
poly10_reg = PolynomialRegression(10)
poly10_reg.fit(X_train, y_train)
y_predict_poly10 = poly10_reg.predict(X_test)
poly10_reg.score(X_test, y_test)
```

0.8106397218492938

```python
mean_squared_error(y_test, y_predict_poly10)
```

0.9212930722150721

#### degree=100

```python
poly100_reg = PolynomialRegression(100)
poly100_reg.fit(X_train, y_train)
y_predict_poly100 = poly100_reg.predict(X_test)
poly100_reg.score(X_test, y_test)
```

-2893100737.14305

```python
mean_squared_error(y_test, y_predict_poly100)
```

14075780270.824253
