<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# 多元线性回归

$$y = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n$$

$$\hat{y} = \theta_0 + \theta_1X_1^{(i)} + \theta_2X_2^{(i)} + ... + \theta_nX_n^{(i)}$$

**目标：** 找到$$\theta_0,\theta_1,\theta_2,...,\theta_n$$使

$$\sum_{i=1}^m(y^{(i)} - \hat{y}^{(i)})^2$$

尽可能小

## 处理式子

$$\hat{y}^{(i)} = \theta_0 + \theta_1X_1^{(i)} + \theta_2X_2^{(i)} + ... + \theta_nX_n^{(i)}$$

$$\theta = (\theta_0, \theta_1, \theta_2,...,\theta_n)^T$$

$$\hat{y}^{(i)} = \theta_0X_0^{(i)} + \theta_1X_1^{(i)} + \theta_2X_2^{(i)} + ... + \theta_nX_n^{(i)}, X_0^{(i)}\equiv1$$

$$X^{(i)} = (X_0^{(i)},X_1^{(i)},X_2^{(i)}...,X_n^{(i)})$$

$$\hat{y}^{(i)} = X^{(i)}·\theta$$

$$\mathbf{X}_b =
\left( \begin{array}{ccc}
1 & X_1^{(1)} & X_2^{(1)} & \ldots & X_n^{(1)}\\\\
1 & X_1^{(2)} & X_2^{(2)} & \ldots & X_n^{(2)}\\\\
\ldots &  &  & & \ldots \\\\
1 & X_1^{(m)} & X_2^{(m)} & \ldots & X_n^{(m)}
\end{array} \right)$$

$$\mathbf{\theta} =
\left( \begin{array} {ccc}
\theta_0\\\\
\theta_1\\\\
\theta_2\\\\
\ldots \\\\
\theta_n
\end{array}\right)$$

**预测化简为**

$$\hat{y} = X_b · \theta$$

<br/>

则目标可化简为：使
$$(y - X_b·\theta)^T(y - X_b·\theta)$$
尽可能小

可推导出
$$\theta = (X_b^TX_b)^{-1}X_b^Ty$$
即，多元线性回归的正规方程解（Normal Equation）

<br/>

问题：时间复杂度高：O(n^3)（优化O(n^2.4)）

优点： 不需要对数据做归一化处理

## 实现多元线性回归模型

```python
import numpy as np
from sklearn import datasets
```

```python
boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50]
y = y[y < 50]
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
```

```python
import numpy as np

class LinearRegression():
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0],\
            "the size of X_train must be equal to y_train"
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0],\
            "the size of X_train must be equal to y_train"

        def J(theta, x_b, y):
            try:
                return np.sum((y - x_b.dot(theta)) ** 2) / len(x_b)
            except:
                return float('inf')

        def dJ(theta, x_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(x_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = np.sum((x_b.dot(theta) - y).dot(x_b[:,i]))
            # return res * 2 / len(x_b)
            return x_b.T.dot(x_b.dot(theta) - y) * 2. / len(y)

        def gradient_descent(x_b, y, initial_theta, eta, n_iters=10000, epsilon=1e-8):
            theta = initial_theta
            i_iter = 0
            while i_iter < n_iters:
                gradient = dJ(theta, x_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                i_iter = i_iter + 1
                if(abs(J(theta, x_b, y) - J(last_theta, x_b, y)) < epsilon):
                    break
            return theta
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]

        return self

    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        assert X_train.shape[0] == y_train.shape[0],\
            "the size of X_train must be equal to y_train"
        assert n_iters >= 1,\
            "n_iters must >= 1"
        def dJ_sgd(theta, x_b_i, y_i):
            return x_b_i.T.dot(x_b_i.dot(theta) - y_i) * 2.
        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):
            def learn_rate(t):
                return t0 / (t + t1)
            theta = initial_theta
            m = len(X_b)
            for cur_iters in range(n_iters):
                indexs = np.random.permutation(m)
                X_b_new = X_b[indexs]
                y_new = y[indexs]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learn_rate(cur_iters * m + i) * gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train),1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]

        return self

    def predict(self, X_predict):
        assert self.coef_ is not None and self.interception_ is not None,\
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_),\
            "the feature of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return self._r2_score(y_test, y_predict)

    def _r2_score(self, y_true, y_predict):
        assert len(y_true) == len(y_predict),\
            "the size of y_true must be equal to the size of y_predict"
        mean_squared_error = np.sum((y_true - y_predict) ** 2) / len(y_true)
        return 1 - mean_squared_error / np.var(y_true)

    def __repr__(self):
        return "LinearRegression()"
```

```python
reg = LinearRegression()
reg.fit_normal(X_train, y_train)
```

LinearRegression()

```python
reg.coef_
```

array([-1.20354261e-01,  3.64423279e-02, -3.61493155e-02,  5.12978140e-02,
           -1.15775825e+01,  3.42740062e+00, -2.32311760e-02, -1.19487594e+00,
            2.60101728e-01, -1.40219119e-02, -8.35430488e-01,  7.80472852e-03,
           -3.80923751e-01])

```python
reg.interception_
```

34.11739972322438

```python
reg.score(X_test, y_test)
```

0.8129794056212711