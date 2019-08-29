# 向量化运算

$$a = \frac{\sum_{i=1}^m(x^{(i)} - \bar{x})(y^{(i)} - \bar{y})}{\sum_{i=1}^m(x^{(i)} - \bar{x})^2}$$
符合形式：
$$\sum_{i=1}^m w^{(i)} · v^{(i)}$$
可以使使用向量点乘来提高运算效率： $$w·v$$

```python
import numpy as np
import matplotlib.pyplot as plt

m = 1000000
x = np.random.random(size=m)
y = x * 2.0 + 3.0 + np.random.normal()
```

```python
import numpy as np

class SimpleLinearRegressionV1:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1 and y_train.ndim == 1,\
            "needs single feature train data"
        assert len(x_train) == len(y_train),\
            "the size of x and y must be equals"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        self.a_ = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean) ** 2)
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        assert isinstance(x_predict, (int, float)) or x_predict.ndim == 1,\
            "needs single feature data to predict"
        assert self.a_ is not None and self.b_ is not None,\
            "must be fit before oredict"
        return self.a_ * x_predict + self.b_

    def __repr__(self):
        return "SimpleLinearRegressionV1()"

class SimpleLinearRegressionV2:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1 and y_train.ndim == 1,\
            "needs single feature train data"
        assert len(x_train) == len(y_train),\
            "the size of x and y must be equals"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot((x_train - x_mean))
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        assert isinstance(x_predict, (int, float)) or x_predict.ndim == 1,\
            "needs single feature data to predict"
        assert self.a_ is not None and self.b_ is not None,\
            "must be fit before oredict"
        return self.a_ * x_predict + self.b_

    def score(self, x_test, y_test):
        assert len(x_test) == len(y_test),\
        "the size of y_true must be equal to the size of y_predict"
        return 1 - np.sum((y_test - self.predict(x_test)) ** 2) / len(y_test) / np.var(y_test)

    def __repr__(self):
        return "SimpleLinearRegressionV2()"
```

```python
%%time
slr1 = SimpleLinearRegressionV1()
slr1.fit(x, y)
print(slr1.predict(6))
```

14.47465267558519
Wall time: 269 ms

```python
%%time
slr2 = SimpleLinearRegressionV2()
slr2.fit(x, y)
print(slr2.predict(6))
```

14.474652675585194
Wall time: 43 ms

其中v2是向量化之后的，v1和上一小节一样，没变，可以看到效率得到了不错的提升。