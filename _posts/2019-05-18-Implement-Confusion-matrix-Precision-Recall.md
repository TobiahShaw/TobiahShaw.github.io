
# 实现混淆矩阵，精准率和召回率

```python
import numpy as np
from sklearn import datasets
```

```python
digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target==9] = 1
y[digits.target!=9] = 0
```

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
```

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)
```

0.9755555555555555

```python
y_log_predict = log_reg.predict(X_test)
```

```python
def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))
def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))
def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))
def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))
```

```python
TN(y_test, y_log_predict)
```

403

```python
FP(y_test, y_log_predict)
```

2

```python
FN(y_test, y_log_predict)
```

9

```python
TP(y_test, y_log_predict)
```

36

```python
def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_test, y_predict), FP(y_test, y_predict)],
        [FN(y_test, y_predict), TP(y_test, y_predict)]
    ])

confusion_matrix(y_test, y_log_predict)
```

array([[403,   2],
           [  9,  36]])

```python
def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0
```

```python
precision_score(y_test, y_log_predict)
```

0.9473684210526315

```python
def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0
```

```python
recall_score(y_test, y_log_predict)
```

0.8

## scikit-learn 中的实现

```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

confusion_matrix(y_test, y_log_predict)
```

array([[403,   2],
           [  9,  36]], dtype=int64)

```python
precision_score(y_test, y_log_predict)
```

0.9473684210526315

```python
recall_score(y_test, y_log_predict)
```

0.8
