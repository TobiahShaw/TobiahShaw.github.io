
# Precision-Recall 曲线

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()

y[digits.target == 9] = 1
y[digits.target != 9] = 0

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

decision_scores = log_reg.decision_function(X_test)
```

```python
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precisions = []
recalls = []

thresholds = np.arange(np.min(decision_scores), np.max(decision_scores), 0.1)

for threshold in thresholds:
    y_predict = np.array(decision_scores >= threshold, dtype=int)
    precisions.append(precision_score(y_test, y_predict))
    recalls.append(recall_score(y_test, y_predict))
```

```python
plt.plot(thresholds, precisions)
plt.plot(thresholds, recalls)
plt.show()
```

![png](..\assets\img\ClassificationPerformanceMeasures\6_output_3_0.png)

## p-r 曲线

```python
plt.plot(precisions, recalls)
plt.show()
```

![png](..\assets\img\ClassificationPerformanceMeasures\6_output_5_0.png)

## scikit-learn 中的 Precision-Recall 曲线

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, decision_scores)
```

```python
precisions.shape
```

(145,)

```python
recalls.shape
```

(145,)

```python
thresholds.shape
```

(144,)

```python
plt.plot(thresholds, precisions[:-1])
plt.plot(thresholds, recalls[:-1])
plt.show()
```

![png](..\assets\img\ClassificationPerformanceMeasures\6_output_11_0.png)

```python
plt.plot(precisions, recalls)
plt.show()
```

![png](..\assets\img\ClassificationPerformanceMeasures\6_output_12_0.png)
