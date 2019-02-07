
# 测试算法效率


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
```


```python
iris = datasets.load_iris()
```


```python
X = iris.data
y = iris.target
```


```python
X.shape
```




    (150, 4)




```python
y.shape
```




    (150,)



### train test split


```python
shuffle_indexes = np.random.permutation(len(X))
```


```python
test_ratio = 0.2
test_size = int(len(X) * test_ratio)
```


```python
test_size
```




    30




```python
test_indexes = shuffle_indexes[:test_size]
train_indexes = shuffle_indexes[test_size:]
```


```python
X_train = X[train_indexes]
y_train = y[train_indexes]
```


```python
X_test = X[test_indexes]
y_test = y[test_indexes]
```


```python
print(X_train.shape)
print(y_train.shape)
```

    (120, 4)
    (120,)
    


```python
print(X_test.shape)
print(y_test.shape)
```

    (30, 4)
    (30,)
    

### 使用封装好的算法


```python
%run ../util/model_selection.py

def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0],\
        "the size of X must equal to the size of x"
    assert 0.0 <= test_ratio <= 1.0,\
        "test_ratio must valid"
    if seed:
        np.random.seed(seed)
    shuffle_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test ,y_train, y_test
```


```python
X_train, X_test ,y_train, y_test = train_test_split(X,y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

    (120, 4)
    (120,)
    (30, 4)
    (30,)
    


```python
%run kNN.py
```


```python
my_kNN_classifier = kNNClassifier(3)
```


```python
my_kNN_classifier.fit(X_train=X_train, y_train=y_train)
```




    kNN(k=3)




```python
y_predict = my_kNN_classifier.predict(X_predict=X_test)
```


```python
y_predict
```




    array([2, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 0, 0, 2, 1, 0, 0, 2, 0, 1, 2, 1,
           0, 1, 2, 0, 0, 2, 1, 2])




```python
y_test
```




    array([2, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 0, 0, 2, 1, 0, 0, 2, 0, 1, 2, 2,
           0, 1, 2, 0, 0, 2, 1, 2])




```python
accuracy = sum(y_predict == y_test) / len(y_test)
```


```python
accuracy
```




    0.9666666666666667



### sklearn 中的 train_test_split


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

    (120, 4)
    (120,)
    (30, 4)
    (30,)
    
