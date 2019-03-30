# scikit-learn 中的线性回归

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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
```

## 使用scikit-learn中的LinearRegression

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
```

```python
lin_reg.fit(X_train, y_train)
```

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)

```python
lin_reg.coef_
```

array([-1.15625837e-01,  3.13179564e-02, -4.35662825e-02, -9.73281610e-02,
           -1.09500653e+01,  3.49898935e+00, -1.41780625e-02, -1.06249020e+00,
            2.46031503e-01, -1.23291876e-02, -8.79440522e-01,  8.31653623e-03,
           -3.98593455e-01])

```python
lin_reg.intercept_
```

32.5975615887

```python
lin_reg.score(X_test, y_test)
```

0.8009390227581032

## kNN Regressor

```python
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor()
```

```python
knn_reg.fit(X_train, y_train)
```

KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=None, n_neighbors=5, p=2,
              weights='uniform')

```python
knn_reg.score(X_test, y_test)
```

0.602674505080953

```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        "weights" : ["uniform"],
        "n_neighbors": [i for i in range(1, 11)]
    },
    {
        "weights" : ["distance"],
        "n_neighbors": [i for i in range(1, 11)],
        "p": [i for i in range(1, 6)]
    }
]

knn_reg_se = KNeighborsRegressor()
grid_search = GridSearchCV(knn_reg_se, param_grid, verbose=1)
grid_search.fit(X_train, y_train)
```

    D:\ProgramData\Anaconda3\envs\tensorflow\lib\site-packages\sklearn\model_selection\_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
      warnings.warn(CV_WARNING, FutureWarning)
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    Fitting 3 folds for each of 60 candidates, totalling 180 fits


    [Parallel(n_jobs=1)]: Done 180 out of 180 | elapsed:    1.5s finished
    D:\ProgramData\Anaconda3\envs\tensorflow\lib\site-packages\sklearn\model_selection\_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)





    GridSearchCV(cv='warn', error_score='raise-deprecating',
           estimator=KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=None, n_neighbors=5, p=2,
              weights='uniform'),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid=[{'weights': ['uniform'], 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, {'weights': ['distance'], 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'p': [1, 2, 3, 4, 5]}],
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=1)

```python
grid_search.best_params_
```

{'n_neighbors': 6, 'p': 1, 'weights': 'distance'}

```python
# cv score 交叉验证
grid_search.best_score_
```

0.6060528490355778


```python
knn_reg_se = grid_search.best_estimator_
knn_reg_se.score(X_test, y_test)
```

0.7353138117643773