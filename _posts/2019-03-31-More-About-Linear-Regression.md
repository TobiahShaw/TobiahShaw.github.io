
# 更多关于线性回归模型的讨论


```python
import numpy as np
from sklearn import datasets

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]
```


```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
# 不关系数据准确的，所以不对数据进行train test split
lin_reg.fit(X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)




```python
# 数值代表特征对结果的相关程度，正负代表是正相关还是负相关
lin_reg.coef_
```




    array([-1.06715912e-01,  3.53133180e-02, -4.38830943e-02,  4.52209315e-01,
           -1.23981083e+01,  3.75945346e+00, -2.36790549e-02, -1.21096549e+00,
            2.51301879e-01, -1.37774382e-02, -8.38180086e-01,  7.85316354e-03,
           -3.50107918e-01])




```python
# 从负相关到正相关特征排序
np.argsort(lin_reg.coef_)
```




    array([ 4,  7, 10, 12,  0,  2,  6,  9, 11,  1,  8,  3,  5], dtype=int64)




```python
boston.feature_names[np.argsort(lin_reg.coef_)]
```




    array(['NOX', 'DIS', 'PTRATIO', 'LSTAT', 'CRIM', 'INDUS', 'AGE', 'TAX',
           'B', 'ZN', 'RAD', 'CHAS', 'RM'], dtype='<U7')




```python
print(boston.DESCR)
```

    .. _boston_dataset:
    
    Boston house prices dataset
    ---------------------------
    
    **Data Set Characteristics:**  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    .. topic:: References
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
    
    

### 可解释性

由上述排序可以看到，房间数和是否临河，对房价正相关影响最大，房间数越多，临河房价越高；负相关中NOX（房子周边一氧化氮浓度）越小，DIS（距离五个劳务雇佣中心加权平均距离）越近，房价越高。


我们可以根据可解释性提高我们的预测准确率

### 八股

1. 训练测试集分离
2. 训练模型
3. 评价性能 R Squared

### 特点

- 典型的参数学习
    （对比kNN：非参数学习）
- 只能解决回归问题
- 对数据又假设：线性

### 优点
- 对数据又可解释性

### 缺点
- 使用正规方程解，时间复杂度高
