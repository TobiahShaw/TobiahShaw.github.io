
# 使用 PCA 降噪

## 手写识别例子

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
```

```python
digits = datasets.load_digits()
X = digits.data
y = digits.target
```

### 加入噪声

```python
noisy_digits = X + np.random.normal(0, 4, size=X.shape)
```

```python
example_digits = noisy_digits[y==0,:][:10]
for num in range(1, 10):
    X_num = noisy_digits[y==num,:][:10]
    example_digits = np.vstack([example_digits, X_num])
```

```python
example_digits.shape
```

(100, 64)

```python
def plot_digits(data):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10), 
                            subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8), cmap='binary', interpolation='nearest',
                 clim=(0, 16))
    plt.show()
```

### 绘制加入噪声后的图像

```python
plot_digits(example_digits)
```

![png](..\assets\img\PCA\output_9_0_0.png)

### 使用PCA降噪

```python
from sklearn.decomposition import PCA
```

```python
pca = PCA(0.5)
pca.fit(noisy_digits)
noisy_digits_reduction = pca.transform(noisy_digits)
noisy_digits_reduction = pca.inverse_transform(noisy_digits_reduction)
```

```python
example_digits_reduction = noisy_digits_reduction[y==0,:][:10]
for num in range(1, 10):
    X_num = noisy_digits_reduction[y==num,:][:10]
    example_digits_reduction = np.vstack([example_digits_reduction, X_num])
```

```python
plot_digits(example_digits_reduction)
```

![png](..\assets\img\PCA\output_14_0.png)

### 原本信息

```python
example_digits_original = X[y==0,:][:10]
for num in range(1, 10):
    X_num = X[y==num,:][:10]
    example_digits_original = np.vstack([example_digits_original, X_num])
```

```python
plot_digits(example_digits_original)
```

![png](..\assets\img\PCA\output_17_0_0.png)