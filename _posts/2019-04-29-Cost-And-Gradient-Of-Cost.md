# 逻辑回归的损失函数

逻辑回归：

$$\hat{p} = \sigma(\theta^T \cdot x_b) = \frac{1}{1 + e^{-\theta^T \cdot x_b}}$$

$$\hat{y} = \begin{cases}
1, \quad \hat{p} \geq 0.5 \\\\
0, \quad \hat{p} < 0.5
\end{cases}
$$

则损失函数为：

$$cost = \begin{cases}
如果y=1,p越小,cost越大\\\\
如果y=0,p越大,cost越大
\end{cases}$$

我们可以用使用这种函数作为损失函数

$$cost=\begin{cases}
-log(\hat{p}), \quad if \quad y=1\\\\
-log(1-\hat{p}), \quad if \quad y=0
\end{cases}$$

可以推导为：

$$cost=-ylog(\hat{p})-(1-y)log(1-\hat{p}) \qquad其中 (y \in \{0,1\})$$

则,对应m个样本

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^my^{(i)}log(\hat{p}^{(i)}) + (1-y^{(i)})log(1-\hat{p}^{(i)})$$

$$\hat{p}^{(i)} = \sigma(x_b^{(i)} \theta) = \frac{1}{1 + e^{-x_b^{(i)} \theta}}$$

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^my^{(i)}log(\sigma(x_b^{(i)} \theta)) + (1-y^{(i)})log(1-\sigma(x_b^{(i)} \theta))$$

**对于这个算式，没有公式解，只能使用梯度下降法求解**

# 逻辑回归损失函数的梯度

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^my^{(i)}log(\sigma(x_b^{(i)} \theta)) + (1-y^{(i)})log(1-\sigma(x_b^{(i)} \theta))$$

首先对sigmoid函数求导

$$\sigma(t) = \frac{1}{1+e^{-t}} = (1+e^{-t})^{-1} \qquad \sigma(t)^{'}=(1+e^{-t})^{-2} \cdot e^{-t}$$

然后对$log\sigma(t)$求导

$$(log\sigma(t))^{'}=\frac{1}{\sigma(t)} \cdot \sigma(t)^{'}=(1+e^{-t})^{-1} \cdot e^{-t} = \frac{e^{-t}}{1+e^{-t}} = \frac{1+e^{-t}-1}{1+e^{-t}} = 1-\frac{1}{1+e^{-t}}$$

$$(log\sigma(t))^{'} = 1-\sigma(t)$$

**所以**

$$\frac{d(y^{(i)}log\sigma(x_b^{(i)} \theta))}{d\theta_j} = y^{(i)}(1-\sigma((x_b^{(i)} \theta)) \cdot X_j^{(i)} = y^{(i)}X_j^{(i)}-y^{(i)}\sigma(X_b^{(i)}\theta) \cdot X_j^{(i)}$$

然后对$log(1-\sigma(t))$求导

$$(log(1-\sigma(t)))^{'}=\frac{1}{1-\sigma(t)} \cdot (-1) \cdot \sigma(t)^{'} = -\frac{1}{1-\sigma(t)} \cdot (1+e^{-t})^{-2} \cdot e^{-t}$$

其中

$$-\frac{1}{1-\sigma(t)} = -\frac{1}{\frac{1+e^{-t}}{1+e^{-t}} - \frac{1}{1+e^{-t}}} = -\frac{1+e^{-t}}{e^{-t}}$$

则

$$(log(1-\sigma(t)))^{'}= -(1+e^{-t})^{-1} = -\sigma(t)$$

**所以**

$$\frac{d((1-y^{(i)})log(1-\sigma(x_b^{(i)} \theta)))}{d(\theta_j)} = (1-y^{(i)}) \cdot (-\sigma(x_b^{(i)} \theta)) \cdot X_j^{(i)} = -\sigma(X_b^{(i)}\theta) \cdot X_j^{(i)} + y^{(i)}\sigma(X_b^{(i)}\theta) \cdot X_j^{(i)}$$

**所以**

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^m(\sigma(X_b^{(i)}\theta)-y^{(i)}) X_j^{(i)} = \frac{1}{m}\sum_{i=1}^m(\hat{p}^{(i)}-y^{(i)}) X_j^{(i)}$$

**所以，梯度：**

$$\nabla J(\theta) = \begin{pmatrix}
\frac{\partial J(\theta)}{\partial \theta_0} \\\\
\frac{\partial J(\theta)}{\partial \theta_1} \\\\
\frac{\partial J(\theta)}{\partial \theta_2} \\\\
\ldots \\\\
\frac{\partial J(\theta)}{\partial \theta_n} \\\\
\end{pmatrix} = \frac{1}{m}\begin{pmatrix}
\sum_{i=1}^m(\sigma(X_b^{(i)}\theta)-y^{(i)}) \\\\
\sum_{i=1}^m(\sigma(X_b^{(i)}\theta)-y^{(i)}) \cdot X_1^{(i)}\\\\
\sum_{i=1}^m(\sigma(X_b^{(i)}\theta)-y^{(i)}) \cdot X_2^{(i)}\\\\
\ldots \\\\
\sum_{i=1}^m(\sigma(X_b^{(i)}\theta)-y^{(i)}) \cdot X_n^{(i)}\\\\
\end{pmatrix} = \frac{1}{m}\begin{pmatrix}
\sum_{i=1}^m(\hat{p}^{(i)}-y^{(i)}) \\\\
\sum_{i=1}^m(\hat{p}^{(i)}-y^{(i)}) \cdot X_1^{(i)}\\\\
\sum_{i=1}^m(\hat{p}^{(i)}-y^{(i)}) \cdot X_2^{(i)}\\\\
\ldots \\\\
\sum_{i=1}^m(\hat{p}^{(i)}-y^{(i)}) \cdot X_n^{(i)}\\\\
\end{pmatrix}$$

**向量化**

$$\nabla J(\theta) = \frac{1}{m} \cdot X_b^T \cdot(\sigma(X_b\theta)-y)$$
