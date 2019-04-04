
# 多元线性回归中的梯度下降法

其中
$$-\eta\frac{dJ}{d\theta}$$
应变为
$$-\eta\nabla J$$
其中
$$\nabla J = (\frac{\partial J}{\partial \theta_0}, \frac{\partial J}{\partial \theta_1}, \ldots ,\frac{\partial J}{\partial \theta_n})$$
梯度代表方向，对应J增大最快的方向

### 目标化简

线性回归目标：使
$$\sum_{i=1}^m (y^{(i)} - \hat{y}^{(i)})^2$$
尽可能小

$$\hat{y}^{(i)} = \theta_0 + \theta_1 X_1^{(i)} + \theta_2 X_2^{(i)} + \ldots + \theta_n X_n^{(i)}$$

则目标可变为：使
$$\sum_{i=1}^m (y^{(i)} - \theta_0 - \theta_1 X_1^{(i)} - \theta_2 X_2^{(i)} - \ldots -  \theta_n X_n^{(i)})^2$$
尽可能小

即上述函数对应损失函数J

$$\nabla J(\theta) = \begin{pmatrix}
\frac{\partial J}{\partial \theta_0} \\\\
\frac{\partial J}{\partial \theta_1} \\\\
\frac{\partial J}{\partial \theta_2} \\\\
\ldots \\\\
\frac{\partial J}{\partial \theta_n}
\end{pmatrix} = 
\begin{pmatrix}
\sum_{i=1}^m 2(y^{(i)} - X_b^{(i)}\theta) \cdot (-1)\\\\
\sum_{i=1}^m 2(y^{(i)} - X_b^{(i)}\theta) \cdot (-X^{(i)}_1) \\\\
\sum_{i=1}^m 2(y^{(i)} - X_b^{(i)}\theta) \cdot (-X^{(i)}_2) \\\\
\ldots \\\\
\sum_{i=1}^m 2(y^{(i)} - X_b^{(i)}\theta) \cdot (-X^{(i)}_n)
\end{pmatrix}
= 2 \cdot \begin{pmatrix}
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \\\\
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_1 \\\\
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_2 \\\\
\ldots \\\\
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_n
\end{pmatrix}$$

但是此时梯度会和样本数量m相关，我们更改损失函数为下述方式(MSE)：
$$\frac{1}{m}\sum_{i=1}^m (y^{(i)} - \hat{y}^{(i)})^2$$

则：
$$\nabla J(\theta) = 
\frac{2}{m} \cdot \begin{pmatrix}
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \\\\
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_1 \\\\
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_2 \\\\
\ldots \\\\
\sum_{i=1}^m (X_b^{(i)}\theta - y^{(i)}) \cdot X^{(i)}_n
\end{pmatrix}$$
