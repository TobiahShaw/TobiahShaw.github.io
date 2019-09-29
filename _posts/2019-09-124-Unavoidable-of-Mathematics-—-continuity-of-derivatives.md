# 必知必会的数学之导数

## 导数的定义

### 定义一

$$ f'(x_0) = \lim_{\Delta x → 0} \frac{\Delta y}{\Delta x} = \lim_{\Delta x → 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x} $$

当以上极限**存在** 时，这个极限称为函数 f(x) 在这个点的导数。

**注：** 也可写作$ f'(x_0) = \lim_{h → 0} \frac{f(x_0 + h) - f(x_0)}{\Delta x}, f'(x_0) = \lim_{x → x_0} \frac{f(x) - f(x_0)}{x - x_0} $， 且定义中 "-f(x0)" 重点关注，不可忽略，除非为 0 时，不写出来。

### 定义二

右导数：

$$ f_+'(x_0) = \lim_{\Delta x → 0^+} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x} $$

左导数：

$$ f_-'(x_0) = \lim_{\Delta x → 0^-} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x} $$

### 定理一

$$ f'(x_0) = A \Leftrightarrow f_+'(x_0) = f_-'(x_0) = A $$

### 定理二

若函数 $ y = f(x) $ 在 $ x = x_0 $ 处可导，则 $ y = f(x) $ 在 $ x = x_0 $ 处连续；**反之不对**。（可导必连续，连续不一定可导）

### 定义三

如果函数 f(x) 在区间 I 上每一点都可导（端点是指单侧可导），则称 f(x) 在区间 I 上可导，此时 f'(x) 依然是个函数，叫做原来函数 f(x) 的导函数。

## 求导法则

### 常数和基本初等函数的导数公式

$$ (C)' = 0 \\ (x^\mu)' = \mu x^{\mu - 1} \\ (sin x)' = cos x \\ (cos x)' = -sin x \\ (tan x)' = sec^2 x \\ (cot x)' = -csc^2 x \\ (sec x)' = sec \  xtgx \\ (csc x)' = -csc \ xctgx \\ (a^x)' = a^x ln a \\ (e^x)' = e^x \\ (log_a x)' = \frac{1}{x ln a} \\ (ln x)' = \frac{1}{x} \\ (arcsin \  x)' = \frac{1}{\sqrt{1 - x^2}} \\ (arccos \  x)' = -\frac{1}{\sqrt{1 - x^2}} \\ (arctan \  x)' = \frac{1}{1 - x^2} \\ (arccot \  x)' = -\frac{1}{1 - x^2} $$

### 四则运算求导法则

设 u、v 均可导，则：

$$ (u \pm v)' = u' \pm v' \\ (uv)' = u'v + uv' \\ (\frac{u}{v})' = \frac{u'v - uv'}{v^2} $$

### 反函数求导法则

设 $ y = f(x) $ 再某区间单调可导且 $ f'(x) \neq 0 $，则其反函数 $ x = \phi(y) $（**注意反函数的形式**）在对应区间也可导，且

$$ \frac{dx}{dy} = \frac{1}{\frac{dy}{dx}} 即 \phi'(y) = \frac{1}{f'(x)} $$

#### 推导

由于我们可以理解导数为切线斜率：

$$ y = f(x) → y_x' = tan \alpha \\ x = \phi(y) → x_y' = tan \beta $$

因为二者互为反函数，则：

$$ \alpha + \beta = \frac{\pi}{2} \\ 则：y_x' = \frac{1}{x_y'} \\ 即：x_y' = \frac{1}{y_x'} $$
