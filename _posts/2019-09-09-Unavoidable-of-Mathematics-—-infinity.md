# 必知必会的数学之无穷大与无穷小

## 无穷小

### 定义一

如果函数 $ f(x) $ 当 $ x → x_0 或 （x → \infty）$ 时的极限为 0，那么函数 $ f(x) $ 为当 $ x → x_0 或 （x → \infty）$ 时的无穷小。

注意：无穷小不是一个很小的数，0是无穷小。

### 定理一（无穷小和极限值的关系）

$$ \lim_{x → x_0 (x → \infty)} f(x) = A \Longleftrightarrow f(x) = A + \alpha (无穷小) $$

注意：

（1）有限个无穷小的和仍是无穷小；

（2）有界函数和无穷小的乘积仍是无穷小；

（3）有限个无穷小的积认识无穷小。

### 无穷小的比较

#### 高阶、低阶、同阶及等价的概念

设 $ \lim \alpha = 0, \lim \beta = 0, \alpha \neq 0 $

（1）若 $ \lim \frac{\beta}{\alpha} = 0 $ ，则称 $ \beta $ 是比 $ \alpha $ 的高阶无穷小，记作 $ \beta = o(\alpha) $；

（2）若 $ \lim \frac{\beta}{\alpha} = \infty $ ，则称 $ \beta $ 是比 $ \alpha $ 的低阶无穷小；

（3）若 $ \lim \frac{\beta}{\alpha} = c \neq 0 $ ，则称 $ \beta $ 是 $ \alpha $ 的同阶无穷小；

（4）若 $ \lim \frac{\beta}{\alpha^k} = c \neq 0 $ ，则称 $ \beta $ 是 $ \alpha $ 的k阶无穷小；

（5）若 $ \lim \frac{\beta}{\alpha} = 1 $ ，则称 $ \beta $ 是 $ \alpha $ 的等价无穷小；

#### 等价无穷小的性质

（1）自反性 a ~ a；
（2）对称性 若 a ~ b，则 b ~ a；
（3）传递性 若 a ~ b，b ~ c，则 a ~ c；

#### 无穷小有关的基本定理

定理一：

$$ \beta \sim \alpha \Leftrightarrow \alpha + o(\alpha) $$

定理二：

$$ 若 \alpha \sim \alpha_1, \beta \sim \beta_1, 则 \lim \frac{\beta}{\alpha} = \lim \frac{\beta_1}{\alpha_1} $$

**注：** 仅可在乘除法中使用，常用等价无穷小：

$$ x → 0时,sin x \sim arcsin x \sim tan x \sim arctan x \sim ln(1+x) \sim e^x -1 \sim x;\\ 1 - cos x \sim \frac{1}{2}x^2;\\ (1 + \alpha x)^\beta - 1 \sim \alpha \beta x;\\ \alpha^x - 1 \sim x ln \alpha $$

**注2：** 等价无穷小替换在求极限中常用，仅限乘除法。在满足整体乘除关系时可先算出其中一部分的极限再进行乘除。

## 无穷大

### 定义二

设函数 $ f(x) $ 在 $ x_0 $ 某一去心邻域内有定义（或 |x| 大于某一正数时有定义），如果对于任意给定的正数 M （无论它多么大），总存在 $ \delta > 0 $ （或 X > 0），对适合不等式 $ 0 < | x - x_0 | < \delta $ （或 |x| > X）的**一切** x，对用的函数值总满足不等式 $ |f(x)| > M $，那么称 $ f(x) $ 是当$ x → x_0 或 （x → \infty）$ 时的无穷大。

注意：无穷大和无界的区别在于，无穷大强调任意性，无界强调存在性。

### 定理二（无穷小和无穷大的关系）

在自变量的统一变化过程中，如果 $ f(x) $ 是无穷大，那么  $ \frac{1}{f(x)} $ 为无穷小；反之，如果 $ f(x) $ 是无穷小，且 $ f(x) \neq 0 $，那么  $ \frac{1}{f(x)} $ 为无穷大。
