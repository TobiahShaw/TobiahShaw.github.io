
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# 最小二乘法

线性回归过程主要解决的就是如何通过样本来获取最佳的拟合线。最常用的方法便是最小二乘法，它是一种数学优化技术，它通过最小化误差的平方和寻找数据的最佳函数匹配。

## 下面是最小二乘法的数学推导

目标：找到a和b，使得
$$\sum_{i=1}^{m}{(y^{(i)} - a{\hat{x}}^{(i)} - b)}^{2}$$
尽可能小

设
$$J(a, b) = \sum_{i=1}^{m}{(y^{(i)} - a{\hat{x}}^{(i)} - b)}^{2}$$
J(a, b)分别对a， b求偏导，即：
$$\frac{\partial{J(a, b)}}{\partial{a}} = 0$$
$$\frac{\partial{J(a, b)}}{\partial{b}} = 0$$

对b求导：
$$\frac{\partial{J(a, b)}}{\partial{b}} = \sum_{i=1}^{m} 2(y^{(i)} - ax^{(i)} - b)(-1) = 0$$

$$\sum_{i=1}^{m} (y^{(i)} - ax^{(i)} - b) = 0$$

$$\sum_{i=1}^{m}y^{(i)} - a\sum_{i=1}^{m}x^{(i)} - \sum_{i=1}^{m}b = 0$$

$$\sum_{i=1}^{m}y^{(i)} - a\sum_{i=1}^{m}x^{(i)} - mb = 0$$

$$mb = \sum_{i=1}^{m}y^{(i)} - a\sum_{i=1}^{m}x^{(i)}$$

$$b = \bar{y} - a\bar{x}$$

对a求导
$$\frac{\partial{J(a, b)}}{\partial{a}} = \sum_{i=1}^{m} 2(y^{(i)} - ax^{(i)} - b)(-x^{(i)}) = 0$$

$$\sum_{i=1}^{m} (y^{(i)} - ax^{(i)} - b) x^{(i)} = 0$$

将$$b = \bar{y} - a\bar{x}$$带入得到

$$\sum_{i=1}^{m} (y^{(i)} - ax^{(i)} - \bar{y} + a\bar{x}) x^{(i)} = 0$$

$$\sum_{i=1}^{m} (x^{(i)}y^{(i)} - a(x^{(i)})^2 - x^{(i)}\bar{y} + a\bar{x}x^{(i)}) = 0$$

$$\sum_{i=1}^{m} (x^{(i)}y^{(i)} - x^{(i)}\bar{y} - a(x^{(i)})^2  + a\bar{x}x^{(i)}) = 0$$

$$\sum_{i=1}^{m} (x^{(i)}y^{(i)} - x^{(i)}\bar{y}) - \sum_{i=1}^{m}(a(x^{(i)})^2 - a\bar{x}x^{(i)}) = 0$$

$$\sum_{i=1}^{m} (x^{(i)}y^{(i)} - x^{(i)}\bar{y}) - a\sum_{i=1}^{m}((x^{(i)})^2 - \bar{x}x^{(i)}) = 0$$

$$a\sum_{i=1}^{m}((x^{(i)})^2 - \bar{x}x^{(i)}) = \sum_{i=1}^{m} (x^{(i)}y^{(i)} - x^{(i)}\bar{y})$$

$$a = \frac{\sum_{i=1}^{m} (x^{(i)}y^{(i)} - x^{(i)}\bar{y})}{\sum_{i=1}^{m}((x^{(i)})^2 - \bar{x}x^{(i)})}$$

由于
$$\sum_{(i=1)}^m x^{(i)}\bar{y} = \bar{y}\sum_{(i=1)}^m x^{(i)} = m\bar{y}·\bar{x} = \bar{x}\sum_{(i=1)}^m y^{(i)} = \sum_{(i=1)}^m \bar{x}y^{(i)} = \sum_{(i=1)}^m \bar{x}\bar{y}$$
得：
$$a = \frac{\sum_{i=1}^m (x^{(i)}y^{(i)} - x^{(i)}\bar{y} - \bar{x}y^{(i)}) + \bar{x}\bar{y}}{\sum_{i=1}^m ((x^{(i)})^2 - \bar{x}x^{(i)} - \bar{x}x^{(i)} + {\bar{x}}^2)}$$

$$a = \frac{\sum_{i=1}^m(x^{(i)} - \bar{x})(y^{(i)} - \bar{y})}{\sum_{i=1}^m(x^{(i)} - \bar{x})^2}$$

结果：
$$a = \frac{\sum_{i=1}^m(x^{(i)} - \bar{x})(y^{(i)} - \bar{y})}{\sum_{i=1}^m(x^{(i)} - \bar{x})^2}$$

$$b = \bar{y} - a\bar{x}$$
