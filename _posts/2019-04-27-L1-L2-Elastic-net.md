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

# Lp 范数

$$||x||_p = (\sum_{i=1}^n|x_i|^p)^{\frac{1}{p}}$$

称为Lp范数

## L2 正则项

Ridge中：

$$\sum_{i=1}^n\theta^2$$

为L2正则项

## L1 正则项

LASSO中：

$$\sum_{i=1}^n|\theta|$$

为L1正则项

## Ln 正则项

$$\sum_{i=1}^m|\theta|^n$$

## L0 正则项

$$J(\theta) = MSE(y,\hat{y};\theta) + min\{number-of-non-zero-\theta\}$$

其中的$min\{number-of-non-zero-\theta\}$代表theta中非零的个数尽可能小，称为L0正则项

**实际上我们很少使用L0正则，实际用L1取代，因为L0正则的优化是个NP难的问题**

## 弹性网 Elastic Net

$$J(\theta) = MSE(y,\hat{y};\theta) + r\alpha\sum_{i=1}^n|\theta_i| + \frac{1-r}{2}\alpha\sum_{i=1}^n\theta_i^2$$

同时结合了岭回归和LASSO的优势

计算量足够应该优先选择岭回归，不足时优先选择弹性网
