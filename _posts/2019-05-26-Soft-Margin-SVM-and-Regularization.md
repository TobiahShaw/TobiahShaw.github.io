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

# Soft Margin SVM 和 SVM 的正则化

- 具有容错能力

在hard margin SVM 中 $s.t.\quad y^{(i)}(w^Tx^{(i)} + b) \geq 1$ 表示，所有点都落在两条平行直线$w^Tx + b = 1$ 和 $w^Tx + b = -1$之外

在soft margin SVM中，我们把条件稍作宽松 $s.t.\quad y^{(i)}(w^Tx^{(i)} + b) \geq 1 - \zeta_i \quad \zeta_i \geq 0$

我们又不希望$\zeta_i$太大，我们就需要这样表征$$min\frac{1}{2}||w||^2 + \sum_{i=1}^m \zeta_i$$

## Soft Margin SVM 数学表达

- L1正则

$$min\frac{1}{2}||w||^2 + C\sum_{i=1}^m \zeta_i$$

$$s.t.\quad y^{(i)}(w^Tx^{(i)} + b) \geq 1 - \zeta_i \quad \zeta_i \geq 0$$

- L2正则

$$min\frac{1}{2}||w||^2 + C\sum_{i=1}^m \zeta_i^2$$

$$s.t.\quad y^{(i)}(w^Tx^{(i)} + b) \geq 1 - \zeta_i \quad \zeta_i \geq 0$$

## 正则化

可以看到，Soft Margin SVM 其实是在 Hard Margin SVM 后面加上了正则项，只是在 Soft Margin SVM 称之为容错能力
