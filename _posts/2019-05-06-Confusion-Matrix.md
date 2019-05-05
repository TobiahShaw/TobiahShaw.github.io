
# 准确度陷阱和混淆矩阵

## 分类准确度问题

如果有个二分类系统预测准确度为99.9%

如果其中一个分类的发生概率只有0.1%

那么系统只需要全部预测另一个分类就能达到99.9%的准确率

**对于极度偏斜的数据（Skewed Data），只使用分了准确度是远远不够的**

## 混淆矩阵

### 对于二分类问题

![confusion matrix](..\assets\img\ClassificationPerformanceMeasures\confusion-matrix.png)
