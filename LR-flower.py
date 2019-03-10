

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris   #导入数据集iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()  #载入数据集
print(iris.data)

print(iris.target)          #输出真实标签
print(len(iris.target))      #150个样本 每个样本4个特征
print (iris.data.shape)
# 获取花卉两列数据集
DD = iris.data
X = [x[0] for x in DD]
print(X)
Y = [x[1] for x in DD]
print(Y)
print(iris.data.shape)

X = iris.data[:, :2]   #获取花卉两列数据集
Y = iris.target

#逻辑回归模型
lr = LogisticRegression(C=1e5)
lr.fit(X,Y)
A=lr.predict(X)
print('Accuracy of LR Classifier:',lr.score(X,Y))

#meshgrid函数生成两个网格矩阵
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#pcolormesh函数将xx,yy两个网格矩阵和对应的预测结果Z绘制在图片上
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(8,6))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

#绘制散点图
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.scatter(X[100:, 0], X[100:, 1], color='green', marker='s', label='Virginica')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(loc=2)
plt.show()