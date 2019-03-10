

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris   #导入数据集iris
from sklearn.linear_model import LogisticRegression
from load_mnist import load_mnist

iris = load_iris()  #载入数据集
#print(iris.data)

local_url='D:/data_for_influence/zhejiang2014.csv'
data_sets = load_mnist(local_url,validation_size=1000)
print(data_sets.train.x)
print(data_sets.train.labels)
print(len(data_sets.train.labels))

# print(iris.target)          #输出真实标签
# print(len(iris.target))      #150个样本 每个样本4个特征
# print (iris.data.shape)
# 获取花卉两列数据集

X_train = data_sets.train.x
Y_train = data_sets.train.labels
X_test = data_sets.test.x
Y_test = data_sets.test.labels
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

#逻辑回归模型
lr = LogisticRegression(C=1e5)
lr.fit(X_train,Y_train)
A=lr.predict(X_test)
print('Accuracy of LR Classifier:',lr.score(X_train,Y_train))
print('Accuracy of LR Classifier:',lr.score(X_test,Y_test))
influences = lr.get_loo_influences()

