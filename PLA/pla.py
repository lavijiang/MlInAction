import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = [];labelMat = []
    data = np.loadtxt('../data/plaData.txt', delimiter='\t')
    dataMat = data[:, 0:2]
    labelMat = data[:, 2]
    return dataMat, labelMat

def plotFit(weights,b):
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
        else:
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-b - weights[0] * x) / weights[1]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def calculateLabel(X, w, b):
    y = np.dot(X, w) + b
    return y

def perceptron(X,labels,w,b=0,r=1):
    Y = calculateLabel(X,w,b)   #求出预测值
    Y2 = Y*labels
    errIndexs = np.where(Y2<=0)[0] #求出yi(w*xi+b)<=0的索引
    print(np.where(Y2<=0))
    if np.size(errIndexs) == 0:
        return w, b
    else:
        x = X[errIndexs[0]]
        y = labels[errIndexs[0]]
        w = w + np.dot(x, y) * r
        b = b + r * y
        return perceptron(X, labels, w, b, r)

data,labels = loadDataSet()
w,b=perceptron(data,labels,[0,0])
plotFit(w,b)


