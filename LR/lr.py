import numpy as np
import matplotlib.pyplot as plt
import random

def loadDataSet():
    dataMat = [];labelMat = []
    data = np.loadtxt('testSet.txt', delimiter='\t')
    dataMat = data[:,0:2]
    dataMat = np.insert(dataMat, 0, 1, axis=1)
    labelMat = data[:,2]
    return dataMat,labelMat

#防止overflow
def sigmoid(inX):
    if inX >= 0:
        return 1.0/(1+np.exp(-inX))
    else:
        return np.exp(inX)/(1+np.exp(inX))

#梯度上升,求最大值(批量)
def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat-h)
        weights = weights+alpha*dataMatrix.transpose()*error
    return weights

#随机梯度上升,求最大值(一次计算一个)
#如果存在样本点不能正确分类，迭代时会造成参数震荡
def stocGradAscent0(dataMatrix,classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha*error*dataMatrix[i]
    return weights

#改进的随机梯度上升
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            #学习率衰减
            alpha = 4/(1.0+j+i)+0.01
            #随机选取样本
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha*error*dataMatrix[dataIndex[randIndex]]
            del(dataIndex[randIndex])
    return weights

#画出决策边界
def plotBestFit(weights):
    #将numpy矩阵转换为数组
    #weights = wei.getA()
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#a,b = loadDataSet()
#plotBestFit(stocGradAscent1(np.array(a),b))

def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else :
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    #模型训练
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainingWeights = stocGradAscent1(np.array(trainingSet),trainingLabels,500)
    #观察训练效果
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if  int(classifyVector(np.array(lineArr),trainingWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is [%f]" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after [%d] iterations the average error rate is [%f]" % (numTests, errorSum/float(numTests)))

multiTest()

