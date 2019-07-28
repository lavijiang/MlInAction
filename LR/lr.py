import numpy as np

def loadDataSet():
    dataMat = [];labelMat = []
    data = np.loadtxt('testSet.txt', delimiter='\t')
    dataMat = data[:,0:2]
    dataMat = np.insert(dataMat, 2, 1, axis=1)
    labelMat = data[:,2]
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

#梯度上升,求最大值
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


a,b = loadDataSet()
print(gradAscent(a,b))