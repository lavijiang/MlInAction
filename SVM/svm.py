import numpy as np
import pandas as pd

def loadDataSet():
    dataMat = [];labelMat = []
    data = np.loadtxt('testSet.txt', delimiter='\t')
    dataMat = data[:, 0:2]
    dataMat = np.insert(dataMat, 0, 1, axis=1)
    labelMat = data[:, 2]
    for i in range(len(labelMat)):
        if labelMat[i] == 0:
            labelMat[i] = -1
    return dataMat, labelMat

def selectJrand(i,m):
    j = i
    while j == i:
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

#SMO First Edition