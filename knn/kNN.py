from numpy import *
import operator

def createDataSet():
    group = array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#inX(待分类向量)、dataSet(测试集)、labels(测试集对应的标签)
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]              #取行数
    #距离计算
    diffMat = tile(inX,(dataSetSize,1)) - dataSet     #分类向量在行维度复制为矩阵
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)         #行维度加和
    #print(sqDistances)
    distances = sqDistances**0.5
    #print(distances)
    sortedDistIndicies = distances.argsort()    #将索引从小到大排序
    print(sortedDistIndicies)
    classCount = {}                             #字典
    #选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #items将一个字典以迭代器的形式返回
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #for k,v in sortedClassCount:
    #   print(k,v)
    return sortedClassCount[0][0]

#groups,labelss = createDataSet()
#print(classify0([0,0],groups,labelss,3))

#将文件解析为测试数据和标签
def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    #文件行数
    numberOfLines = len(arrayOlines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    #解析文件数据到列表
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#归一化特征值 newValue = (oldValue-min)/(max-min)
def autoNorm(dataSet):
    #每列的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals



