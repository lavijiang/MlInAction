from numpy import *

#创建实验样本 postingList是词条分割后的文档集合，classVec类别标签
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak','how','to','stop','him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0,1,0,1,0,1] #1代表侮辱文字 0代表正常
    return postingList,classVec

#创建一个包含在所有文档中出现的不重复词的集合
def createVocabList(dataSet):
    #创建空集
    vocabSet = set([])
    for document in dataSet:
        #创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#创建文档向量 vocabList词汇表 inputSet文档
def setOfWords2Vec(vocabList,inputSet):
    #创建一个其中所含元素都为0的向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!",word)
    return returnVec

#朴素贝叶斯分类器训练函数



