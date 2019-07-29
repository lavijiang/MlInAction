import bayes

listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print(myVocabList)
returnVec = bayes.setOfWords2Vec(myVocabList,listOPosts[0])
print(listOPosts[0])
print(returnVec)