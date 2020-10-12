'''
@Project:MachineLearningAction

@Author:lincoln

@File:bayes

@Time:2020-06-11 14:22:48

@Description:
'''
import BaseFunc as bf
import numpy as np

docList, tag = bf.loadDataset()

def TrainBayesModel(docList, tag):
    # 数据集大小
    numData = len(docList)
    # 词典列表
    vocabList = bf.createVocabList(docList)
    #文本向量
    docVecList = []
    #类别统计
    categories = {}
    #模型参数
    priorProb = {}
    rearProb = {}


    #将文本数据集转化为向量
    for doc in docList:
        docVec = bf.createDocVec(vocabList, doc)
        docVecList.append(docVec)

    #统计类别及对应的数据数目
    for category in tag:
        if category not in categories:
            categories[category] = 0
        categories[category] += 1

    #模型参数统计
    for category, times in categories.items():
        priorProb[category] = times/numData
        rearProb[category] = [0] * len(vocabList)
    for d, l in zip(docVecList, tag):
        for word, index in zip(d,range(len(d))):
            if word >0:
                rearProb[l][index] += 1
    for key, count in rearProb.items():
        count = np.array(count)/categories[key]
        rearProb[key] = count

    return vocabList, priorProb, rearProb


#上面我训练除了模型的参数，那么如何用得到的模型参数对新的数据做估计呢？

def classify(vocabList, priorProb, rearProb, doc):
    docvec = bf.createDocVec(vocabList, doc)
    largestProb = 0.0
    bestCategory = -1
    for category in priorProb.keys():
        prob = priorProb[category]
        for word, index in zip(docvec, range(len(docvec))):
            if word == 1:
                prob = prob * rearProb[category][index]
        if prob>largestProb:
            bestCategory=category
            largestProb = prob
    return bestCategory, largestProb


vocabList, priorProb, rearProb = TrainBayesModel(docList, tag)
print(vocabList, priorProb, rearProb)
doc = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
c, p = classify(vocabList, priorProb, rearProb,doc)
print("category:", c, "prob:", p)

