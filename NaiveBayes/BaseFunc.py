'''
@Project:MachineLearningAction

@Author:lincoln

@File:DocClassification

@Time:2020-06-11 12:55:15

@Description: 使用朴素贝叶斯进行文档分类
'''
def loadDataset():
    '''
    该函数用于生成实验数据集
    :return:
    '''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    tag = [0,1,0,1,0,1]
    return postingList, tag


def createVocabList(dataSet):
    '''
    该函数用于统计数据集此表用于后验概率估计
    :param dataSet:实验数据集
    :return:数据集包含的词汇列表
    '''
    vocbSet = set([])
    for doc in dataSet:
        vocbSet = vocbSet | set(doc)
    return list(vocbSet)


def createDocVec(Vocab, doc):
    '''
    本函数用于将文档转换为文档向量
    :param Vocab: 数据集词汇列表
    :param doc: 文本文档
    :return: 文档向量
    '''
    docvec = [0]*len(Vocab)
    for word in doc:
        if word in Vocab:
            docvec[Vocab.index(word)] = 1
    return docvec


