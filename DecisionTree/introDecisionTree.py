'''
@Project:MachineLearningAction

@Author:lincoln

@File:tree

@Time:2020-06-08 15:42:49

@Description:该描述了构建决策树的过程
'''
import math
import operator

def createDataset():
    '''
    本函数用于产生实验数据
    :return:实验数据、属性名称
    '''
    mydat = [[1, 1, "yes"],
             [1, 1, "yes"],
             [1, 0, "no"],
             [0, 1, "no"],
             [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return mydat, labels

def calShannonEnt(dataset):
    '''
    计算数据集的信息熵
    :param dataset: 数据集
    :return: 信息熵
    '''
    numEntities = len(dataset)
    labelDict = {}
    for entity in dataset:
        label = entity[-1]
        if label not in labelDict:
            labelDict[label] = 0
        labelDict[label] += 1
    entropy = 0.0
    for key,value in labelDict.items():
        prob = value/numEntities
        entropy -= prob * math.log(prob,2)
    return entropy

#划分数据集
def splitDataset(dataset, axis, value):
    '''
    在特征axis上按照给定value划分数据集
    :param dataset:  给定的数据集
    :param axis: 特征索引
    :param value: 特征值
    :return: axis属性上与value相等的实例列表，axis特征被剔除
    '''
    subDataset = []
    for entity in dataset:
        if entity[axis] == value:
            reducedEntity = entity[:axis]
            reducedEntity.extend(entity[axis+1:])
            subDataset.append(reducedEntity)
    return subDataset


def chooseBestFeatureToSplitDataset(dataset):
    '''
    选择最优划分数据集的特征
    :param dataset: 数据集
    :return: 最优特征
    '''
    #计算特征数目
    numFeatures = len(dataset[0])-1
    #未划分的数据集的熵
    baseEntropy = calShannonEnt(dataset)
    #最优划分特征
    bestSplitFeature = -1
    #最大信息增益
    largestInfoGain = 0.0
    for i in range(numFeatures):
        FeaValues = [entity[i] for entity in dataset]
        FeaValues = set(FeaValues)
        splitedEntropy = 0.0
        for value in FeaValues:
            subDataset = splitDataset(dataset, i, value)
            splitedEntropy += (len(subDataset)/len(dataset))*calShannonEnt(subDataset)
        infoGain = baseEntropy - splitedEntropy
        if infoGain > largestInfoGain:
            largestInfoGain = infoGain
            bestSplitFeature = i
    return bestSplitFeature


def majorityCnt(classList):
    '''
    该函数用于返回类别列表中数目最多的类别
    :param classList:类别列表
    :return:数目最多的类别
    '''
    classCnt = {}
    for categ in classList:
        if categ not in classCnt:
            classCnt[categ] = 0
        classCnt[categ] += 1
    sortedClassCnt = sorted(classCnt.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCnt[0][0]

def createDecisTree(dataset,labels):
    '''
    该函数用于构建决策树
    :param dataset:数据集
    :param Labels:属性名称
    :return:决策树
    '''
    classList = [entity[-1] for entity in dataset]
    # 数据实例同属于同一个类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有的特征
    if len(labels) == 0:
        return majorityCnt(classList)

    # 选择最优的特征
    bestFeature = chooseBestFeatureToSplitDataset(dataset)
    # 最优特征值集合
    uniqueFeatureValues = set([entity[bestFeature] for entity in dataset])

    bestFeatureText = labels[bestFeature]
    myTree = {bestFeatureText: {}}

    subLabels = labels[:bestFeature]
    subLabels.extend(labels[bestFeature+1:])
    # 递归构建决策树
    for value in uniqueFeatureValues:
        #subLabels = subLabels[:]
        myTree[bestFeatureText][value] = \
            createDecisTree(splitDataset(dataset, bestFeature, value),subLabels)
    return myTree

dataset, labels = createDataset()
tree = createDecisTree(dataset, labels)
print(tree)

#如何使用我们得到的决策树进行分类呢？
def classify(decisiontree, labels, feature):
    '''
    本函数用来执行决策树分类
    :param decisiontree:在训练集上得到的决策树
    :param labels:待分类实例特征名称
    :param feature:待分类实例特征
    :return: 待分类实例标签
    '''
    featuretext = list(decisiontree.keys())[0]
    featureindex = labels.index(featuretext)
    nextDict = decisiontree[featuretext]
    for key in nextDict.keys():
        if feature[featureindex] == key:
            if type(nextDict[key]).__name__ == "dict":
                classLabel = classify(nextDict[key], labels, feature)
            else:
                classLabel = nextDict[key]
    return classLabel

print(classify(tree, labels, [0,1]))


#上面的决策树都是存储在内存中的，每次使用都要从新训练，浪费资源。
# 我们可以将训练好的决策树存储在硬盘上，下次需要的时候直接从硬盘加载。
import pickle

# 存
def storeTree(mytree, filename):
    fw = open(filename, 'wb')
    pickle.dump(mytree, fw)
    fw.close()


# 取
def getTree(TreeName):
    fr = open(TreeName,"rb")
    return pickle.load(fr)


storeTree(tree, "decisTree.txt")
model = getTree("decisTree.txt")
print(model)
print(type(model))










