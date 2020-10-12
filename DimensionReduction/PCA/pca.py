"""
伪代码：
    去除平均值
    计算协方差矩阵
    计算协方差矩阵的特征值和特征向量
    将特征值从大到小排序
    保留最上面的N个特征向量
    将数据转换到上述N个特征向量构建的新空间中
"""
import numpy as np

def loadDataSet(filename, delim="\t"):
    f = open(filename)
    strArr = [line.strip().split(delim) for line in f.readlines()]
    arr = [list(map(float, line)) for line in strArr]
    return np.mat(arr)

def pca(dataMat, topNfeat=9999999):
    #  去平均值
    meanVals = np.mean(dataMat, axis=0)
    meanRem = dataMat - meanVals
    #  计算协方差矩阵
    covMat = np.cov(meanRem, rowvar=False)
    #  计算协方差矩阵的特征值和特征向量
    eigVals, eigVecs = np.linalg.eig(np.mat(covMat))
    #  特征值特征向量排序
    eigValInd = np.argsort(eigVals)
    #  选取特征值
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    #  选取对应的特征向量
    reEigVecs = eigVecs[:,eigValInd]
    #  将数据转换到新的空间
    lowDimData = meanRem*reEigVecs
    #  数据重构
    reconMat =(lowDimData * reEigVecs.T) + meanVals
    return lowDimData, reconMat

data = loadDataSet("../dataset/testSet.txt")
lowdimdata, reconMat = pca(dataMat=data, topNfeat=1)

