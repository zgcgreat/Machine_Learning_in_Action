# -*- coding: utf-8 -*-
# time: 2016.9.17

from math import log
import operator
import treePlotter

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)   # 数据集实例总数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():  # 为所有可能的分类创建字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)   # 以2为底求对数
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

myDat, labels = createDataSet()
print(myDat)

shannonEnt = calcShannonEnt(myDat)
print(shannonEnt)


'''
划分数据集
    依据给定特征划分数据集，axis表示第几个特征，value代表该特征所对应的值，返回的是划分后的数据集
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:  # 抽取
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

print(splitDataSet(myDat, 1, 1))


# 选择最好的数据集（特征）划分方式，返回最佳特征下标
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征个数
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;bestFeature = -1
    for i in range(numFeatures):    # 遍历特征  第i个
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]  # 第i个特征取值集合
        uniqueVals = set(featList)

        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 该特征划分所对应的entropy
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            # 计算最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

print(chooseBestFeatureToSplit(myDat))
print(myDat)



# 多数表决的方法觉定叶子节点的分类，当所有的特征全部用完时仍属于多类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # operator中的排序函数
    return sortedClassCount[0][0]


# 创建数的函数代码   python中用字典类型来存储树的结构，返回的结果是my-tree-字典
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 类别完全相同则停止继续划分，返回类标签-叶子节点
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)   # 遍历完所有的特征时返回出现次数最多的
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}    # 字典类型存储树的信息
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  # 得到列表包含所有的属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]   # 复制类标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

myTree = createTree(myDat, labels)
print(myTree)


treePlotter.createPlot()