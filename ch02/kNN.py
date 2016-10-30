# -*- coding: utf-8 -*-
# time: 2016.8.21

import numpy
from numpy import *
import operator  # 运算符模块
import math


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = createDataSet()
print('group:', group)
print('labels:', labels)


# k-近邻算法
def classify0(inX, dataSet, labels, k):  # 输入向量， 输入的训练样本， 标签向量
    '''

    :param inX: 输入的测试向量
    :param dataSet: 训练样本集
    :param labels: 训练样本标签
    :param k: top k最相近的
    :return:
    '''

    # shape 返回矩阵的[行数，列数]
    # shape[0]获取数据集的行数
    # 行数就是样本的数量
    dataSetSize = dataSet.shape[0]

    '''
    求距离： 根号(x^2+y^2)
    '''

    # tile属于numpy模块下的函数
    # tile(A, reps) 返回一个shape=reps的矩阵，矩阵的每个元素是A
    # 比如A=[0, 1, 2], 那么，tile(A, reps)=[0, 1, 2, 0, 1, 2]
    # tile(A, (2, 2)) = =[[0, 1, 2, 0, 1, 2],
    #                     [0, 1, 2, 0, 1, 2]]
    # 比如inx = [0, 1], dataset就用函数返回的结果，那么
    # tile(inx, (4, 1)) = [[0.0, 1.0],
    #                       [0.0, 1.0],
    #                       [0.0, 1.0],
    #                       [0.0, 1.0]]
    # 作差之后
    # diffMat = [[0.0, -0.1],
    #            [-1.0, 0.0],
    #            [0.0, 1.0],
    #            [0.0, 0.9]]

    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 把待分类数据向量复制成与训练集等阶，对应项求差
    # print (diffMat)
    # diffMat是输入样本与每个训练样本的差值， 然后对其每个x和y的差值进行平方运算
    # diffMat 是一个矩阵，矩阵**2表示对矩阵中的每个元素进行**2操作，即平方
    # sqDifffMat = [[1.0, 0.01],
    #               [1.0, 0.0],
    #               [0.0, 1.0],
    #               [0.0, 0.81]]
    sqDiffMat = diffMat ** 2

    # axis=1表示按照横轴，sum表示累加，即按照行进行累加
    # sqDistances = [[1.01],
    #                [1.0],
    #                [1.0],
    #                [0.81]]

    sqDistances = sqDiffMat.sum(axis=1)  # #axis=1，将一个矩阵的每一行向量相加
    # 对平方和进行开根号
    distances = sqDistances ** 0.5

    # 按照升序进行快速排序，返回的是元素组的下标
    # 比如，x = [30, 10, 20, 40]
    # 升序排序后应该是[10, 20, 30, 40], 他们的原下标是[1, 2, 0, 3]
    # 那么numpy.argsort(x)=[1, 2, 0, 3]
    sortedDistIndicies = distances.argsort()  # 返回结果是索引值
    classCount = {}

    # 投票过程，就是统计前k个最近的样本所属类别包含的样本个数
    for i in range(k):  # 遍历前K个样本训练集
        # index = sortedDistances[i]是第i个最相近的样本下标
        # voteIlabel = labels[index]是样本index对应的分类结果('A' or 'B')
        voteIlabel = labels[sortedDistIndicies[i]]  # 对应分类
        # classCount.get(voteIlabel, 0)返回voteIlabel的值，如果不存在，则返回0
        # 然后将票数增1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 奇数

    # 把分类结果进行排序，然后返回得票数最多的分类结果
    sortedclassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedclassCount[0][0]  # 返回统计最多的分类


# print classify0([0, 0], group, labels, 3)

'''使用k-近邻算法改进约会网站的配对效果'''


# 1、准备数据
# 将文本记录到转换Numpy的解析程序
def file2matrix(filename):
    '''
    从文件中读入训练数据，并存为矩阵
    '''
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)  # 得到文件行数
    returnMat = zeros((numberOfLines, 3))  # 创建一个2维矩阵用于存放训练样本数据，每一行存放3个数据
    classLabelVector = []  # 创建一个一维数组用于存放训练样本标签
    index = 0

    # 解析文件数据到列表
    for line in arrayOfLines:
        line = line.strip()  # 把回车符号去掉
        listFromLine = line.split('\t')  # 把每一行数据用\t分割
        # 把分割好的数据放至数据集，其中index是样本数据的下标，就是放到第几行
        returnMat[index, :] = listFromLine[0:3]
        # 把样本对应的标签放至标签集，顺序与样本集对应
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

# print('2维矩阵\n:', datingDataMat)
# print('标签集\n：', datingLabels[0:20])

'''
# 分析数据，使用Matplotlib创建散点图
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()  # 准备一张白纸
ax = fig.add_subplot(111)  # 使用fig ， 添加一张子图
# 使用散点图，使用datingDataMat矩阵的第二、第三列数据，分别表示特征值“玩视频游戏所耗时间百分比”和“每周所消耗的冰淇淋公升数”
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()
'''

def autoNorm(dataSet):
    '''
    训练数据归一化

    '''
    # 获取数据集中每一列的最小数值
    # 以createDataSet()中的数据为例，group.min(0)=[0, 0]
    minVals =dataSet.min(0)
    # 获取数据集中每一列的最大数值
    # group.max(0) = [1, 1.1]
    maxVals = dataSet.max(0)
    # 最大值与最小值的差值
    ranges = maxVals - minVals
    # 创建一个与dataSet同shape的全0矩阵，用于存放归一化后的数据
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]  # 获取数据集行数
    # 把最小值扩充为与dataSet同shape，然后作差
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 把最大最小差值扩充为dataSet同shape，然后作商，是指对应元素进行除法运算，而不是矩阵除法
    # 矩阵除法在numpy中要用linalg.solve(A, B)
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

normMat, ranges, minVals = autoNorm(datingDataMat)
print('归一化结果：')
print( normMat, ranges, minVals)


def datingClassTest():
    # 将数据集中10%的数据留作测试用，其余的90%用于训练
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 加载测试数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with : %d, the real answer is : %d, the result is :%s' % (classifierResult, datingLabels[i], classifierResult == datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print('the total error rate is: %f' % (errorCount/float(numTestVecs)))
    print errorCount

# datingClassTest()


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input('percentage of time spent playing video games?'))
    ffMiles = float(raw_input('frequent flier miles earned per year?'))
    iceCream = float(raw_input('liters of ice cream consumed per year？'))  # raw_input输入后返回字符串
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print('you will probably like this person:', resultList[classifierResult - 1])

classifyPerson()
