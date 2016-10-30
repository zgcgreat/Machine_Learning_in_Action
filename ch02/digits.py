# -*- coding: utf-8 -*-
# time: 2016.9.3

from numpy import *
import operator
from os import listdir


def img2vector(filename):
    """
    将图片转化为01矩阵
    每张图片是32*32像素，也就是一共1024个字节
    因此转换的时候，每行表示一个样本，每个样本含1024个字节
    """
    # 每个样本数据是1024=32*32个字节
    returnVect = zeros((1, 1024))
    fr = open(filename)
    # 循环读取32行，32列
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

'''
testVector = img2vector('./digits/testDigits/0_13.txt')
print testVector[0, 0:31]
print testVector[0, 32:63]
'''

# 手写识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    # 加载训练数据
    trainingFileList = listdir('./digits/trainingDigits/')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 从文件中解析出当前图像的标签，也就是数字几
        fileNameStr = trainingFileList[i]  # 分割得到标签，从文件名解析得到分类数据
        fileStr = fileNameStr.split('.')[0]  # 去掉 .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('./digits/trainingDigits/%s' % fileNameStr)

    # 加载测试数据
    testFileList = listdir('./digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer id: %d, the predict result is： %s" % (classifierResult, classNumStr, classifierResult==classNumStr))
        if (classifierResult !=classNumStr): errorCount += 1.0
    print ("\nthe total number of error is: %d / %d" % (errorCount, mTest))
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))



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



if __name__ == "__main__":
    handwritingClassTest()
