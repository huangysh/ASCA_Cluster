# -*- coding: utf-8 -*-

# **********************************************************************************************************************
# MIT License

# Copyright (c) 2020 School of Environmental Science and Engineering, Shanghai Jiao Tong University

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ----------------------------------------------------------------------------------------------------------------------
# This file is part of the ASCA Algorithm, it is used for  spatial point clustering analysis. This model contains mainly
# three parts, they are points trend analysis, point cluster analysis and spatial visualization.
#
# Author: Yuansheng Huang
# Date: 2020-06-18
# Version: V 1.2

# Literature
# ==========
# Yuansheng Huang, Peng Li, Yiliang He: To centralize or to decentralize? A systematic framework for optimizing
# rural wastewater treatment investment

# Clark and Evans, 1954; Gao, 2013

# **********************************************************************************************************************

# General import
import arcpy
import math
import functools
import numpy as np
from scipy.spatial import Delaunay
from functools import cmp_to_key


# --------------------------read out file--------------------------


def getFeatureName(folder):
    """读取文件夹中所有的.shp文件名（含后缀），并存为list"""
    arcpy.env.workspace = folder
    featureName = []
    for feature in arcpy.ListFeatureClasses():
        featureName.append(feature)

    return featureName


def readArea(areaShape):
    """
    读取shapefile中研究区域的面积

    输入参数
    areaShape: 研究区域矢量地图，用于读取面积值。

    输出参数
    areaValue: 研究区域面积，米。
    """
    areaList = []
    rows = arcpy.SearchCursor(areaShape)
    fields = arcpy.ListFields(areaShape)
    for row in rows:
        for field in fields:
            if field.name == "area" or field.name == "AREA":
                AREA = row.getValue(field.name)
        areaList.append(AREA)
    areaValue = np.sum(areaList)
    return areaValue


def readObstacle(obstacle):
    """
    从shapefile线数据中读取研究区域空间障碍（线段）的起始点坐标，用于删除DT边列表中与障碍线段相交的边。

    输入参数：
    obstacle: 空间障碍shapefile数据，将所有需考虑的障碍（道路，河流，分水岭等）合并为一个文件，且需在vertex处打断障碍以得到起始点坐标。

    输出参数
    obstacleList: 障碍列表
    """
    obstacleList, rows, fields = [], arcpy.SearchCursor(obstacle), arcpy.ListFields(obstacle)
    for row in rows:
        for field in fields:
            if field.name == "START_X" or field.name == "X_START":
                S_X = row.getValue(field.name)
            elif field.name == "START_Y" or field.name == "Y_START":
                S_Y = row.getValue(field.name)
            elif field.name == "END_X" or field.name == "X_END":
                E_X = row.getValue(field.name)
            elif field.name == "END_Y" or field.name == "Y_END":
                E_Y = row.getValue(field.name)
        obstacleList.append([[S_X, S_Y], [E_X, E_Y]])

    if len(obstacleList) == 0:
        raise Exception("EMPTY LIST: YOU GOT AN EMPTY LIST!!! PLEASE CHECK INPUT FILE!")
    return obstacleList


def readSpatialPoint(pointShape):
    """
    读取空间点坐标数据，并保存为列表

    输入参数
    pointShape: Path to point shapefile

    输出参数
    pointList: 空间点坐标列表
    spatialRef: 空间参考
    """
    pointList, rows, fields, ID = [], arcpy.SearchCursor(pointShape), arcpy.ListFields(pointShape), 0
    spatialRef = arcpy.Describe(pointShape).spatialReference
    for row in rows:
        for field in fields:
            if field.name == "POINT_X":
                X = row.getValue(field.name)
            if field.name == "POINT_Y":
                Y = row.getValue(field.name)
            if field.name == "Q":
                Q = row.getValue(field.name)
            if field.name == "RASTERVALU":
                H = row.getValue(field.name)
        pointList.append([ID, X, Y, Q, H])
        ID += 1

    if len(pointList) < 1:
        raise Exception("EMPTY LIST: YOU GOT AN EMPTY LIST, PLEASE CHECK YOUR INPUT FILE!!!")
    return pointList, spatialRef


def checkList(pointlist):
    """
    检查空间点列表，删除重复的点。这里的重复是指，坐标重复。

    输入参数
    pointlist: 空间点坐标列表

    输出参数
    output: 删除重复后的空间点坐标列表
    """
    point = [i[1:] for i in pointlist]
    idList = [i[:1] for i in pointlist]

    output1 = []
    for i in range(len(point)):
        if point[i] not in output1:
            output1.append(point[i])
        else:
            idList.remove(idList[i])

    output = []
    for i in range(len(point)):
        output.append(idList.extend(point[i]))
    return output


def getNearestDistance(pointList):
    """
    此函数用于计算各点到其最近点间的距离，并返回距离列表。

    输入参数
    pointList: 空间点坐标列表。

    输出参数
    nearestDistanceList: 最近距离列表。
    """
    nearestDistanceList = []
    for i in range(len(pointList)):
        distanceToPoint = []
        for j in range(len(pointList)):
            if i != j:
                length2D = math.hypot(pointList[i][1] - pointList[j][1], pointList[i][2] - pointList[j][2])
                heightDiff = pointList[i][4] - pointList[j][4]
                length3D = math.hypot(length2D, heightDiff)
                distanceToPoint.append(length3D)
            else:
                continue
        nearestDistance = min(distanceToPoint)
        nearestDistanceList.append(nearestDistance)

        if len(nearestDistanceList) < 1:
            raise Exception("EMPTY LIST: YOU GOT AN EMPTY LIST, PLEASE CHECK YOUR INPUT FILE!!!")
    return nearestDistanceList


def NNI(pointList, distanceList, areaValue):
    """
    用于计算空间点集的最邻近指数。输出值将用于定义空间点的分布模式，当NNI>1时，空间点集呈均匀分布，当NNI<1时，空间点集呈聚集分布。

    输入参数
    pointList: 空间点坐标列表
    nearestDistanceList: 最近距离列表
    areaValue: 研究区域面积，米。

    输出参数
    index:  空间点集的最邻近指数
    z_test: z检验数值
    """
    N = len(pointList)
    ran = 0.5 * math.sqrt(areaValue / N)

    sumD = np.sum(distanceList)
    SE = 0.26236 / (math.sqrt(N ** 2) / areaValue)
    indexValue = (sumD / N) / ran
    z_test = ((sumD / N) - ran) / SE
    return indexValue, z_test


# ----------------starting cluster------------------

def getDelaunayTriangle(pointList):
    """
    获取空间点集points的Delaunay Triangle (DT) 及DT的顶点索引和坐标。

    输入参数
    pointList: 空间点坐标列表

    输出参数
    triangleVertexIndex: DT顶点索引列表
    triangleVertexCoordinate: DT顶点坐标列表
    """
    pointListS = [i[1:3] for i in pointList]
    points = np.array(pointListS)
    DT = Delaunay(points)
    triangleVertexIndex = DT.simplices[:].tolist()

    triangleVertexCoordinate = []
    for T in triangleVertexIndex:
        triangle = []
        for v in T:
            triangle.append(pointList[v])
        triangleVertexCoordinate.append(triangle)
    return triangleVertexIndex, triangleVertexCoordinate


def unfoldList(nestedList):
    """
    用于展开嵌套列表，将在后续的算法中反复运用。

    输入参数
    nestedList: 嵌套列表

    输出参数
    unfoldedList: 展开后的列表
    """
    unfoldedList = [i for j in nestedList for i in j]
    return unfoldedList


def clusteredList(pointList, marker):  # todo. Is this function useful
    """
    根据列表中元素所包含的特定字符/元素，对列表中的元素进行分类。

    输入参数
    pointList: 列表
    marker: 用于分类的特殊标记（最好是按一定顺序排列）

    输出参数
    clusteredList: 输出，嵌套列表
    """
    clusteredList = []
    for item in pointList:
        clustered = []
        for m in marker:
            if m in item:
                clustered.append(item)
            clusteredList.append([m, clustered])
    return clusteredList


def uniqueListElement(listWithCopyElement):
    """
    删除列表（嵌套列表）中重复的元素，将在后续的算法中反复运用。
    """
    listWithUniqueElement = []
    for i in listWithCopyElement:
        if i not in listWithUniqueElement:
            listWithUniqueElement.append(i)
        else:
            continue
    return listWithUniqueElement


def getEdgeID(indexA, indexB):
    """
    获取边的ID号。indexA，indexB:分别为表顶点的ID号，即索引号。
    """
    if indexA == indexB:
        raise Exception("ERROR: Indexes point to the same point!!!")

    maxIndex, minIndex = max(indexA, indexB), min(indexA, indexB)
    edgeID = "V" + "_" + str(minIndex) + "_" + str(maxIndex)
    return edgeID


def getLength(pointA, pointB):
    """
    This function used to calculate the Euclidean distance in three dimensions.
    """
    length2D = math.hypot(pointA[1] - pointB[1], pointA[2] - pointB[2])
    heightDiff = pointA[4] - pointB[4]
    length = math.hypot(length2D, heightDiff)
    return length


def getEdgeLength(triangleVertexIndex, triangleVertexCoordinate):
    """
    用于获取DT网格的边长及边顶点在PointList列表中的索引号。

    输入参数
    triangleVertexIndex: getDelaunayTriangle函数输出参数，以三角形为单位。
    triangleVertexCoordinate: getDelaunayTriangle函数输出参数，以三角形为单位。

    输出参数
    triangleEdgeList: DT网格边长列表，其结构同输入。
    edgeList: 展开并去除重复值后的triangleEdgeList列表。
    """
    length, triangleEdgeList = len(triangleVertexCoordinate), []
    for i in range(length):
        triangleIndex = triangleVertexIndex[i]
        triangleCoordinate = triangleVertexCoordinate[i]
        edgeA = [getEdgeID(triangleIndex[0], triangleIndex[1]), min(triangleIndex[0], triangleIndex[1]),
                 max(triangleIndex[0], triangleIndex[1]), getLength(triangleCoordinate[0], triangleCoordinate[1])]
        edgeB = [getEdgeID(triangleIndex[0], triangleIndex[2]), min(triangleIndex[0], triangleIndex[2]),
                 max(triangleIndex[0], triangleIndex[2]), getLength(triangleCoordinate[0], triangleCoordinate[2])]
        edgeC = [getEdgeID(triangleIndex[1], triangleIndex[2]), min(triangleIndex[1], triangleIndex[2]),
                 max(triangleIndex[1], triangleIndex[2]), getLength(triangleCoordinate[1], triangleCoordinate[2])]
        edgesList = [edgeA, edgeB, edgeC]
        triangleEdgeList.append(edgesList)

    unfoldedList = unfoldList(triangleEdgeList)
    edgeList = uniqueListElement(unfoldedList)

    return triangleEdgeList, edgeList


# --------------------------delete global long edge--------------------------

def getGlobalEdgeStatistic(edgeList):
    """
    用于计算全局边长的统计量，全局边长均值和全局边长变异。

    输出参数
    globalEdgeMean, globalEdgeVariation: 全局边长均值和全局边长变异。
    """
    edgeLength = [i[-1] for i in edgeList]
    globalEdgeMean = np.mean(edgeLength)
    if len(edgeLength) >= 2:
        globalEdgeVariation = np.std(edgeLength, ddof=1)
    else:
        raise ZeroDivisionError
    return globalEdgeMean, globalEdgeVariation


def getFirstOrderEdges(pointList, edgeList):
    """
    获取各顶点的一阶邻域边。

    输出参数
    firstOrderEdges: 各点的一阶邻域边
    firstOrderPoints: 各点的一阶邻域点
    """
    firstOrderEdges, firstOrderPoints = [], []
    for point in pointList:
        index = point[0]
        firstOrderEdge, firstOrderPoin = [index], []
        for edge in edgeList:
            if index in edge[1:3]:
                firstOrderEdge.append(edge)
                firstOrderPoin.extend(edge[1:3])
            else:
                continue

        fop = list(set(firstOrderPoin))
        if index in fop:
            fop.remove(index)
        else:
            continue

        firstOrderPoint = [index]
        firstOrderPoint.extend(fop)

        firstOrderEdges.append(firstOrderEdge)
        firstOrderPoints.append(firstOrderPoint)
    return firstOrderEdges, firstOrderPoints


def getFirstOrderEdgesMean(firstOrderEdges):  # 20200618, updated
    """
    计算各顶点的一阶邻域均值。Pi为索引号

    输出参数
    firstOrderEdgesMean: 各点的一阶邻域边长均值。[[Pi, AVGi], [ ]…]
    """
    firstOrderEdgesMean = []
    for i in firstOrderEdges:
        edge = [x[3] for x in i[1:]]
        element = [i[0], np.mean(edge)]
        firstOrderEdgesMean.append(element)
    return firstOrderEdgesMean


def getGlobalCutValue(globalEdgeMean, globalEdgeVariation, firstOrderEdgesMean):
    """
    计算各顶点的全局约束准则，用于删除全局长边。

    输入参数
    globalEdgeMean, globalEdgeVariation: 全局边长均值和全局边长变异。
    firstOrderEdgesMean: 各点的一阶邻域边长均值。

    输出参数
    globalCutValueList: 全局约束准则列表
    """
    globalCutValueList = []
    for i in firstOrderEdgesMean:
        GCVi = globalEdgeMean + 0.5 * (globalEdgeMean / i[1]) * globalEdgeVariation
        element = [i[0], GCVi]
        globalCutValueList.append(element)
    return globalCutValueList


def getGlobalOtherEdge(edgeList, globalCutValueList):
    """
    获取DT网格中的全局其他边，直接从edgeList列表中删除全局长边。

    输入参数
    firstOrderEdges:
    globalCutValueList: 全局约束准则列表

    输出参数
    globalOtherEdgeList: 删除全局长边后的edgeList。
    """
    longEdge, globalOtherEdgeList = [], edgeList[:]
    for point in globalCutValueList:
        for edge in edgeList:
            if point[0] in edge[1:3] and edge[3] >= point[1]:
                if edge in globalOtherEdgeList:
                    globalOtherEdgeList.remove(edge)
                else:
                    continue
            else:
                continue
    return globalOtherEdgeList


def aggregation(edgeList):
    """
    用于获取孤立点以外的其他点所构成的点簇，每个点簇所包含的点为一个嵌套元素。在cluster函数中调用。
    此函数将嵌套列表中有相同元素的子列表合并，并将索引号较小的一个元素设置为两个子元素的并，较大一个设置为空列表[]。

    输入参数
    edgeList: 删除全局长边后的edgeList。

    输出参数
    indexList: 合并后的列表，嵌套列表，每个子列表表示一个点簇子列表的元素无为点索引号。
    """
    indexListX = [i[1:3] for i in edgeList]  # get index
    for i in range(len(indexListX)):
        for j in range(len(indexListX)):
            x = list(set(indexListX[i] + indexListX[j]))
            y = len(indexListX[j]) + len(indexListX[i])
            if i == j:
                break
            elif len(x) < y:
                indexListX[i] = x
                indexListX[j] = []

    indexList = []
    for i in indexListX:
        if len(i) > 1:
            indexList.append(i)
        else:
            continue
    return indexList


def cluster(pointList, indexList, marker):
    """
    给pointList中的元素添加标记，以区分各点簇。

    输入参数
    pointList: 空间点坐标列表
    indexList: aggregation函数输出值
    marker: 类簇标记，如“G”。

    输出参数
    pointList: 在每个元素尾部添加类簇标记的pointList，结构同输入。
    """
    clusterPointIndex = [i for j in indexList for i in j]
    for i in pointList:
        index = pointList.index(i)
        marker0 = marker + "0"
        if index not in clusterPointIndex:
            i.append(marker0)
        else:
            continue

    for lst in indexList:  # 标记其他点
        markerX = marker + str(indexList.index(lst) + 1)
        for i in pointList:
            for ele in lst:
                if ele == i[0]:
                    i.append(markerX)
                else:
                    continue
    return


# --------------------------删除局部长边--------------------------

def getSubgraphEdge(pointList, edgeList, indexList):
    """
    用于获取删除全局长边和障碍边后的所有子图，每个子图为一个元素，每个元素包含子图所有的边。次函数基于aggregation函数结果实现。

    输入参数
    pointList: 在每个元素尾部添加类簇标记的pointList，结构同输入。
    indexList: 合并后的列表，嵌套列表，每个子列表表示一个点簇子列表的元素无为点索引号。
    edgeList: 删除全局长边后的edgeList。

    输出参数
    subgraphEdgeList: 子图边列表
    subgraphVertexList: 子图顶点列表
    """
    subgraphVertexList = []
    for A in indexList:  # 获取子图顶点坐标
        vertex = [pointList[i] for i in A]
        subgraphVertexList.append(vertex)

    subgraphEdgeList = []
    for subgraphVertex in indexList:
        subgraphEdge = []
        for i in subgraphVertex:
            for j in edgeList:
                if i in j[1:3] and j not in subgraphEdge:
                    subgraphEdge.append(j)
                else:
                    continue
        subgraphEdgeList.append(subgraphEdge)
    return subgraphVertexList, subgraphEdgeList


def getSecondOrderEdges(subgraphVertexList, subgraphEdgeList):
    """
    计算子图各顶点的二阶邻域边长。

    输入参数
    subgraphVertexList, subgraphEdgeList: getSubgraphEdge函数输出值。

    输出参数
    subgraphSecondOrderEdgeMean: 各子图各顶点二阶邻域边长均值。
    """
    length, subgraphSecondOrderEdgeMean = len(subgraphVertexList), []
    for i in range(length):  # 获取一个子图，迭代
        subgraphVertex, subgraphEdge = subgraphVertexList[i], subgraphEdgeList[i]  # vertex and edge of subgraph.
        _, firstOrderPoints = getFirstOrderEdges(subgraphVertex, subgraphEdge)

        firstOrderPointList = [i[1:] for i in firstOrderPoints]
        indexList = [i[0] for i in firstOrderPoints]
        secondOrderMean = []
        for n in range(len(firstOrderPointList)):
            subgraphSecondOrderEdgeC, index,  = [], indexList[n]
            for p in firstOrderPointList[n]:
                for e in subgraphEdge:
                    if p in e[1:3]:
                        subgraphSecondOrderEdgeC.append(e)
            subgraphSecondOrderEdgeU = uniqueListElement(subgraphSecondOrderEdgeC)
            edgeLengthPi = [i[-1] for i in subgraphSecondOrderEdgeU]
            Pi_mean = np.mean(edgeLengthPi)
            secondOrderMean.append([index, Pi_mean])

        subgraphSecondOrderEdgeMean.append(secondOrderMean)
    return subgraphSecondOrderEdgeMean


def getSubgraphEdgeStatistic(subgraphVertexList, subgraphEdgeList):  # updated by Ethan Huang in 20200618
    """
    计算子图各顶点的一阶边长均值及局部边长平均变异。

    输入参数
    subgraphVertexList, subgraphEdgeList: getSubgraphEdge函数输出值。

    输出参数
    subgraphMeanVariation: 子图局部边长平均变异。
    """
    length, subgraphMeanVariation = len(subgraphVertexList), []
    for p in range(length):  # 迭代子图
        subgraphVertex, subgraphEdge = subgraphVertexList[p], subgraphEdgeList[p]  # vertex and edge of subgraph.
        firstOrderEdges, _ = getFirstOrderEdges(subgraphVertex, subgraphEdge)

        firstOrderEdgeList = [i[1:] for i in firstOrderEdges]
        firstOrderEdgeVariationList = []
        for edgeList in firstOrderEdgeList:
            firstOrderEdgeLength = [e[-1] for e in edgeList]  # 子图i中第n点的一阶邻域边长
            if len(firstOrderEdgeLength) >= 2:
                firstOrderEdgeVariation = np.std(firstOrderEdgeLength, ddof=1)
                firstOrderEdgeVariationList.append(firstOrderEdgeVariation)
            else:
                firstOrderEdgeVariation = np.std(firstOrderEdgeLength, ddof=0)
                firstOrderEdgeVariationList.append(firstOrderEdgeVariation)

        meanVariation = np.mean(firstOrderEdgeVariationList)
        subgraphMeanVariation.append(meanVariation)
    return subgraphMeanVariation


def getLocalCutValue(subgraphMeanVariation, subgraphSecondOrderEdgeMean):
    """
    计算局部约束准则.

    输入参数
    subgraphMeanVariation: 子图局部边长平均变异。
    subgraphSecondOrderEdgeMean: 各子图各顶点二阶邻域边长均值。

    输出参数
    subgraphLocalCutValueList: 局部边长约束准则
    """
    length, subgraphLocalCutValueList = len(subgraphMeanVariation), []
    for i in range(length):
        subgraphMV, subgraphSecondOrderMean = subgraphMeanVariation[i], subgraphSecondOrderEdgeMean[i]
        localCutValueList = []
        for e in subgraphSecondOrderMean:
            localCutValue = e[1] + 0.5 * subgraphMV  # todo 0.5？
            localCutValueList.append([e[0], localCutValue])
        subgraphLocalCutValueList.append(localCutValueList)
    return subgraphLocalCutValueList


def getLocalOtherEdge(edgeList, subgraphLocalCutValueList):
    """
    删除局部长边，获取全局其他边。

    输入参数
    edgeList: 删除全局长边后的edgeList。
    subgraphLocalCutValueList: 局部边长约束准则

    输出参数
    localOtherEdge：删除局部长边后的edgeList。
    """
    localOtherEdge = edgeList[:]
    for sg in subgraphLocalCutValueList:
        for pnt in sg:
            for e in localOtherEdge:
                if pnt[0] in e[1:3] and e[-1] >= pnt[1]:
                    if e in localOtherEdge:
                        localOtherEdge.remove(e)
                else:
                    continue
    return localOtherEdge


# --------------------------删除限制长边--------------------------

def deleteRestrictionEdge(edgeList, restritionNumber):
    """
    删除边长大于限定值的DT边。

    输入参数
    edgeList: 删除局部长边后的edgeList。
    restritionNumber: 边长限定值，数值型。

    输出参数
    edges: 删除限定长边后的edgeList。
    """
    edges = edgeList[:]
    for e in edges:
        if e[3] >= restritionNumber:
            edges.remove(e)
        else:
            continue
    return edges


# --------------------------删除不可达边--------------------------

# 以下函数用于空间叠置分析。基于向量旋转角的二维线段相交判定
# ......................................................................................................................
# This is a 2D line segment intersection decision algorithm, And refer to the following reference:
# https://blog.csdn.net/weixin_42736373/article/details/84587005
# ......................................................................................................................

class IntersectTest(object):
    def __init__(self, p1, p2, q1, q2):
        self.result = self.intersectTest(p1, p2, q1, q2)

    def coordiante(self, x1, x2, k):
        if x1[k] < x2[k]:
            return -1
        elif x1[k] == x2[k]:
            return 0
        else:
            return 1

    def intersectTest(self, p1, p2, q1, q2):
        p = self.subtraction(p2, p1)
        q = self.subtraction(q2, q1)
        denominator = self.crossProduct(p, q)
        t_molecule = self.crossProduct(self.subtraction(q1, p1), q)  # (q1 - p1) × q
        if denominator == 0:
            if t_molecule == 0:
                p_q = [p1, p2, q1, q2]
                if p1 != q1 and p1 != q2 and p2 != q1 and p2 != q2:
                    p_q = sorted(p_q, key=cmp_to_key
                    (functools.partial(self.coordiante, k=1 if (p2[0] - p1[0]) / (p2[1] - p1[1]) == 0 else 0)))
                    if p_q[0:2] == [p1, p2] or p_q[0:2] == [p2, p1] or p_q[0:2] == [q1, q2] or p_q[0:2] == [q2, q1]:
                        return 1
                    else:
                        return 1  # 相交
                else:
                    return 1  # 相交
            else:
                return 0  # parallel

        t = t_molecule / denominator
        if 0 <= t <= 1:
            u_molecule = self.crossProduct(self.subtraction(q1, p1), p)  # (q1 - p1) × p
            u = u_molecule / denominator
            if 0 <= u <= 1:  # 相交
                return 1
            else:
                return 0
        else:
            return 0

    def subtraction(self, a, b):
        c = []
        for i, j in zip(a, b):
            c.append(i-j)
        return c

    def crossProduct(self, a, b):
        return a[0]*b[1]-a[1]*b[0]


# ......................................................................................................................

def getReachableEdge(edgeList, obstacleList, pointList):
    """
    删除与障碍相交的边，返回余下DT边列表，在根据各点的一阶领域点再次做标记。

    输入参数
    edgeList: 删除限定长边后的edgeList。
    obstacleList: 障碍列表[[[Sx1, Sy1],[Ex1, Ey1]], ...]
    pointList:

    输出参数
    reachableEdge: 删除不可达边后的edgeList。
    """
    edgeL = [[pointList[e[1]], pointList[e[2]]] for e in edgeList]

    unreach, reachable, reachableEdge = [], [], []
    for i in obstacleList:
        for j in edgeL:
            intersect = IntersectTest(i[0], i[1], j[0][1:3], j[1][1:3]).result
            if intersect == 1 and j not in unreach:
                unreach.append(j)
            else:
                continue

    for e in edgeL:
        if e not in unreach:
            reachable.append(e)
        else:
            continue

    for p in reachable:
        indexA = p[0][0]
        indexB = p[1][0]
        for E in edgeList:
            if indexA in E[1:3] and indexB in E[1:3]:
                reachableEdge.append(E)
            else:
                continue
    return reachableEdge


# --------------------------ArcGIS界面的可视化与输出--------------------------

def createShapeFile(pointList, spatialRef, output):
    """
    根据坐标点列表创建point文件，并为其设定坐标参考。

    输入参数
    pointList: 多次聚类标记后的pointList。
    spatialRef: 空间参考
    output: 文件输出位置及名称
    """
    point = arcpy.Point()
    pointGeometryList = []
    for i in range(len(pointList)):
        point.X = pointList[i][1]
        point.Y = pointList[i][2]

        pointGeometry = arcpy.PointGeometry(point, spatialRef)
        pointGeometryList.append(pointGeometry)

    arcpy.CopyFeatures_management(pointGeometryList, output)
    return


def addMarkerFields(fileName, pointList):
    """
    给输出shape文件增加字段

    输入参数
    fileName: 需增加字段的文件名称及路径
    pointList: 多次聚类标记后的pointList。
    """
    arcpy.AddField_management(fileName, "ID_T", "FLOAT")
    arcpy.AddField_management(fileName, "markerO", "TEXT")  # obstacle
    arcpy.AddField_management(fileName, "markerG", "TEXT")  # global
    arcpy.AddField_management(fileName, "markerL", "TEXT")  # local
    arcpy.AddField_management(fileName, "markerC", "TEXT")  # Constraint

    counter, rows = 0, arcpy.UpdateCursor(fileName)
    for row in rows:
        row.setValue("ID_T", pointList[counter][0])
        row.setValue("markerO", pointList[counter][-4])
        row.setValue("markerG", pointList[counter][-3])
        row.setValue("markerL", pointList[counter][-2])
        row.setValue("markerC", pointList[counter][-1])
        rows.updateRow(row)
        counter += 1
    return


def outputWriteToTxt(filePath, name, inList, pointList):
    """
    This functions writes to a .txt file.

    Input Arguments
    outListStep_point: Path to folder where .txt file is stored
    name: Name of .txt file
    inList: List with entries to write to .txt file
    """
    outfile = filePath + name + ".txt"
    myDocument = open(outfile, 'w')
    myDocument.write("=========================================================================================" + "\n")
    myDocument.write("This file summarized the cluster result! " + "\n")
    myDocument.write("=========================================================================================" + "\n")
    myDocument.write("Please notice that 'O0, C0...' represents isolated points! " + "\n")
    myDocument.write("\n")

    myDocument.write("Numbet of points: " + str(inList[0]) + "\n")
    myDocument.write("NNI: " + str(inList[1]) + "\n")
    myDocument.write("Number of cluster: " + str(inList[2]-1) + "C" + "\n" + "\n")  # 不计散点
    myDocument.write("-----------------------------------------------------------------------------------------" + "\n")
    myDocument.write("Details of the clustering results" + "\n")
    myDocument.write("-----------------------------------------------------------------------------------------" + "\n")
    label = list(set([i[-1] for i in pointList]))
    listLabel = [i[-1] for i in pointList]
    for i in label:
        lst = []
        for j in listLabel:
            if j == i:
                lst.append(j)
            else:
                continue
        myDocument.write(i + ": " + str(len(lst)) + "\n")
    myDocument.close()
    return
