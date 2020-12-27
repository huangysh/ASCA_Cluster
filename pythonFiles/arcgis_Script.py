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

# general import
import gc
import os
import sys

from ASCA_Functions import *

pythonScriptPath = r"...\RuST\ASCA_Cluster\pythonFiles"

gc.disable()
pythonPath = os.getcwd()
sys.path.append(pythonPath)
sys.path.append(pythonScriptPath)

# ======================================================================================================================
# 调用ArcGIS界面输入
# ======================================================================================================================
arcpy.env.overwriteOutput = True
buildings = sys.argv[1]  # Building shape file
studyArea = sys.argv[2]
obstacleFile = sys.argv[3]  # optical parameter.
num = int(sys.argv[4])  # 管道长度约束[米] optical parameter.
outputFolder = sys.argv[5]

# folder setting
outputFile = outputFolder + "/" + "Cluster.shp"
addFiledFile = outputFile  # sys.argv[5] + ".shp"

# ----------------------------------------------------------------------------------------------------------------------
# 空间点分布模式判定
# ----------------------------------------------------------------------------------------------------------------------
pointList, spatialRef = readSpatialPoint(buildings)  # 读取空间点及输入文件的

distanceList = getNearestDistance(pointList)
area = float(readArea(studyArea))  # 读取研究区域面积
index, _ = NNI(pointList, distanceList, area)
triangleVertexIndex, triangleVertexCoordinate = getDelaunayTriangle(pointList)  # 核实未使用参数

# 输出空间点集分布趋势
arcpy.AddMessage("\n")
arcpy.AddMessage("************************************")
arcpy.AddMessage("Points spatial cluster analysis was successfully calculated!!")
arcpy.AddMessage("NNI index : " + str(index))
arcpy.AddMessage("************************************")

# 开始空间点集聚类分析
arcpy.AddMessage("\n")
arcpy.AddMessage("====================================")
arcpy.AddMessage("Ready for cluster module...")
arcpy.AddMessage("====================================")
arcpy.AddMessage("\n")

_, edgeList = getEdgeLength(triangleVertexIndex, triangleVertexCoordinate)

if index >= 1:  # 空间点集呈均匀(>1)/随机分布(=1)
    arcpy.AddMessage("Random distribution OR Uniform distribution (NNI >= 1)")
    arcpy.AddMessage("Skip cluster analysis module!!!" + "\n" +
                     "Perform Obstacle and Restriction analysis!!!")

    # obstacle
    if len(obstacleFile) > 1:
        obstacleList = readObstacle(obstacleFile)
        reachableEdge = getReachableEdge(edgeList, obstacleList, pointList)
        indexList_O = aggregation(reachableEdge)
        mark_O = "O"
        cluster(pointList, indexList_O, mark_O)
        arcpy.AddMessage("Unreachable edges were deleted!!!")
    else:
        reachableEdge = edgeList[:]
        pointList = [i + ["O0"] for i in pointList]
        arcpy.AddMessage("No obstacles!!!")

    # restrict
    if num > 0:
        unrestrictedEdge = deleteRestrictionEdge(reachableEdge, num)
        indexList_C = aggregation(unrestrictedEdge)
        mark_C = "C"
        cluster(pointList, indexList_C, mark_C)
        arcpy.AddMessage("Restricted edges were deleted!!!")
    else:
        unrestrictedEdge = reachableEdge[:]
        pointList = [i + i[-1] for i in pointList]
        arcpy.AddMessage("No Length restriction!!!")

    createShapeFile(pointList, spatialRef, outputFile)
    addMarkerFields(addFiledFile, pointList)

    arcpy.AddMessage("-----" + "Spatial Cluster Model successfully performed!" + "-----")


elif index < 1:  # 空间点集呈聚集分布
    arcpy.AddMessage("Spatial points is aggregated, perform cluster analysis Module!!!")

    # obstacle
    if len(obstacleFile) > 1:
        obstacleList = readObstacle(obstacleFile)
        reachableEdge = getReachableEdge(edgeList, obstacleList, pointList)
        indexList_O = aggregation(reachableEdge)
        mark_O = "O"
        cluster(pointList, indexList_O, mark_O)  # return marked pointList
        arcpy.AddMessage("Unreachable edges were deleted!!!" + "\n")
    else:
        reachableEdge = edgeList[:]
        pointList = [i + ["O0"] for i in pointList]
        arcpy.AddMessage("No obstacles!!!" + "\n")

    # global long edge   todo check
    globalEdgeMean, globalEdgeVariation = getGlobalEdgeStatistic(reachableEdge)
    firstOrderEdges, _ = getFirstOrderEdges(pointList, reachableEdge)
    firstOrderEdgesMean = getFirstOrderEdgesMean(firstOrderEdges)
    globalCutValueList = getGlobalCutValue(globalEdgeMean, globalEdgeVariation, firstOrderEdgesMean)
    globalOtherEdgeList = getGlobalOtherEdge(reachableEdge, globalCutValueList)
    indexListG = aggregation(globalOtherEdgeList)
    markG = "G"
    cluster(pointList, indexListG, markG)
    arcpy.AddMessage("Global long edges were deleted !!!" + "\n")

    # local long edge
    subgraphVertexList, subgraphEdgeList = getSubgraphEdge(pointList, globalOtherEdgeList, indexListG)

    subgraphSecondOrderEdgeMean = getSecondOrderEdges(subgraphVertexList, subgraphEdgeList)

    subgraphMeanVariation = getSubgraphEdgeStatistic(subgraphVertexList, subgraphEdgeList)

    subgraphLocalCutValueList = getLocalCutValue(subgraphMeanVariation, subgraphSecondOrderEdgeMean)

    localOtherEdge = getLocalOtherEdge(globalOtherEdgeList, subgraphLocalCutValueList)
    indexListL = aggregation(localOtherEdge)
    markL = "L"
    cluster(pointList, indexListL, markL)
    arcpy.AddMessage("Local long edges were deleted !!!" + "\n")

    # restrict
    if num > 0:
        unrestrictedEdge = deleteRestrictionEdge(localOtherEdge, num)
        indexList_C = aggregation(unrestrictedEdge)
        mark_C = "C"
        cluster(pointList, indexList_C, mark_C)
        arcpy.AddMessage("Restricted edges were deleted!!!" + "\n")
    else:
        unrestrictedEdge = localOtherEdge[:]
        pointList = [i + i[-1] for i in pointList]
        arcpy.AddMessage("No Length restriction!!!" + "\n")
    arcpy.AddMessage("pointList:")
    arcpy.AddMessage(str(pointList))

    createShapeFile(pointList, spatialRef, outputFile)
    addMarkerFields(addFiledFile, pointList)

    D = list(set([i[7] for i in pointList]))  # local long edge
    LIST = [len(pointList), index, len(D)]

    output = outputFolder + "/" + "Cluster"
    name = "Output"
    outputWriteToTxt(output, name, LIST, pointList)

    arcpy.AddMessage("-----" + "Spatial Cluster Model successfully performed!" + "-----")
    arcpy.AddMessage("\n")
