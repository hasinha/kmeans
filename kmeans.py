import pandas as pd
import random as rd
import math
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Initialize centroids. Assigned to points randomly
def initialise_centroids(dataFrame, k, pointCount):
    centroids = list()
    i = 0
    while i < k:
        nextCent = rd.randint(0, pointCount)
        centroids.append(dataFrame.iloc[nextCent - 1])
        i += 1
    return centroids

# Compute Euclidean Distance between specified points
def compute_euclidean_distance(vec1, vec2):
    squaredSum = 0
    for columnName in columnNames:
        squaredSum += (vec1[columnName] - vec2[columnName])**2
    return math.sqrt(squaredSum)

# Perform K-Means clustering using given data set and k clusters
def kmeans(data, k, maxEpochs):
    pointCount = data.shape[0]
    centres = initialise_centroids(data, k, pointCount)
    assignedCentroids = list()
    for index, row in data.iterrows():
        distDict = dict()
        for i in range(len(centres)):
            dist = compute_euclidean_distance(centres[i], row)
            distDict[dist] = i
        assignedCentroids.append(distDict[min(distDict.keys())])
    data['Centroid'] = assignedCentroids
    epochs = 0
    prevCountDict = {}
    while True:
        for i in range(len(centres)):
            members = data.loc[data['Centroid'] == i]
            memberCount = len(members)
            prevCountDict[i] = memberCount
        generateCentroids(centres, i)
        assignNewClusters(centres, data)
        isSameCount = True
        for key, value in prevCountDict.items():
            members = data.loc[data['Centroid'] == key]
            memberCount = len(members)
            if memberCount != value:
                isSameCount = False
                break
        print(prevCountDict)
        print(epochs)
        if isSameCount or epochs == maxEpochs:
            break
        epochs += 1
    print("Final Centres: ", centres)
    members = data.loc[data['Centroid'] == key]
    # generatePlot(data, centres)
    return centres

# Plot clusters generated
def generatePlot(data, centres):
    plt.figure(1, figsize=(20,10))
    colors = {0:'k',1:'b',2:'g'}
    center_color = ['r']
    for i in range(len(centres)):
        members = data.loc[data['Centroid'] == i]
        plt.scatter(members.loc[:,columnNames[2]].values,members.loc[:,columnNames[3]].values,alpha=0.5)
    plt.show()

# Performs assignment step
def assignNewClusters(centres, data):
    assignedCentroids = list()
    for index, row in data.iterrows():
        distDict = dict()
        for i in range(len(centres)):
            dist = compute_euclidean_distance(centres[i], row)
            distDict[dist] = i
        assignedCentroids.append(distDict[min(distDict.keys())])
    data['Centroid'] = assignedCentroids

# Update Centroids step
def generateCentroids(centres, i):
    for i in range(len(centres)):
        featureDict = defaultdict(int)
        members = dataFrame.loc[dataFrame['Centroid'] == i]
        memberCount = len(members)
        for index, row in members.iterrows():
            for columnName in columnNames:
                featureDict[columnName] += row[columnName]
        centroidDict = {}
        for key, val in featureDict.items():
            centroidDict[key] = val / memberCount
        centres[i] = centroidDict

# Used to plot iteration v objective function(Squared Distance)
def generateSquaredDistSum(data, centres):
    squaredDistSum = 0
    for i in range(len(centres)):
        members = data.loc[data['Centroid'] == i]
        for index, row in members.iterrows():
            dist = compute_euclidean_distance(centres[i], row)
            squaredDistSum += dist**2
    return squaredDistSum

# Generate Silhouette Coefficient
def generateCoefficient(data, centres):
    silhouetteDict = {}
    for i in range(len(centres)):
        silhouetteList = list()
        members = data.loc[data['Centroid'] == i]
        nonmembers = data.loc[data['Centroid'] != i]
        for index, curRow in members.iterrows():
            avalue = 0
            bValueDict = defaultdict(list)
            bFinVal = {}
            for index2, row in members.iterrows():
                if index == index2:
                    continue
                avalue += compute_euclidean_distance(curRow, row)
            avalue = avalue / len(members)
            for index3, nonMemRow in nonmembers.iterrows():
                dist = compute_euclidean_distance(curRow, nonMemRow)
                bValueDict[nonMemRow['Centroid']].append(dist)
            for key, value in bValueDict.items():
                bFinVal[np.sum(value) / len(value)] = key
            bValue = min(bFinVal.keys())
            if avalue < bValue:
                silhouetteList.append(1 - (avalue/bValue))
            else:
                silhouetteList.append((bValue/avalue) - 1)
        silhouetteDict[i] = np.sum(silhouetteList) / len(silhouetteList)
    print(silhouetteDict)
    print(np.sum(list(silhouetteDict.values())) / len(silhouetteDict.values()))

dataFrame = pd.read_csv('/path/to/dataset.csv')
columnNames = dataFrame.columns
centres = kmeans(dataFrame, 4, 50)
generateCoefficient(dataFrame, centres)
i = 1
squaredDistList = list()
while i < 11:
    centres = kmeans(dataFrame, i, 50)
    squaredDistList.append(generateSquaredDistSum(dataFrame, centres))
    i += 1
print(squaredDistList)
plt.figure()
plt.plot([1,2,3,4,5,6,7,8,9,10], squaredDistList, 'o-')
plt.show()