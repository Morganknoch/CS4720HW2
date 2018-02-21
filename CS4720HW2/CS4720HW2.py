# Morgan Knoch
# CS 4720 HW #2

import scipy.io
import numpy as np
import math
import copy

def MahalanobisDistance(covariance, mean, dimension, x):
    # This function takes a covariance, mean, dimension and sample

    # Probably will need to actually do the matrix inversion by hand
    cov = covariance
    invertedCovariance = np.identity(dimension)

    covariance = copy.deepcopy(covariance)

    # do row operations down to last row
    hold = dimension
    currentRow = 0
    currentColumn = 0
    for i in range(dimension):
        # see if first number is not one
        if covariance[i][i] != 1:
            # divide row by number
            holdLeadingNum = covariance[i][i]
            for j in range(dimension):
                invertedCovariance[i][j] /= holdLeadingNum
                covariance[i][j] /= holdLeadingNum

        currentRow += 1
        # subtract row from remaining rows
        for k in range(currentRow, dimension):
            # subtract first number times the subtracted original row
            holdNum = covariance[k][currentColumn]
            if holdNum != 0:
                for l in range(dimension):
                    invertedCovariance[k][l] = invertedCovariance[k][l] - (holdNum * invertedCovariance[i][l])
                    covariance[k][l] = covariance[k][l] - (holdNum * covariance[i][l])

        currentColumn += 1

    # go back up to first row
    currentColumn -= 1
    currentRow -= 1
    for i in range(dimension - 1, -1, -1):
        # see if first number is not one
        if covariance[i][i] != 1:
            # divide row by number
            holdLeadingNum = covariance[i][i]
            for j in range(dimension-1, -1, -1):
                invertedCovariance[i][j] /= holdLeadingNum
                covariance[i][j] /= holdLeadingNum

        currentRow -= 1
        # subtract row from remaining rows
        for k in range(currentRow, -1, -1):
            # subtract first number times the subtracted original row
            holdNum = covariance[k][currentColumn]
            if holdNum != 0:
                for l in range(dimension):
                    invertedCovariance[k][l] = invertedCovariance[k][l] - (holdNum * invertedCovariance[i][l])
                    covariance[k][l] = covariance[k][l] - (holdNum * covariance[i][l])

        currentColumn -= 1

    mean.shape = (3,1)
    
    num = np.matmul(np.matmul((x - mean).T, invertedCovariance), (x - mean))

    return num

def DiscriminantFunction(covariance, mean, dimension, priorProbability, x):

    return (-.5 * (MahalanobisDistance(covariance, mean, dimension, x))) - (dimension / 2)*(math.log((2*math.pi))) - (.5 * np.log(np.linalg.det(covariance))) + math.log(priorProbability)

# find mean
def findMean(data):
    mean = []
    for i in data:
        numinrow = 0
        totalrow = 0
        for j in i:
            numinrow += 1
            totalrow += j
        mean.append((totalrow / numinrow))
    return mean

def findCovariance(data, mean, dimension):
    dataFlipped = np.array(data)
    dataFlipped = dataFlipped.T

    meanarray = np.array(mean)
    
    numdata = 0

    covariance = np.zeros((dimension,dimension))
    
    for i in dataFlipped:
        newarray = np.array(i)
        
        ximinusmean = np.subtract(newarray, meanarray)
        ximinusmean = ximinusmean.reshape(dimension, 1)
        ximinusmeantranspose = ximinusmean.T    

        covariance += np.matmul(ximinusmean, ximinusmeantranspose)

        numdata += 1

    return covariance/numdata 

# NUMBER 1 part c

# Upload class data for Number 1 part c
data = scipy.io.loadmat("data_class3.mat")

newdata = data['Data']

firstClass = newdata[0][0]
secondClass = newdata[0][1]
thirdClass = newdata[0][2]

# find the mean for each class
firstClassMean = np.array(findMean(firstClass))
secondClassMean = np.array(findMean(secondClass))
thirdClassMean = np.array(findMean(thirdClass))

# find the covariance for each class
firstClassCov = findCovariance(firstClass, firstClassMean, 3)
secondClassCov = findCovariance(secondClass, secondClassMean, 3)
thirdClassCov = findCovariance(thirdClass, thirdClassMean, 3)

# declare prior probabilities
firstClassPriorProb = 0.6
secondClassPriorProb = 0.2
thirdClassPriorProb = 0.2

# declare x's to be classified
x1 = np.array([1,3,2])
x1.shape = (3,1)

x2 = np.array([4,6,1])
x2.shape = (3,1)

x3 = np.array([7,-1,0])
x3.shape = (3,1)

x4 = np.array([-2,6,5])
x4.shape = (3,1)


# classify points (use the discriminant function to classify) ###########################################

g1x1 = DiscriminantFunction(firstClassCov, firstClassMean, 3, firstClassPriorProb, x1)
g2x1 = DiscriminantFunction(secondClassCov, secondClassMean, 3, secondClassPriorProb, x1)
g3x1 = DiscriminantFunction(thirdClassCov, thirdClassMean, 3, thirdClassPriorProb, x1)

a = max(g1x1,g2x1, g3x1)

g1x2 = DiscriminantFunction(firstClassCov, firstClassMean, 3, firstClassPriorProb, x2)
g2x2 = DiscriminantFunction(secondClassCov, secondClassMean, 3, secondClassPriorProb, x2)
g3x2 = DiscriminantFunction(thirdClassCov, thirdClassMean, 3, thirdClassPriorProb, x2)

b = max(g1x2,g2x2, g3x2)

g1x3 = DiscriminantFunction(firstClassCov, firstClassMean, 3, firstClassPriorProb, x3)
g2x3 = DiscriminantFunction(secondClassCov, secondClassMean, 3, secondClassPriorProb, x3)
g3x3 = DiscriminantFunction(thirdClassCov, thirdClassMean, 3, thirdClassPriorProb, x3)

c = max(g1x3,g2x3, g3x3)

g1x4 = DiscriminantFunction(firstClassCov, firstClassMean, 3, firstClassPriorProb, x4)
g2x4 = DiscriminantFunction(secondClassCov, secondClassMean, 3, secondClassPriorProb, x4)
g3x4 = DiscriminantFunction(thirdClassCov, thirdClassMean, 3, thirdClassPriorProb, x4)

d = max(g1x4,g2x4, g3x4)


################## QUESTION 2 PART 1 ########################

# generate gaussian examples
mean1 = np.array([8,2])
mean1.shape = (2,)
mean2 = np.array([2,8])
mean2.shape = (2,)
cov1 = np.array([[4.1, 0], [0,2.8]])
cov2 = cov1

samplesClass1 = np.random.multivariate_normal(mean1, cov1, size=500)
samplesClass1 = samplesClass1.T

samplesClass2 = np.random.multivariate_normal(mean2, cov2, size=500)
samplesClass2 = samplesClass2.T

# derive decision boundary




# plot decision boundary





