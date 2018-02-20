# Morgan Knoch
# CS 4720 HW #2

import scipy.io
import numpy as np
import math

def MahalanobisDistance(covariance, mean, dimension, x):
    # This function takes a covariance, mean, dimension and sample

    # Probably will need to actually do the matrix inversion by hand

    invertedCovariance = np.identity(dimension)

    # do row operations down to last row
    hold = dimension
    currentRow = 0
    currentColumn = 0
    for i in range(dimension):
        # see if first number is not one
        if covariance[i][i] != 1:
            # divide row by number
            for j in range(dimension):
                invertedCovariance[i][j] /= covariance[i][i]

        currentRow += 1
        # subtract row from remaining rows
        for k in range(currentRow, dimension):
            # subtract first number times the subtracted original row
            if covariance[currentRow][currentColumn] != 0:
                for l in range(dimension):
                    invertedCovariance[k][l] = invertedCovariance[k][l] - (covariance[k][currentColumn] * covariance[i][l])

        currentColumn += 1

    # go back up to first row
    for i in range(dimension, 0, -1):
        # see if first number is not one
        if covariance[i][i] != 1:
            # divide row by number
            for j in range(dimension, 0, -1):
                invertedCovariance[i][j] /= covariance[i][i]

        currentRow -= 1
        # subtract row from remaining rows
        for k in range(currentRow, 0, -1):
            # subtract first number times the subtracted original row
            if covariance[currentRow][currentColumn] != 0:
                for l in range(dimension):
                    invertedCovariance[k][l] = invertedCovariance[k][l] - (covariance[k][currentColumn] * covariance[i][l])

        currentColumn -= 1

    return np.matmul(np.matmul((x - mean).T, invertedCovariance, (x - mean)))

def DiscriminantFunction(covariance, mean, dimension, priorProbability, x):

    return (-.5 * (MahalanobisDistance(covariance, mean, dimension, x))) - (dimension / 2)*(math.log((2*math.pi))) - (.5 * np.log(np.absolute(covariance))) + math.log(priorProbability)






# calculate mean and covariance for every class




# classify points







# generate gaussian examples



# derive decision boundary




# plot decision boundary





