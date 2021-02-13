'''
Makes heatmap visualizations from the results pickles.
'''
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from decimal import Decimal
from scipy import stats

from os import listdir
from os.path import isfile, join

import sentiment_nrc
from getRedBlueColor import RedBlue

# import the list of emotions and relevant functions from sentiment_nrc
emotions = sentiment_nrc.SentimentCorrelator.emotions
# The four base vocabularies
yAdditions = ["NRC\n(non-match)", "NRC\n(neutral)", "Stopwords\n(neutral)", "Compr.\n(neutral)"]

emotions_X = emotions[:-2]
emotions_Y = yAdditions + emotions[:-2]

sentimentStringToInt = sentiment_nrc.SentimentCorrelator().sentimentStringToInt

def sentimentStringToInt_X(string):
    return emotions_X.index(string)

def sentimentStringToInt_Y(string):
    return emotions_Y.index(string)

def translateY(i):
    return i+len(yAdditions)

def getPickle(filepath):
    with open(filepath, 'rb') as f:
        pickleObject = pickle.load(f)
    return pickleObject

def starPV(pValue):
    '''
    Return the number of stars of significance where < 0.01 = *, < 0.001 = **, etc.
    '''
    sign = float(translateStat(pValue))
    outVal = int(-math.log(abs(pValue), 10) - 1) if pValue != 0 else 5.0
    if outVal < 0:
        outVal = 0.0

    if outVal > 5:
        outVal = 5.0
    return sign * outVal

def matrixStarPV(matrix):
    '''
    Get number of stars for each cell.
    '''
    matrix2 = [[starPV(y) for y in x] for x in matrix]
    for i in range(len(emotions_X)):
        for j in range(len(emotions_X)):
            if i <= j:
                matrix2[i][translateY(j)] = 0
    return matrix2

def translateStat(stat):
    '''
    Only preserve the sign of t-statistics
    '''
    stat_new = 1 if stat >= 0 else -1
    return stat_new

def makeMatrix(po, po_stopwords, po_comprehensive):
    '''
    make the matrix of values (not colors)
    '''
    matrix = [[0 for _unused_y in range(len(emotions_Y))] for _unused_x in range(len(emotions_X))]
    statList = []

    # sentiments against non-matches, NRC neutral, neutral with stopwords,
    # and comprehensive lexicon
    for i in range(len(emotions_X)):
        foundObs = po['foundObservations'][i]
        notFoundObs = po['notFoundObservations'][i]
        stat, pValue = stats.ttest_ind(foundObs, notFoundObs, equal_var=False)
        matrix[i][0] = translateStat(stat) * pValue

        neutralObs = po['neutralValues']
        statN, pValueNeutral = stats.ttest_ind(foundObs, neutralObs, equal_var=False)
        matrix[i][1] = translateStat(statN) * pValueNeutral
        statList.append(statN)

        stopwordsObs = po_stopwords['neutralValues']
        statS, pValueStopwords = stats.ttest_ind(foundObs, stopwordsObs, equal_var=False)
        matrix[i][2] = translateStat(statS) * pValueStopwords

        comprehensiveObs = po_comprehensive['neutralValues']
        statC, pValueComp = stats.ttest_ind(foundObs, comprehensiveObs, equal_var=False)
        matrix[i][3] = translateStat(statC) * pValueComp

    # sentiments against each other
    for i in range(len(emotions_X)):
        for j in range(i):
            sent1Obs = po['foundObservations'][i]
            sent2Obs = po['foundObservations'][j]
            statCompare, pValue = stats.ttest_ind(sent1Obs, sent2Obs, equal_var=False)
            matrix[i][translateY(j)] = translateStat(statCompare) * pValue

    return matrix

def makeHeatMap(filepath, saveGraphs=True):
    stopwordsPath = "-stopwords-".join(filepath.split("-"))
    comprehensivePath = "-stopwords-comprehensive-".join(filepath.split("-"))

    po = getPickle(filepath)
    po_stopwords = getPickle(stopwordsPath)
    po_comprehensive = getPickle(comprehensivePath)

    matrix = np.array(makeMatrix(po, po_stopwords, po_comprehensive))
    matrix2 = np.array(matrixStarPV(matrix))

    # Plot it!
    fig, ax = plt.subplots()
    plt.pcolor(matrix2, cmap=RedBlue, alpha=0.8, vmin=-5, vmax=5)

    # the color bar scale on the right
    # plt.colorbar()

    # turn off the frame
    ax.set_frame_on(False)

    # put the ticks at the middle of each cell
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5, minor=False)

    # add labels
    ax.set_xticklabels(emotions_Y, minor=False)
    ax.set_yticklabels(emotions_X, minor=False)

    # put p-values as text on each cell
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if translateY(y) <= x:
                continue
            signedValue = matrix[y, x]
            value = abs(signedValue)
            color = "white" if value < 0.0001 else "black"
            displayValue = "+\n" if signedValue > 0 else "-\n"
            displayValue += '%.1E' % Decimal(value) if value < 0.001 else '%.4f' % value
            displayValue += "\n" + "*" * int(starPV(value))

            plt.text(x + 0.5,
                     y + 0.5,
                     displayValue,
                     horizontalalignment='center',
                     verticalalignment='center',
                     color=color,
                     size=9)

    # table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # add title
    title = filepath.split("/")[-1].split(".")[0]
    plt.title(title, y=1.08)

    # save plot
    if saveGraphs:
        plt.tight_layout()
        fig.set_size_inches(15, 9)
        plt.savefig('../results/visualizations/' + title + '.png',
                    dpi=300,
                    bbox_inches='tight')

    # plt.show()


def visualizeEveryPickle(saveGraphs=True):
    '''
    Visualize every correlator result pickle in the directory
    '''
    path = sentiment_nrc.SentimentCorrelator.getBaseDir('../results')
    pickleFiles = [join("../results", f) for f in listdir(path) if (
                        isfile(join(path, f))
                        and f[-2:] == ".p"
                        and len(f.split("-")) == 2)]  # only get the basic xxxx-results.p

    for p in pickleFiles:
        makeHeatMap(p, saveGraphs=saveGraphs)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # code to visualize one correlator result:
    '''
    testName = 'majorMinor'
    makeHeatMap("../results/" + testName + "-result.p")
    '''

    # Method to visualize every pickle
    visualizeEveryPickle(saveGraphs=True)
