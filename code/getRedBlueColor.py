'''
Modify the pyplot.cm.Blues colormap to instead map from Red to White to Blue
from 0 to 1.
'''
from matplotlib.pyplot import cm
# from pprint import pprint as print

# noinspection PyUnresolvedReferences
RedBlue = cm.Blues  # @UndefinedVariable
# noinspection PyProtectedMember
colorMap = RedBlue._segmentdata

colorKeys = [0.0, 0.25, 0.5, 0.75, 1.0]

blueDict = {tup[0]: tup[1] for tup in colorMap['blue']}
greenDict = {tup[0]: tup[1] for tup in colorMap['green']}
redDict = {tup[0]: tup[1] for tup in colorMap['red']}

newBlue = []

for x in colorKeys[0:4]:
    blueColor = redDict[1-x]
    newBlue.append((x/2, blueColor, blueColor))

for x in colorKeys:
    blueColor = blueDict[x]
    newBlue.append((0.5 + x/2, blueColor, blueColor))


newRed = []

for x in colorKeys[0:4]:
    redColor = blueDict[1-x]
    newRed.append((x/2, redColor, redColor))

for x in colorKeys:
    redColor = redDict[x]
    newRed.append((0.5 + x/2, redColor, redColor))

newGreen = []

for x in colorKeys[0:4]:
    greenColor = greenDict[1-x]
    newGreen.append((x/2, greenColor, greenColor))

for x in colorKeys:
    greenColor = greenDict[x]
    newGreen.append((0.5 + x/2, greenColor, greenColor))

newSegmentData = {
    'alpha': colorMap['alpha'],
    'blue': newBlue,
    'green': newGreen,
    'red': newRed,
}

RedBlue._segmentdata = newSegmentData
