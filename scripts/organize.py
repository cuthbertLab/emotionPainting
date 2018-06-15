from music21 import common, converter, text, meter
from os import path, listdir, mkdir
from shutil import copy
from functools import partial

from collections import Counter
#from langid import *

#inputPath = '/Users/Phi/Desktop/2cool4school/music21/new_in'
inputPath = '../wikifonia_en_chords_lyrics'
outputPath ='../new_out/enOrganized/lyrics_and_chords'


def update(numRun, totalRun, latestOutput):
    print("Run %s (%d/%d)" % (latestOutput, numRun, totalRun))

def organizeParallel(inputPath, outputPath, categorizeFunctionList):

    mxlfiles = [path.join(inputPath, f) for f in listdir(inputPath) if (path.isfile(path.join(inputPath, f)) and f[-3:] == "mxl")]
    newOrganizeOneSong = partial(organizeOneSong, outputPath=outputPath, categorizeFunctionList=categorizeFunctionList)
    unused_output = common.runParallel(mxlfiles, newOrganizeOneSong, updateFunction=update)
    print("Succecefully organized files from path \"" + inputPath+ "\".")


def organizeOneSong(filePath, outputPath, categorizeFunctionList):
    try:
        score = converter.parse(filePath)
    except Exception:
        return

    for fun in categorizeFunctionList:
        catagorized, folderName = fun(score)
        if catagorized:
            newDir = path.join(outputPath,folderName)
            if not path.isdir(newDir):
                mkdir(newDir)
            copy(filePath,newDir)


def hasLyrics(score):
    ly = text.assembleLyrics(score)
    if len(ly):
        return (True, "has_lyrics")
    else:
        return(True, "no_lyrics")

def hasChords(score):
    chords = []
    for pt in range(len(score.parts)):
        chords += score.parts[pt].recurse().getElementsByClass('Chord').stream()

    if len(chords):
        return (True, "has_chords")
    else:
        return(True, "no_chords")


def catagorizeByLanguage(score):
        ly = text.assembleLyrics(score)
        if not len(ly):
            return(True, "no_lyrics")
        lang = langid.classify(ly)

        if lang:
            return True,lang[0]
        else:
            return False,""




def xmlFileInfo(filePath):

    try:
        score = converter.parse(filePath)
    except Exception:
        return ["Could not parse" , filePath, "X", "X", "False"]

    #count chords
    chords = []
    for pt in range(len(score.parts)):
        chords += score.parts[pt].recurse().getElementsByClass('Chord').stream()

    #count lyrics
    ly = text.assembleAllLyrics(score)

    #test if syllabic matter
    syllabic = False
    justNotes = score.parts[0].recurse().getElementsByClass('Note').stream()
    for n in justNotes:
        lyricList = n.lyrics
        for l in lyricList:
            syl = l.syllabic
            if syl in ('begin', 'middle', 'end'):
                syllabic = True
                break

    # Filename, title, #chords, #lyrics, syllabic
    return [filePath.split("/")[-1], score.metadata.title, str(len(chords)), str(len(ly)), str(syllabic)]


import string

def printInfoParallel(inputPath):

    mxlfiles = [path.join(inputPath, f) for f in listdir(inputPath) if (path.isfile(path.join(inputPath, f)) and f[-3:] == "mxl")]
    toPrint = "Filename\tTitle\t# Chords\t#lyrics\tSyllabic\n"
    output = common.runParallel(mxlfiles, xmlFileInfo, updateFunction=update)
    output = sorted(output, key=lambda x: x[1])

    infoDict = {}
    needsReview = []
    delete=[]

    for out in output:
        key = " ".join([x.strip().translate(None, string.punctuation) for x in out[1].split()])
        if key in infoDict:
            diffchords = int(out[2]) - int(infoDict[key][2])
            difflyrics = int(out[3]) - int(infoDict[key][3])

            if diffchords >=0 and difflyrics>=0:
                if out[4] or out[4] == infoDict[key][4]:
                    delete.append(infoDict[key])
                    infoDict[key] = out
                else:
                    needsReview+=[infoDict[key],out]

            elif diffchords <=0 and difflyrics<=0:
                if out[4] and out[4] != infoDict[key][4]:
                    needsReview+=[infoDict[key],out]
                else:
                    delete.append(out)
            else:
                needsReview+=[infoDict[key],out]
        else:
            infoDict[key] = out


    for st in output:
        toPrint += "\t".join(st) + "\n"

    with open("Needs_Review.txt", "w") as res:
            res.write("\n".join(["\t".join(x) for x in needsReview]))


    with open("duplicatesToDelete.txt", "w") as res:
            res.write("\n".join(["\t".join(x) for x in delete]))


def checkMeters(filePath):
    try:
        score = converter.parse(filePath)
    except Exception:
        return "null"

    return score.recurse().getElementsByClass(meter.TimeSignature)[0].ratioString


from collections import Counter


def metersParallel(inputPath):

    mxlfiles = [path.join(inputPath, f) for f in listdir(inputPath) if (path.isfile(path.join(inputPath, f)) and f[-3:] == "mxl")]
    meters = common.runParallel(mxlfiles, checkMeters, updateFunction=update)
    c = Counter(meters)
    print(c.most_common())



if __name__ == '__main__':

    metersParallel(inputPath)




