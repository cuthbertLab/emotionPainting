import os

inputPath = '../wikifonia_en_chords_lyrics'
deletePath = '../wikifonia_duplicates_removed'

with open('NoSyllables.txt') as d2d:
    allFiles = d2d.readlines()

for f in allFiles:
    info = f.split('\t')
    fn = info[0]
    inP = inputPath+ os.sep + fn
    outP = deletePath + os.sep + fn
    # print(inP, outP)
    try:
        os.rename(inP, outP)
    except FileNotFoundError:  # may already have been passed
        print('skipping ', fn)
