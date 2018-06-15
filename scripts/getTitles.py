import os

from music21 import converter

import sentiment_nrc

def main():
    sc = sentiment_nrc.SentimentCorrelator()
    fileList = sc.getFileList()
    infoDict = {}
    for i, f in enumerate(fileList):
        c = converter.parse(f)
        fShort = int(f.split(os.sep)[-1][10:][:-4])
        title = c.metadata.title
        info = "{:10d} {:40s} {:40s}\n".format(fShort, title, c.metadata.composer)
        infoDict[title] = info
        if i % 10 == 0:
            print(i)
    
    with open('titles1895.txt', 'wb') as f:
        for t in sorted(list(infoDict)):
            f.write(infoDict[t].encode('utf-8'))

if __name__ == '__main__':
    main()