'''
Get examples of words with each sentiment.

Used for a table.
'''
import sentiment_nrc
import random

class AffectExamples():
    def __init__(self):
        sc = sentiment_nrc.SentimentCorrelator()
        sc.useStopwords = False
        self.sc = sc
        self.wordList = sc.parseNrc()
    
    def run(self):
        for i in range(10):
            self.runOneAffect(i)
        self.runNeutralWords()
        self.runContradictions()
    
    def runOneAffect(self, i):
        sentName = self.sc.sentimentIntToString(i)
        wordsWithAffect = [word for word in self.wordList if self.wordList[word][i]]
        wordsWithAffectOnly = self.getWordsWithAffectOnly(wordsWithAffect, i)
                  
        random.shuffle(wordsWithAffect)
        random.shuffle(wordsWithAffectOnly)
          
        print(sentName)
        print(', '.join(wordsWithAffect[:8]))
        print(', '.join(wordsWithAffectOnly[:8]))
        print()
    

    def runNeutralWords(self):
        allWords = [word for word in self.wordList]
        wordsWithAffectOnly = self.getWordsWithAffectOnly(allWords, -1)
        random.shuffle(wordsWithAffectOnly)
        print('neutral')
        print(', '.join(wordsWithAffectOnly[:8]))
        print()        


    def runContradictions(self):
        wordsWithNegative = set(word for word in self.wordList if self.wordList[word][5])
        wordsWithPositive = set(word for word in self.wordList if self.wordList[word][6])
        intersec = wordsWithNegative.intersection(wordsWithPositive)
        print(intersec)
        print(len(intersec))

        wordsWithJoy = set(word for word in self.wordList if self.wordList[word][4])
        wordsWithSadness = set(word for word in self.wordList if self.wordList[word][7])
        intersec = wordsWithJoy.intersection(wordsWithSadness)
        print(intersec)
        print(len(intersec))


    def getWordsWithAffectOnly(self, wordsWithAffect, i):
        wordsWithAffectOnly = []
        for word in wordsWithAffect:
            foundAnyOtherAffect = False
            for j in range(10):
                if j == i:
                    continue
                if self.wordList[word][j]:
                    foundAnyOtherAffect = True
                    break
            if not foundAnyOtherAffect:
                wordsWithAffectOnly.append(word)
        return wordsWithAffectOnly

if __name__ == "__main__":
    AffectExamples().run()
