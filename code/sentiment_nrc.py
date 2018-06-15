'''
Main sentiment_nrc correlation routine.  Requires Python 3 to run.
'''
import sys
if sys.version_info[0] < 3:
    raise Exception("This library requires Python 3 to run")


# from functools import partial
import inspect
import itertools
import os
from statistics import median

from numpy import mean, std
from os import listdir
from os.path import isfile, join, dirname, sep, exists
# import re
from scipy import stats
import string
import porter2
import pickle

from music21 import converter, common, chord # , search
from music21.features import native
from music21.ext import six


class BaseOneWork(object):
    '''
    BaseOneWork is the abstract base class for running a single work
    through an analysis system that matches lyrics with sentiments
    and performs analytical observations at that point.
    
    The object must be lightweight and pickleable.  Hence, unless
    saveScore is set to True, the parsed score is not saved in the
    object.  Making it lightweight allows it to be passed into and
    out of the runParallel routine quickly, taking advantage of
    multiple processors/cores.
    '''
    numberOfEmotions = 10
    
    def __init__(self, nrcParsed=None, fileName=None, saveScore=False):
        self.stemWords = False
        self.nrcParsed = nrcParsed
        self.fileName = fileName
        
        self.totalLyrics = 0
        self.lyricsWithSentiment = [0] * self.numberOfEmotions
        # for Unpaired t-test, we need to have two groups of data: expriment and control 
        self.foundObservations = [[] for _ in range(self.numberOfEmotions)]
        self.notFoundObservations = [[] for _ in range(self.numberOfEmotions)]
         
        self.neutralCount = 0
        self.neutralList = []
        
        self.notInDatabaseCount = 0
        self.notInDatabaseList = []
        
        self.saveScore = saveScore  # fix up the score and save it...
        self.savedScore = None

        self.operativeNote = None 

        self.storedJoyObservations = None  # vs. sadness
        self.storedNegativeObservations = None # vs. positive
        self.comprehensiveVocabulary = False
        

    def run(self):
        '''
        Parse the score, runs runTest (which should be overridden by
        subclasses) and saves the score if .saveScore is True
        '''
        score = converter.parse(self.fileName)
        self.setupTest(score)
        self.runTest(score)
        if self.saveScore:
            self.savedScore = score # this will REALLY slow down a parallel processing routine.
        self.operativeNote = None # clear any references to score elements

    def setupTest(self, score):
        '''
        Set any preliminary information needed to run the test itself
        '''
        pass

    def normalizeLyric(self, lyric):
        '''
        Routine to normalize lyrics by removing 
        
        >>> BaseOneWork().normalizeLyric('Hello!')
        'hello'
        
        '''
        if not isinstance(lyric, six.string_types): # str or unicode
            return lyric

        lyric = lyric.lower()
        if six.PY2:
            lyric = lyric.translate(None, string.punctuation) #rid string of punctuations
        else:
            translator = str.maketrans('', '', string.punctuation)                    
            lyric = lyric.translate(translator)

        if self.stemWords:
            lyric = porter2.stem(lyric)
        return lyric

    def fixupScore(self, lyric, additionalData=''):
        '''
        Add information about matching sentiments to the score.
        
        Does nothing unless self.saveScore is True
        '''
        if not self.saveScore:
            return

        if lyric not in self.nrcParsed:
            return

        n = self.operativeNote
        if n is None:
            return

        emotions = ["anger", "anticipation", "disgust", "fear", "joy", "negative", 
                    "positive", "sadness", "surprise", "trust", "neutral", "unknown"]

        sentimentLyric = ''        
        for i in range(len(self.nrcParsed[lyric])):
            if self.nrcParsed[lyric][i] == 1:
                sentimentNameAbbreviated = emotions[i][0:3]
                sentimentLyric += ':' + sentimentNameAbbreviated

        n.addLyric(str(additionalData) + '.' + sentimentLyric, applyRaw=True)
        
    def runTest(self, score):
        justNotes = score.parts[0].recurse().getElementsByClass('Note')
        tempStrings = {}
    
        self.operativeNote = None
    
        for n in justNotes:
            lyricList = n.lyrics
    
    
            for i in range(len(lyricList)):    
                #lyric = lyricList[i].text.lower().translate(None, string.punctuation) 
                #rid string of punctuations
                lyric = lyricList[i].text
                lyric = self.normalizeLyric(lyric)
    
                syllabic = lyricList[i].syllabic
    
                #handles case where sylables are seperated
                if syllabic == "begin":
                    tempStrings[i] = lyric
                    self.operativeNote = n
                    continue
                elif syllabic == "middle":
                    if i in tempStrings.keys(): 
                        tempStrings[i] += lyric
                    else:
                        tempStrings[i] = lyric
                    continue
                elif syllabic == "end":
                    lyric = tempStrings[i] + lyric if i in tempStrings.keys() else lyric
                else:
                    self.operativeNote = n
    
                observation = self.getObservation(lyric)
                if observation is None:
                    continue

                # done merging lyrics into words -- let's start observing...
                self.totalLyrics += 1
            
                if lyric not in self.nrcParsed:
                    self.notInDatabaseCount += 1
                    self.notInDatabaseList += [observation]  
                    # print(lyric)
                    if not self.comprehensiveVocabulary:
                        continue
                    else: 
                        # we are using all vocabulary words
                        # and this one is not in the vocabulary.
                        neutralWord = True
                        for sentimentIndex in range(self.numberOfEmotions):
                            self.notFoundObservations[sentimentIndex] += [observation]
    
                else:
                    neutralWord = True
                   
                    for sentimentIndex, val in enumerate(self.nrcParsed[lyric]): 
                        if val == 1:
                            neutralWord = False
                            self.lyricsWithSentiment[sentimentIndex] += 1
                            self.foundObservations[sentimentIndex] += [observation]
                        else:
                            self.notFoundObservations[sentimentIndex] += [observation]
    
    
                if neutralWord:
                    self.neutralCount += 1
                    self.neutralList += [observation]
                    
                self.fixupScore(lyric, str(round(observation, 2)))    
                tempStrings[i] = ""


    def getObservation(self, lyricText):
        '''
        return a value for measuring the observation on this lyric.
        
        Use self.operativeNote to get the most important note for this lyric
        (currently, the first note associated with the lyric, or the only
        note if the lyric is a singular lyric)
        
        Return None if the observation should be skipped
        
        Override this in tests.  This one returns a bogus calculation
        of the pitchClass of the note -- to
        test that it should be a meaningless p-value in this case
        '''
        if self.operativeNote is None:
            return None
        return self.operativeNote.pitch.pitchClass
    
class SentimentCorrelator(object):
    '''
    abstract base class that returns information about a 
    collection of leadsheets (or other pieces) and
    correlations with sentiments
    '''
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "negative", 
                "positive", "sadness", "surprise", "trust", "neutral", "unknown"]

    # change this in subclasses to give a name to print out
    testName = 'example'
    
    # change this in subclasses to set an object to parse each piece of data.
    oneWorkClass = BaseOneWork

    def __init__(self, endNumber=None):
        self.usePickle = True
        self.startNumber = None
        self.endNumber = endNumber
        self.fileList = None # override default files
        self.nrcPath = self.getBaseDir('NRC-v0.92.txt')
        self.nrcParsed = None
        
        self.stemWords = False
        self.useStopwords = True
        self.comprehensiveVocabulary = False
        
        self.stopwordPath = self.getBaseDir('stopwords.txt')
        self.totalFiles = 0
        
    @staticmethod
    def getBaseDir(innerFile):
        '''
        Returns the base directory of the sentiment project, holding scripts, writing, etc.
        '''
        return common.pathTools.cleanpath(dirname(inspect.getfile(
            SentimentCorrelator.getBaseDir)) + sep + innerFile)

    def getFileList(self):
        '''
        returns the default file list, given self.startNumber and self.endNumber
        '''
        path = self.getBaseDir('../wikifonia_en_chords_lyrics')
        mxlFiles = [join(path, f) for f in listdir(path) if (
                        isfile(join(path, f)) and f[-3:] == "mxl")]        
        sliceObj = slice(self.startNumber, self.endNumber)
        return mxlFiles[sliceObj]
        
    @staticmethod
    def update(numRun, totalRun, latestOutput):
        '''
        Print out information about how far we are in the running; 
        called by music21.common.runParallel
        '''
        print("Run (%d/%d)" % (numRun, totalRun))

    def parseNrc(self):
        '''
        Takes in a NRC text file and parse it into a dictionary mapping each word to a list of ints, 
        indicating the score for each sentiment. The field of the list is as below
        0       1             2       3      4    5         6         7        8         9
        [anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust]    
        '''
        if not os.path.exists(self.nrcPath):
            raise FileNotFoundError(
                'Download the NRC Word association Lexicon and store it in this directory \n' +
                'as NRC-v0.92.txt.  http://saifmohammad.com/WebPages/AccessResource.htm ')
        
        with open(self.nrcPath, 'r') as nrc:
            wordToEmotions = {}
            for line in nrc:
                entry = line.split("\t")  # word - sentiment - 
                if len(entry) != 3: # preamble
                    continue
                else:
                    word, sentimentName, sentimentValue = entry
                    if self.stemWords:
#                       wordBefore = word
                        word = porter2.stem(word)
#                         if word != wordBefore:
#                             print(wordBefore + " stemmed to " + word)
                    # when we encounter "anger" it's a new word...
                    if sentimentName == 'anger': # word not in wordToEmotions:
                        wordToEmotions[word] = [] 
                    
                    sentimentValue = int(sentimentValue) # 0 or 1
                    # we do not use sentimentName, counting on the fact that each
                    # word appears 10 times in order as anger, anticipation, disgust, etc.
                    wordToEmotions[word].append(sentimentValue)

        if self.useStopwords:
            self.parseStopwords(wordToEmotions)

        return wordToEmotions

    def parseStopwords(self, wordToEmotions):
        '''
        Add a list of stopwords as neutral sentiments here.
        
        Uses a list from http://www.ranks.nl/stopwords with negative words ("couldn't") removed
        and also words related to height (above, below, under)
        '''
        with open(self.stopwordPath, 'r') as stopWords:
            for sw in stopWords:
                sw = sw.strip()
                if sw in wordToEmotions:
                    continue
                wordToEmotions[sw] = [0 for _ in range(10)]
        return wordToEmotions


    def sentimentStringToInt(self, sentString):
        '''
        given a sentiment string, return the index of it in this list...
        
         0       1             2       3      4    5         6         7        8         9
        [anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust]
        '''    
        return self.emotions.index(sentString)

    def sentimentIntToString(self, sentInt):
        '''
        given a sentiment position, return its name.
        '''
        return self.emotions[sentInt]

    def clearValues(self):
        '''
        Clear all counting values (generally used just for setting up the counting 
        attributes the first time.
        '''
        self.totalFiles = 0
        self.totalLyrics = 0
        self.lyricsWithSentiment = [0] * 10 # number of occurences of each sentiment
        # for Unpaired t-test, we need to have two groups of data: expriment and control 
        self.foundObservations = [[] for _ in range(10)]
        self.notFoundObservations = [[] for _ in range(10)] 
        # positive and negative refer to whether a sentiment was observed or not
        # it has nothing to do with the "postiive" and "negative" sentiment types
        
        self.neutralValues = []
        self.neutralCount = 0
        self.notInDatabaseValues = []
        self.notInDatabaseCount = 0

    def calculateValues(self, listOfObservations):
        '''
        Given a list of BaseOneWork objects (or their derived classes),
        combine their observations into this object's observation counts.
        
        For instance, self.totalLyrics will be the sum of each object's 
        `.totalLyrics` attributes.  Because many objects are lists of 0s and 1s or
        lists of, say, pitch height numbers, we use combineLists to merge these.
        '''
        def combineList(list1, list2):
            outList = list(map(lambda x: x[0] + x[1], zip(list1, list2)))
            return outList
    
        for oneSong in listOfObservations:
            # oneSong is a BaseOneWork object
            self.totalFiles += 1
            self.totalLyrics += oneSong.totalLyrics
            self.lyricsWithSentiment = combineList(self.lyricsWithSentiment, 
                                                   oneSong.lyricsWithSentiment)
            self.foundObservations = combineList(self.foundObservations, 
                                                 oneSong.foundObservations)
            self.notFoundObservations = combineList(self.notFoundObservations, 
                                                    oneSong.notFoundObservations)
            self.neutralCount += oneSong.neutralCount
            self.neutralValues.extend(oneSong.neutralList)
            self.notInDatabaseCount += oneSong.notInDatabaseCount
            self.notInDatabaseValues.extend(oneSong.notInDatabaseList)

    def getResultsHeader(self):
        '''
        make a nice header for printing out.
        '''
        toPrint = ""
    
        toPrint += "Printing sentiment-" + self.testName + " results: \n"
        toPrint += "Words in NRC File = " +  str(len(self.nrcParsed)) + "\n" 
        toPrint += "Number of files = " + str(self.totalFiles) + "\n"
        toPrint += "Total lyrics count = " + str(self.totalLyrics) + "\n"
        toPrint += "Neutral, sentiment-free words: " + str(self.neutralCount) + "\n"
        toPrint += "Words occurrences not in our database: "+ str(self.notInDatabaseCount) + "\n"
        toPrint += "Mean value of neutral observations: " + str(mean(self.neutralValues)) + "\n"
        return toPrint

    def runAll(self):
        '''
        Main routine to run a test -- call after
        everything has been set.
        '''
        if self.fileList is None:
            self.fileList = self.getFileList()
        if self.nrcParsed is None:
            self.nrcParsed = self.parseNrc()
    
        self.clearValues()
        if self.usePickle and exists(self.resultsFilename() + '.p'):
                self.readPickleValues()
        else:
            output = common.runParallel(self.fileList, self.oneSong, self.update,
                                        updateSendsIterable=True)
            self.calculateValues(output)
            if self.usePickle:
                self.pickleValues()
        
        toPrint = self.getResultsHeader()
        
        for sentimentNumber in range(10):
            thisFound = self.foundObservations[sentimentNumber]
            thisNotFound = self.notFoundObservations[sentimentNumber]
            toPrint += "\nSentiment: " + self.sentimentIntToString(sentimentNumber) + "\t" 
            toPrint += "Occurrence: " + str(self.lyricsWithSentiment[sentimentNumber]) + "\n"
            #print(deviations[sentimentInt])
            if not thisFound:
                continue # no observations
            # toPrint += "\n" + str(thisDeviation)
            
            toPrint += self.classSpecificResults(thisFound, thisNotFound, sentimentNumber)

        toPrint += "\n"

        for sent1, sent2 in itertools.product(range(10), repeat=2):
            if sent2 <= sent1:
                continue

            sent1Obs = self.foundObservations[sent1]
            sent2Obs = self.foundObservations[sent2]
                        
            _, pValue = stats.ttest_ind(sent1Obs, 
                                        sent2Obs, 
                                        equal_var=False)
            toPrint += "*** p-value of        {:10s} vs. {:10s} is : {:7.5f} \n".format(
                self.sentimentIntToString(sent1),
                self.sentimentIntToString(sent2),
                pValue)
            toPrint += "*** diff of median of {:10s}  -  {:10s} is : {} \n\n".format(
                self.sentimentIntToString(sent1),
                self.sentimentIntToString(sent2),
                median(sent1Obs) - median(sent2Obs))

        print(toPrint)
        self.writeFile(toPrint)

    def oneSong(self, fileName):
        '''
        This method is run separately in different Cores/processors once for each
        work being analyzed.  It
        creates a new BaseOneWork object, calls its run method and then passes
        the workObject back across the processor core to the main process.
        
        That part is the bottleneck for obtaining extreme multicore speeds on
        say a 24-core system, so
        make sure that the BaseOneWork derived class is as light as possible
        '''
        workObject = self.oneWorkClass(self.nrcParsed, fileName)
        workObject.stemWords = self.stemWords
        workObject.comprehensiveVocabulary = self.comprehensiveVocabulary
        workObject.run()
        return workObject

    def writeFile(self, toPrint):
        '''
        Optionally, write out these results into a file for saving.  
        Filename comes from the test name.
        '''
        fn = self.resultsFilename() + '.txt'
        with open(fn, "w") as res:
            res.write(toPrint)
    
    def resultsFilename(self):
        tn = self.testName
        if self.endNumber is not None:
            tn += "-" + str(self.endNumber)
        if self.useStopwords:
            tn += "-stopwords"
        if self.stemWords:
            tn += "-stemmed"
        if self.comprehensiveVocabulary:
            tn += '-comprehensive'
        
        fn = ".." + sep + "results" + sep + tn + "-result"
        return self.getBaseDir(fn)

    def pickleValues(self):
        fn = self.resultsFilename() + '.p'
        pickleObject = {}
        for k in ('totalFiles', 'totalLyrics', 'lyricsWithSentiment', 'foundObservations',
                  'notFoundObservations', 'neutralCount', 'neutralValues',
                  'notInDatabaseCount', 'notInDatabaseValues'):
            pickleObject[k] = getattr(self, k)
        with open(fn, 'wb') as f:
            pickle.dump(pickleObject, f)

    def readPickleValues(self):
        fn = self.resultsFilename() + '.p'
        with open(fn, 'rb') as f:
            pickleObject = pickle.load(f)
        for k in ('totalFiles', 'totalLyrics', 'lyricsWithSentiment', 'foundObservations',
                  'notFoundObservations', 'neutralCount', 'neutralValues',
                  'notInDatabaseCount', 'notInDatabaseValues'):
            setattr(self, k, pickleObject[k])

    def classSpecificResults(self, thisFound, thisNotFound, sentimentNumber):
        '''
        This method should be overwritten to perform calculations specific to
        the type of test being run.
        
        This one does nothing...
        '''
        _, pValue = stats.ttest_ind(thisFound, thisNotFound, equal_var=False)
        _, pValueN = stats.ttest_ind(thisFound, self.neutralValues, equal_var=False)
        
        toPrint = "Nothing to report, except this meaningless pValue: " + str(pValue) + "\n"
        toPrint += "And this one: " + str(pValueN) + "\n"
        toPrint += "Average of values: " + str(mean(thisFound)) + "\n"
        toPrint += "Average of nons: " + str(mean(thisNotFound)) + "\n"
        return toPrint    

class WordLengthOneWork(BaseOneWork):
    def getObservation(self, lyricText):
        if lyricText is None:
            return None
        return len(lyricText)
    
class WordLengthCorrelator(SentimentCorrelator):
    '''
    Because we got significant values when I made the example be based on word length,
    I've made this class to show it.
    '''
    testName = 'wordLength'
    oneWorkClass = WordLengthOneWork

class PitchOneWork(BaseOneWork):
    '''
    Derived class of BaseOneWork for checking the pitch height of individual notes
    compared to the average pitch height in the score.  Observations returned are the
    number of std. deviations of pitch height away from the average, in order to
    normalize between, say, Gregorian Chant and Luigi Nono.
    '''
    def setupTest(self, score):
        psList = [p.pitch.ps for p in score.parts[0].recurse().getElementsByClass('Note')]
        self.meanPitch = mean(psList)
        self.pitchStdDev = std(psList)
    
    def getObservation(self, lyricText):
        '''
        The observation for this lyric is the number of deviations of the pitch from
        the mean of the piece
        '''
        n = self.operativeNote
        if n is None:
            return None
        deviationsFromSongAverage = (n.pitch.ps - self.meanPitch) / self.pitchStdDev 
        return deviationsFromSongAverage       
    
class PitchCorrelator(SentimentCorrelator):
    testName = 'pitch'
    oneWorkClass = PitchOneWork
    
    def classSpecificResults(self, thisFound, thisNotFound, sentimentNumber):
        _, pValue = stats.ttest_ind(thisFound, thisNotFound, equal_var=False)
        
        deviationsFromMean = mean(thisFound)
        deviationsNonMatching = mean(thisNotFound)
        toPrint = ""
        toPrint += "Standard deviations away from song average pitch     : {:+7.5f} ".format(
            deviationsFromMean)
        toPrint += "(p = {:7.5f})\n".format(pValue)
        toPrint += "Standard deviations away for non-matching words in DB: {:+7.5f} ".format(
            deviationsNonMatching) + "\n"
        
        return toPrint

class RelativePitchOneWork(BaseOneWork):
    def getObservation(self, lyricText):
        '''
        Compare the pitch of this note to the pitch of the note immediately preceeding
        '''
        n = self.operativeNote
        if n is None:
            return None
        prev = n.previous('Note')
        if prev is None or not hasattr(prev, 'pitch'):
            return None
        return n.pitch.ps - prev.pitch.ps

class RelativePitchCorrelator(SentimentCorrelator):
    testName = 'relativePitch'
    oneWorkClass = RelativePitchOneWork

    def classSpecificResults(self, thisFound, thisNotFound, sentimentNumber):
        _, pValue = stats.ttest_ind(thisFound, thisNotFound, equal_var=False)
            
        meanPosDifference = mean(thisFound)
        meanNegDifference = mean(thisNotFound)
        toPrint = ""
        toPrint += "Semitones from previous pitch (mean)            : {:+7.5f} ".format(
            meanPosDifference)
        toPrint += "(p = {:7.5f})\n".format(pValue)
        toPrint += "Semitones from previous pitch (mean) for non-obs: {:+7.5f} ".format(
            meanNegDifference) + "\n"
        return toPrint

                     
class ConsonanceOneWork(BaseOneWork):
    def getObservation(self, lyricText):
        '''
        The observation for this lyric is 1 if the active chord symbol 
        including this note is consonant
        and 0 if not.  None is returned if there is no active chord symbol.
        '''
        n = self.operativeNote
        if n is None:
            return None
        cs = n.getContextByClass('ChordSymbol')
        if cs is None:
            return None
        
        csp = cs.pitches
        newChord = chord.Chord((n.pitch,) + csp)
        consonanceThis = 1 if newChord.isConsonant() else 0
        return consonanceThis


class ConsonanceCorrelator(SentimentCorrelator):
    testName = 'consonance'
    oneWorkClass = ConsonanceOneWork
    testWord = 'consonant'

    def classSpecificResults(self, thisFound, thisNotFound, sentimentNumber):
        _, pValue = stats.ttest_ind(thisFound, thisNotFound, equal_var=False)
        _, pValueN = stats.ttest_ind(thisFound, self.neutralValues, equal_var=False)
        posAverage = mean(thisFound)
        negAverage = mean(thisNotFound)
        neuAverage = mean(self.neutralValues)
        
        percentMoreConsonanceNeg = 100 * (posAverage - negAverage) / negAverage
        percentMoreConsonanceNeu = 100 * (posAverage - neuAverage) / neuAverage

        toPrint = ""        
        toPrint += "More {} than no such sentiment: {:+10.3f}% (p={:7.5f})\n".format(
            self.testWord, percentMoreConsonanceNeg, pValue)
        toPrint += "More {} than neutral words:     {:+10.3f}% (p={:7.5f})\n".format(
            self.testWord, percentMoreConsonanceNeu, pValueN)
        toPrint += "{} % for sentiment:     {:6.2f}%\n".format(self.testWord, posAverage * 100)
        toPrint += "{} % for non-sentiment: {:6.2f}%\n".format(self.testWord, negAverage * 100)
        toPrint += "{} % for neutral words: {:6.2f}%\n".format(self.testWord, neuAverage * 100)
        return toPrint    



class LooseConsonanceOneWork(BaseOneWork):
    '''
    Like ConsonanceOneWork, but with a looser definition of consonance, where if the
    melody note is in the chord, it is consonant, regardless of whether 
    the chord itself is consonant or not.
    '''
    def getObservation(self, lyricText):
        '''
        The observation for this lyric is 1 if the active chord symbol 
        including this note is consonant
        and 0 if not.  None is returned if there is no active chord symbol.
        '''
        n = self.operativeNote
        if n is None:
            return None
        cs = n.getContextByClass('ChordSymbol')
        if cs is None:
            return None
        
        csPitchClasses = [p.pitchClass for p in cs.pitches]
        consonanceThis = 1 if n.pitch.pitchClass in csPitchClasses else 0
        return consonanceThis


class LooseConsonanceCorrelator(ConsonanceCorrelator):
    testName = 'looseConsonance'
    oneWorkClass = LooseConsonanceOneWork


class MajorMinorOneWork(BaseOneWork):
    '''
    Simply returns 1 if the lyric is in the context of a major chord, 0 if in the
    context of a minor chord, and None for any other chord (including dominant 7ths etc.)
    '''
    def getObservation(self, lyricText):
        n = self.operativeNote
        if n is None:
            return None
        cs = n.getContextByClass('ChordSymbol')
        if cs is None:
            return None
        
        if cs.isMajorTriad():
            return 1
        elif cs.isMinorTriad():
            return 0
        else:
            return None

class MajorMinorCorrelator(ConsonanceCorrelator):
    testName = 'majorMinor'
    oneWorkClass = MajorMinorOneWork
    testWord = 'major'
    
class BeatStrengthOneWork(BaseOneWork):
    def getObservation(self, lyricText):
        '''
        Returns the beatStrength of the note.
        '''
        n = self.operativeNote
        if n is None:
            return None
        return n.beatStrength

class BeatStengthCorrelator(SentimentCorrelator):
    testName = 'beatStrength'
    oneWorkClass = BeatStrengthOneWork

    def classSpecificResults(self, thisFound, thisNotFound, sentimentNumber):
        _, pValue = stats.ttest_ind(thisFound, thisNotFound, equal_var=False)
            
        meanPosBeatStrength = mean(thisFound)
        meanNegBeatStrength = mean(thisNotFound)
        toPrint = ""
        toPrint += "AverageBeatStength of Matches    : {:+7.5f} ".format(meanPosBeatStrength)
        toPrint += "(p = {:6.6f})\n".format(pValue)
        toPrint += "AverageBeatStength of Non-Matches: {:+7.5f} ".format(
            meanNegBeatStrength) + "\n"
        toPrint += "StdDev of Matches    : {:+7.5f}\n".format(std(thisFound))
        toPrint += "StdDev of NonMatches : {:+7.5f}\n".format(std(thisNotFound))
        
        return toPrint


class NoteLengthOneWork(BaseOneWork):
    def getObservation(self, lyricText):
        '''
        Returns the beatStrength of the note.
        '''
        n = self.operativeNote
        if n is None:
            return None
        return n.quarterLength


class NoteLengthCorrelator(SentimentCorrelator):
    testName = 'noteLength'
    oneWorkClass = NoteLengthOneWork

    def classSpecificResults(self, thisFound, thisNotFound, sentimentNumber):
        _, pValue = stats.ttest_ind(thisFound, thisNotFound, equal_var=False)
            
        meanPosNoteLength = mean(thisFound)
        meanNegNoteLength= mean(thisNotFound)
        toPrint = ""
        toPrint += "Average Note Length of Matches    : {:+7.5f} ".format(meanPosNoteLength)
        toPrint += "(p = {:6.6f})\n".format(pValue)
        toPrint += "Average Note Length of Non-Matches: {:+7.5f} ".format(
            meanNegNoteLength) + "\n"
        toPrint += "StdDev of Matches    : {:+7.5f}\n".format(std(thisFound))
        toPrint += "StdDev of NonMatches : {:+7.5f}\n".format(std(thisNotFound))
        
        return toPrint

class ModeOneWork(BaseOneWork):
    def setupTest(self, score):
        k = score.analyze('key')
        if k is None:
            return None
        m = k.mode
        if m == 'minor':
            self.modeNum = 0
        elif m == 'major':
            self.modeNum = 1
        else: # should not happen
            self.modeNum = None
        
    def getObservation(self, lyricText):
        '''
        Returns 1 for a major piece and 0 for a minor piece.
        '''
        return self.modeNum
        
class ModeCorrelator(SentimentCorrelator):
    '''
    Calculates a correlation between a sentiment and a mode (0 = minor, 1 = major) 
    for the whole piece.
    '''
    testName = 'mode'
    oneWorkClass = ModeOneWork

    def classSpecificResults(self, thisFound, thisNotFound, sentimentNumber):
        _, pValue = stats.ttest_ind(thisFound, thisNotFound, equal_var=False)
            
        meanPosMode = mean(thisFound)
        meanNegMode = mean(thisNotFound)
        toPrint = ""
        toPrint += "Mode Average of Matches    : {:+7.5f} ".format(meanPosMode)
        toPrint += "(p = {:6.6f})\n".format(pValue)
        toPrint += "Mode Average of Non-Matches: {:+7.5f} ".format(meanNegMode) + "\n"
        
        return toPrint
    
class NearModeOneWork(BaseOneWork):
    def __init__(self, nrcParsed=None, fileName=None, saveScore=False):
        super().__init__(nrcParsed, fileName, saveScore)
        self.resultCache = {}
        
    
    def getObservation(self, lyricText):
        '''
        Returns 1 for major and 0 for minor or the area between two measures 
        before and two measures after a lyric begins
        '''
        n = self.operativeNote
        if n is None:
            return None
        mn = n.measureNumber
        mnStart = mn - 2
        mnEnd = mn + 2
        if mnStart < 0:
            mnStart = 0

        if (mnStart, mnEnd) in self.resultCache:
            return self.resultCache[(mnStart, mnEnd)]
        
        sc = None
        for cs in n.contextSites():
            if 'Score' in cs.site.classSet and cs.site.hasPartLikeStreams():
                sc = cs.site
        if sc is None:
            print("No site...")
            return None
        
        try:
            k = sc.measures(mnStart, mnEnd).analyze('key')
        except Exception as e:
            print(e)
            self.resultCache[(mnStart, mnEnd)] = None        
            return None
        if k is None:
            self.resultCache[(mnStart, mnEnd)] = None
            return None
        m = k.mode
        if m == 'minor':
            num = 0
        elif m == 'major':
            num = 1
        else: # should not happen
            num = None
        self.resultCache[(mnStart, mnEnd)] = num
        return num
            

class NearModeCorrelator(ModeCorrelator):    
    testName = 'nearmode'
    oneWorkClass = NearModeOneWork

class TonalCertaintyOneWork(BaseOneWork):
    def setupTest(self, score):
        fe = native.TonalCertainty(score)
        results = fe.extract()
        self.tonalCertainty = results.vector[0]
        
    def getObservation(self, lyricText):
        return self.tonalCertainty
    
class TonalCertaintyCorrelator(ModeCorrelator):
    testName = 'tonalCertainty'
    oneWorkClass = TonalCertaintyOneWork

def runEveryTest(endNumber=None):
    thisModule = sys.modules[SentimentCorrelator.__module__]
    allCorrelatorNames = [x for x in thisModule.__dict__ if 'Correlator' in x]
    allCorrelatorClasses = [getattr(thisModule, x) for x in allCorrelatorNames]
    
    for useStopwords in (False, True):
#        for stemWords in (False, True):
        for ccClass in allCorrelatorClasses:
            print(ccClass.__name__, useStopwords)              
            sc = ccClass(endNumber=endNumber) # endNumber = None or a number like 100
            sc.useStopwords = useStopwords
            #sc.stemWords = stemWords
            sc.runAll()
    
    # comprehensive vocabulary
    for ccClass in allCorrelatorClasses:
        print(ccClass.__name__, "comprehensive")              
        sc = ccClass(endNumber=endNumber) # endNumber = None or a number like 100
        sc.comprehensiveVocabulary = True
        #sc.stemWords = stemWords
        sc.runAll()
                

def testOneWorkWithShow():
    sc = SentimentCorrelator()
    sc.useStopwords = True
    sc.nrcParsed = sc.parseNrc()    
    pOne = RelativePitchOneWork(sc.nrcParsed, 
                        (sc.getBaseDir('../wikifonia_en_chords_lyrics') + os.sep +
                            'wikifonia-4411.mxl'), 
                        True)
    pOne.saveScore = True
    pOne.run()
    pOne.savedScore.show()

if __name__ == '__main__':
#    testOneWorkWithShow()
#    exit()
    
#    runEveryTest()
#    exit()
    
#    ccClass = NoteLengthCorrelator
#    ccClass = SentimentCorrelator
#    ccClass = BeatStengthCorrelator
#    ccClass = ConsonanceCorrelator
#    ccClass = LooseConsonanceCorrelator
#    ccClass = ModeCorrelator
#    ccClass = NearModeCorrelator
#    ccClass = RelativePitchCorrelator
#    ccClass = WordLengthCorrelator
    ccClass = MajorMinorCorrelator

    sc = ccClass(endNumber=None) # endNumber = None or a number like 100
    sc.comprehensiveVocabulary = True
    sc.useStopwords = True
    sc.stemWords = False
    sc.runAll()

