import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

# Getting data corpus about movie review from nltk.corpus
# Corpus will be use to collecting data
from nltk.corpus import movie_reviews

def word_feats(words):
	return dict([(word,true)for words in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])),'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])),'pos') for f in posids]

negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print 'train on %d instance, test on %d instance'%(len(trainfeats),len(testfeats))

classifier = NaiveBayesClassifier.train(trainfeats)
print 'accuracy: ', nltk.classify.util.accuracy(classifier,testfeats)

classifier.show_most_informative_features()