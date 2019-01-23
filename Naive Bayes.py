import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

# Getting data corpus about movie review from nltk.corpus
# Corpus will be use to collecting data
from nltk.corpus import movie_reviews

def word_feats(words):
	return dict([(word,true)for words in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

