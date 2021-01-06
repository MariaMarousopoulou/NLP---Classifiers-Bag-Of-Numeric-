import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
from Util import sideways_emoticons_neg, sideways_emoticons_pos


def sentences_in_tweet_collection(tweet_collection):
    all_sentences = []  # array of arrays
    for tweet in tweet_collection:
        all_sentences.append(sent_tokenize(tweet))
    return all_sentences


def number_of_sentences_in_tweet_collection(array_of_arrays_of_sentences):
    res = 0
    for array in array_of_arrays_of_sentences:
        res = res + len(array)
    return res


print("Tweets collections present in corpora: ", twitter_samples.fileids())
# Tweets w/ metadata
twitter_samples_pos = twitter_samples.docs("positive_tweets.json")
twitter_samples_neg = twitter_samples.docs("negative_tweets.json")
twitter_samples_other = twitter_samples.docs("tweets.20150430-223406.json")

# Tweets: text only
twitter_samples_pos_texts = twitter_samples.strings("positive_tweets.json")
twitter_samples_neg_texts = twitter_samples.strings("negative_tweets.json")
twitter_samples_other_texts = twitter_samples.strings("tweets.20150430-223406.json")

###############################
# 1. Number of tweets
###############################

no_of_pos = len(twitter_samples_pos)
no_of_neg = len(twitter_samples_neg)
no_of_other = len(twitter_samples_other)
print("Number of positive tweets is: ", str(no_of_pos))
print("Number of negative tweets is: ", str(no_of_neg))
print("Number of other tweets is: ", str(no_of_other))
print("Total Number of tweets is: ", str(no_of_other + no_of_neg + no_of_pos))

###############################
# 2. Number of sentences
###############################

tweet = twitter_samples.strings("negative_tweets.json")[1]
print(tweet)
print(sent_tokenize(tweet))

pos_sentences = sentences_in_tweet_collection(twitter_samples_pos_texts)
neg_sentences = sentences_in_tweet_collection(twitter_samples_neg_texts)
other_sentences = sentences_in_tweet_collection(twitter_samples_other_texts)

no_of_sentences_in_pos = number_of_sentences_in_tweet_collection(pos_sentences)
no_of_sentences_in_neg = number_of_sentences_in_tweet_collection(neg_sentences)
no_of_sentences_in_other = number_of_sentences_in_tweet_collection(other_sentences)

print("Number of sentences in positive tweet collection: ", str(no_of_sentences_in_pos))
print("Number of sentences in negative tweet collection: ", str(no_of_sentences_in_neg))
print("Number of sentences in other tweet collection: ", str(no_of_sentences_in_other))
print("Total number of sentences in tweet corpus: ", str(no_of_sentences_in_other + no_of_sentences_in_pos +
                                                         no_of_sentences_in_neg))

###############################
# 2. Number of words
###############################



