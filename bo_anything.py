from nltk import BigramCollocationFinder, BigramAssocMeasures
from nltk import TweetTokenizer
from nltk.corpus import twitter_samples
import pandas as pd
import emoji
from scipy._lib.decorator import getfullargspec
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from EmojiUtil import pos_demoji, neg_demoji
from Sandbox import tokens_in_tweet_collection, text_has_hyperlinks, text_has_hashtag, text_has_emoji
from Sandbox import get_collocations
from TweetWordManipulator import TweetWordManipulator
from Util import sideways_emoticons_pos, sideways_emoticons_neg
from sklearn import tree
from sklearn.metrics import classification_report


# ============================================================================

def bag_of_collocations(first_list_collocations, second_list_collocations):
    collocations_bag = []
    for collocation in first_list_collocations:
        occurances = 0
        for words in second_list_collocations:
            if collocation == words:
                occurances += 1
        if occurances == 0:
            collocations_bag.append(collocation)
    return collocations_bag


def bag_of_words(first_list_of_tokens, second_list_of_tokens):
    words_bag = []
    for word in first_list_of_tokens:
        first_list_appearances = 0
        for token in first_list_of_tokens:
            if word == token:
                first_list_appearances += 1
        second_list_appearances = 0
        for token in second_list_of_tokens:
            if word == token:
                second_list_appearances += 1

        if second_list_appearances == 0:
            words_bag.append(word)
        else:
            if (first_list_appearances / second_list_appearances) > 10:
                words_bag.append(word)
    return list(set(words_bag))


def perform_preprocessing(tweet_collection):  # takes as argument the fileid - use it for bow
    tweet_word_manipulator = TweetWordManipulator(tokens_in_tweet_collection(twitter_samples.strings(tweet_collection)))
    tokens = tweet_word_manipulator.to_lower().remove_stopwords().remove_hyperlinks().remove_emoji(). \
        remove_sideways_emoticons().remove_punctuation().remove_hastags().perform_lemmatization().get_tokens()
    return tokens


def perform_preprocessing_2(tweet_collection):  # takes as argument the fileid - use it for bow
    tweet_word_manipulator = TweetWordManipulator(tokens_in_tweet_collection(tweet_collection))
    tokens = tweet_word_manipulator.to_lower().remove_stopwords().remove_hyperlinks().remove_emoji(). \
        remove_sideways_emoticons().remove_punctuation().remove_hastags().keep_pos_tags().perform_lemmatization().get_tokens()
    return tokens


def sample_seperation(dataset):
    train_size_fraction = 0.5
    break_point = int(round(len(dataset) * train_size_fraction, 0))
    train_set, test_set = dataset[:break_point], dataset[break_point:]
    return train_set, test_set


def export_material():
    #####################################################################################################################
    #                                            DATASET SPLITTING                                                      #
    #####################################################################################################################
    tweets_pos = twitter_samples.docs("positive_tweets.json")
    tweets_neg = twitter_samples.docs("negative_tweets.json")
    tweets_pos_bow = tweets_pos[:2000]
    tweets_neg_bow = tweets_neg[:2000]
    tweets_pos = tweets_pos[2000:]  # this is the dataset pos to be fed to the classifier
    tweets_neg = tweets_neg[2000:]  # this is the dataset neg to be fed to the classifier

    #####################################################################################################################
    #                                            BO GENERATION                                                          #
    #####################################################################################################################

    tweets_pos_text = [tweet.get("text") for tweet in tweets_pos_bow]  # ARRAY OF TWEETS -> RESULT OF .strings()
    tweets_neg_text = [tweet.get("text") for tweet in tweets_neg_bow]  # ARRAY OF TWEETS -> RESULT OF .strings()

    # 1. WORDS!

    all_tokens_pos = perform_preprocessing_2(tweets_pos_text)
    all_tokens_neg = perform_preprocessing_2(tweets_neg_text)
    pos_bow = bag_of_words(all_tokens_pos, all_tokens_neg)
    neg_bow = bag_of_words(all_tokens_neg, all_tokens_pos)

    # 2. COLLOCATIONS! (NO PREPROCESSING !!!!)

    all_col_pos = get_collocations(tweets_pos_text, 1000)
    all_col_neg = get_collocations(tweets_neg_text, 1000)
    pos_boc = bag_of_collocations(all_col_pos, all_col_neg)
    neg_boc = bag_of_collocations(all_col_neg, all_col_pos)

    # 3. HASHTAGS (NO PREPROCESSING !!!!)

    all_hashtags_pos = [token for token in tokens_in_tweet_collection(tweets_pos_text) if text_has_hashtag(token)]
    all_hashtags_neg = [token for token in tokens_in_tweet_collection(tweets_neg_text) if text_has_hashtag(token)]
    pos_boh = bag_of_words(all_hashtags_pos, all_hashtags_neg)
    neg_boh = bag_of_words(all_hashtags_neg, all_hashtags_pos)

    # 4. SIDEWAYS EMOTICONS

    all_sideways_emo_pos_bow = sideways_emoticons_pos
    all_sideways_emo_neg_bow = sideways_emoticons_neg

    # 5. EMOJIS

    all_emojis_pos = [emoji.emojize(pos_e) for pos_e in pos_demoji]
    all_emojis_neg = [emoji.emojize(neg_e) for neg_e in neg_demoji]

    return tweets_pos, tweets_neg, pos_bow, neg_bow, pos_boc, neg_boc, pos_boh, neg_boh, all_sideways_emo_pos_bow, \
           all_sideways_emo_neg_bow, all_emojis_pos, all_emojis_neg



