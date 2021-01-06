import nltk
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from Sandbox import *
from TweetWordManipulator import TweetWordManipulator
from nltk import FreqDist
from nltk.tokenize import TweetTokenizer
import random
import sklearn
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from nltk.classify import SklearnClassifier
from nltk.metrics import precision, recall, f_measure
from nltk import collections
import pandas as pd
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


def sample_seperation(dataset, train_size_fraction):
    break_point = int(round(len(dataset) * train_size_fraction, 0))
    train_set, test_set = dataset[:break_point], dataset[break_point:]
    return train_set, test_set


def perform_preprocessing(tweet_collection):  # takes as argument the fileid
    tweet_word_manipulator = TweetWordManipulator(tokens_in_tweet_collection(twitter_samples.strings(tweet_collection)))
    tokens = tweet_word_manipulator.to_lower().remove_stopwords().remove_hyperlinks().remove_emoji(). \
        remove_sideways_emoticons().remove_punctuation().remove_hastags().perform_lemmatization().get_tokens()
    return tokens


def part_of_speech_tag(tokens):
    tag_tokens = nltk.pos_tag(tokens)
    return tag_tokens


def get_tagged_tokens(tag_tokens):
    tokens_list = []
    for w, t in tag_tokens:
        if t == 'VB' or t == 'VBD' or t == 'VBZ' or t == 'VBP' or t == 'VBN' or t == 'VBG' or t == 'JJ' or t == 'JJR' or t == 'JJS':
            tokens_list.append(w)
    return tokens_list


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
    return set(words_bag)


def array_of_tokens_in_tweet_collection_processed(tweet_collection, pos_flag=0):
    tokens_in_sent = array_of_tokens_in_tweets(tweet_collection)
    for tweet in tokens_in_sent:
        process_tweet = TweetWordManipulator(tweet).remove_emoji().remove_hastags(). \
            remove_hyperlinks().remove_punctuation().remove_sideways_emoticons().remove_stopwords().to_lower().keep_pos_tags(
            pos_flag) \
            .get_tokens()
        a = []
        for processed_token in process_tweet:
            a.append(processed_token)
        tweet.clear()
        for e in a:
            tweet.append(e)
    return tokens_in_sent  # [["token", "token", ...],[],[],....]
    # [[{"token":T, "token":F, ...}, "pos"],[],[],....]


def create_labeled_tuples_w_tokens(tokens_in_tweets, label):
    res = []
    for tweet_token_list in tokens_in_tweets:
        element = (tweet_token_list, label)
        res.append(element)
    return res  # [([.,..], pos|neg),...]


def create_feature_list_for_numeric(labeled_featured_sets):
    feature_list = [k for k, value in labeled_featured_sets[0].items()] + ["tag"]
    return feature_list


def labeled_featured_sets_to_numeric(labeled_featured_sets):
    # takes as input a tuple of the form:
    # ({'contains(waste)': False, 'contains(lot)': False, ...}, "pos")
    f = lambda e: 1 if e else 0
    y = lambda e: 1 if (e == "pos") else 0
    featured_values = [f(v) for (k, v) in labeled_featured_sets[0].items()] + [y(labeled_featured_sets[1])]
    return featured_values


def load_corpus_from_corona(file_name):
    reader = TwitterCorpusReader("./covid19-tweetCorpus", file_name, encoding="utf-8")
    tweet_docs = reader.docs()
    tweet_texts = []
    for tw_obj in tweet_docs:
        tweet_texts.append(tw_obj.get('text').encode('utf-8').decode("utf-8"))
    return tweet_docs, tweet_texts


def generate_classification_report(test_set):
    refset = collections.defaultdict(set)
    testset = collections.defaultdict(set)
    for i, (d, c) in enumerate(test_set):
        refset[c].add(i)
        observed = classifier.classify(d)
        testset[observed].add(i)
    print("Here's the Evaluation of the Naive Bayes Classifier:")
    print('Positive precision:', round(precision(refset['pos'], testset['pos']), 3))
    print('Positive recall:', round(recall(refset['pos'], testset['pos']), 3))
    print('Positive F-measure:', round(f_measure(refset['pos'], testset['pos']), 3))
    print('Negative precision:', round(precision(refset['neg'], testset['neg']), 3))
    print('Negative recall:', round(recall(refset['neg'], testset['neg']), 3))
    print('Negative F-measure:', round(f_measure(refset['neg'], testset['neg']), 3))


#####################################################################################################################
######################################PREPROCESSING##################################################################
#####################################################################################################################


# tweet_texts_pos = twitter_samples.strings("positive_tiny.json")
# tweet_texts_neg = twitter_samples.strings("negative_tiny.json")

tweet_texts_pos = twitter_samples.strings("positive_tweets.json")
tweet_texts_neg = twitter_samples.strings("negative_tweets.json")

tweets_processed_pos = array_of_tokens_in_tweet_collection_processed(tweet_texts_pos)
tweets_processed_neg = array_of_tokens_in_tweet_collection_processed(tweet_texts_neg)

tweets_processed_pos_labeled = create_labeled_tuples_w_tokens(tweets_processed_pos, "pos")
tweets_processed_neg_labeled = create_labeled_tuples_w_tokens(tweets_processed_neg, "neg")

documents = tweets_processed_pos_labeled + tweets_processed_neg_labeled

random.shuffle(documents)

#####################################################################################################################
######################################BAG OF WORDS###################################################################
#####################################################################################################################

# pos_tweet_tokens_bag = tokens_in_tweet_collection(twitter_samples.strings("positive_tiny.json"))
# neg_tweet_tokens_bag = tokens_in_tweet_collection(twitter_samples.strings("negative_tiny.json"))

pos_tweet_tokens_bag = tokens_in_tweet_collection(twitter_samples.strings("positive_tweets.json"))
neg_tweet_tokens_bag = tokens_in_tweet_collection(twitter_samples.strings("negative_tweets.json"))

pos_tweet_tokens_processed = TweetWordManipulator(pos_tweet_tokens_bag).remove_emoji().remove_hastags(). \
    remove_hyperlinks().remove_punctuation().remove_sideways_emoticons().remove_stopwords().to_lower().keep_pos_tags(3). \
    get_tokens()

neg_tweet_tokens_processed = TweetWordManipulator(neg_tweet_tokens_bag).remove_emoji().remove_hastags(). \
    remove_hyperlinks().remove_punctuation().remove_sideways_emoticons().remove_stopwords().to_lower().keep_pos_tags(3). \
    get_tokens()

pos_bow_fr = FreqDist(pos_tweet_tokens_processed).most_common(1000)
neg_bow_fr = FreqDist(neg_tweet_tokens_processed).most_common(1000)

pos_bow = [w for (w, f) in pos_bow_fr]
neg_bow = [w for (w, f) in neg_bow_fr]

bow = pos_bow + neg_bow


def document_features(tweet_tokens):
    document_words = set(tweet_tokens)
    features = {}
    for word in bow:
        features['contains(%s)' % word] = (word in document_words)
    return features


featuresets = [(document_features(tweet), tweet_cat) for (tweet, tweet_cat) in documents]

#############################################################################################
#                                  A. NLTK-NB-CLASSIFIER                                    #
#############################################################################################

print("================================================================")
print("1./ NLTK-NB CLASSIFIER EVALUATIONS: (TWITTER-SAMPLES AND CORONA)")
print("================================================================")

# 1. TWITTER-SAMPLES

train_size_fraction = 0.7
break_point = int(round(len(featuresets) * train_size_fraction, 0))
train_set, test_set = featuresets[:break_point], featuresets[break_point:]

print("===================================================")
print("A1./ NLTK-NB: EVALUATE CLASSIFIER ON TWITTER CORPUS")
print("===================================================")

classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Accuracy Measure: ", round(nltk.classify.accuracy(classifier, test_set), 3))
classifier.show_most_informative_features(5)
generate_classification_report(test_set)

# 2. CORONA SAMPLES

# ===================================
# EDIT/PROCESS CORONA POSITIVE CORPUS
# ===================================

corona_tweet_docs_pos, corona_tweet_texts_pos = load_corpus_from_corona("corona_pos_tagged.json")
corona_tweets_processed_pos = array_of_tokens_in_tweet_collection_processed(corona_tweet_texts_pos)
corona_tweets_processed_pos_labeled = create_labeled_tuples_w_tokens(corona_tweets_processed_pos, "pos")
documents_pos = corona_tweets_processed_pos_labeled
corona_featuresets_pos = [(document_features(tweet), tweet_cat) for (tweet, tweet_cat) in documents_pos]

# ===================================
# EDIT/PROCESS CORONA NEGATIVE CORPUS
# ===================================

corona_tweet_docs_neg, corona_tweet_texts_neg = load_corpus_from_corona("corona_neg_tagged.json")
corona_tweets_processed_neg = array_of_tokens_in_tweet_collection_processed(corona_tweet_texts_neg)
corona_tweets_processed_neg_labeled = create_labeled_tuples_w_tokens(corona_tweets_processed_neg, "neg")
documents_neg = corona_tweets_processed_neg_labeled
corona_featuresets_neg = [(document_features(tweet), tweet_cat) for (tweet, tweet_cat) in documents_neg]

# print("Accuracy Measure: ", round(nltk.classify.accuracy(classifier, corona_neg_test_set), 3))
# generate_classification_report(corona_neg_test_set)


print("=========================================================")
print("A2./ NLTK-NB: EVALUATE CLASSIFIER ON CORONA TAGGED CORPUS")
print("=========================================================")

corona_featuresets = corona_featuresets_pos + corona_featuresets_neg
corona_test_set = corona_featuresets
print("Accuracy Measure: ", round(nltk.classify.accuracy(classifier, corona_test_set), 3))
classifier.show_most_informative_features(5)
generate_classification_report(corona_test_set)

#############################################################################################
#                                  B. SCIKIT-DT CLASSIFIER                                  #
#############################################################################################

print()
print()
print("=======================================================================================")
print("B./ SCIKIT-LEARN DECISION TREE EVALUATIONS: (TWITTER-SAMPLES AND CORONA) CLASSIFICATION")
print("=======================================================================================")

# B1. TWITTER-SAMPLES


features_for_dataframe = create_feature_list_for_numeric(featuresets[0])

twitter_samples_featured_sets_array = [labeled_featured_sets_to_numeric(x) for x in featuresets]

twitter_samples_dataframe = pd.DataFrame(twitter_samples_featured_sets_array, columns=features_for_dataframe,
                                         dtype="int8")

print("==================================================================")
print("B1./ SCIKIT-LEARN DT: EVALUATE CLASSIFIER ON TWITTER SAMPLE CORPUS")
print("==================================================================")
print("Twitter-samples Dataframe shape is: ", twitter_samples_dataframe.shape)

X = twitter_samples_dataframe.drop("tag", 1)

y = twitter_samples_dataframe["tag"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

clfDT = tree.DecisionTreeClassifier()
clfDT.fit(x_train, y_train)
y_test_pred_DT = clfDT.predict(x_test)

confMatrixTestDT = confusion_matrix(y_test, y_test_pred_DT, labels=None)

print(confMatrixTestDT)
print('TWITTER SAMPLES RESULTS FOR DT CLASSIFIER')
print("Accuracy Score:")
print(accuracy_score(y_test, y_test_pred_DT))
print('Tree: Macro Precision, recall, f1-score')
print(precision_recall_fscore_support(y_test, y_test_pred_DT, average='macro'))
print(f1_score(y_test, y_test_pred_DT))
print("***********************")
print(classification_report(y_test, y_test_pred_DT))

print("=====================")
print("=====================")
print("=====================")

# B2. CORONA-SAMPLES


corona_samples_featured_sets_array = [labeled_featured_sets_to_numeric(x) for x in corona_test_set]

corona_samples_dataframe = pd.DataFrame(corona_samples_featured_sets_array, columns=features_for_dataframe,
                                        dtype="int8")
print("===================================================")
print("B2./ SCIKIT-LEARN DT: EVALUATE CLASSIFIER ON CORONA")
print("===================================================")

print("Corona samples Dataframe shape is: ", corona_samples_dataframe.shape)
X_covid = corona_samples_dataframe.drop("tag", 1)
y_covid_true = corona_samples_dataframe["tag"]
y_covid_pred_DT = clfDT.predict(X_covid)
confMatrixTestDT = confusion_matrix(y_covid_true, y_covid_pred_DT, labels=None)
print(confMatrixTestDT)
print('CORONA SAMPLES RESULTS FOR DT CLASSIFIER')
print("Accuracy Score:")
print(accuracy_score(y_covid_true, y_covid_pred_DT))
print('Tree: Macro Precision, recall, f1-score')
print(precision_recall_fscore_support(y_covid_true, y_covid_pred_DT, average='macro'))
print(f1_score(y_covid_true, y_covid_pred_DT))
print("***********************")
print(classification_report(y_covid_true, y_covid_pred_DT))

#############################################################################################
#                             C. SCIKIT LOGISTIC REGRESSION CLASSIFIER                      #
#############################################################################################

log_reg_model = LogisticRegression(max_iter=10000)
clfLR = log_reg_model.fit(x_train, y_train)

y_pred_twitter_LR = clfLR.predict(x_test)

print("===================================================================================")
print("C1./ SCIKIT-LEARN LOGISTIC REGRESSION: EVALUATE CLASSIFIER ON TWITTER SAMPLE CORPUS")
print("===================================================================================")

confMatrixTestLR = confusion_matrix(y_test, y_pred_twitter_LR, labels=None)

print(confMatrixTestLR)
print('TWITTER SAMPLES RESULTS FOR LR CLASSIFIER')
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred_twitter_LR))
print('Tree: Macro Precision, recall, f1-score')
print(precision_recall_fscore_support(y_test, y_pred_twitter_LR, average='macro'))
print(f1_score(y_test, y_pred_twitter_LR))
print("***********************")
print(classification_report(y_test, y_pred_twitter_LR))

print("============================================================================")
print("C2./ SCIKIT-LEARN LOGISTIC REGRESSION: EVALUATE CLASSIFIER ON CORONA SAMPLES")
print("============================================================================")


y_pred_corona_LR = clfLR.predict(X_covid)

confMatrixTestLR = confusion_matrix(y_covid_true, y_pred_corona_LR, labels=None)

print(confMatrixTestLR)
print('TWITTER SAMPLES RESULTS FOR LR CLASSIFIER')
print("Accuracy Score:")
print(accuracy_score(y_covid_true, y_pred_corona_LR))
print('Tree: Macro Precision, recall, f1-score')
print(precision_recall_fscore_support(y_covid_true, y_pred_corona_LR, average='macro'))
print(f1_score(y_covid_true, y_pred_corona_LR))
print("***********************")
print(classification_report(y_covid_true, y_pred_corona_LR, target_names=["neg", "pos"]))

