from nltk import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import TwitterCorpusReader
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from Sandbox import tokens_in_tweet_collection, text_has_hashtag, text_has_emoji
from Util import sideways_emoticons_neg, sideways_emoticons_pos
from bo_anything import export_material, perform_preprocessing_2
import pandas as pd


reader = TwitterCorpusReader("./covid19-tweetCorpus", "corona_tagged.json", encoding="utf-8")

tweet_docs = reader.docs()

tweet_objects = []

for i in range(len(tweet_docs)):

    tweet_objects.append(tweet_docs[i])


tweets_pos, tweets_neg, pos_bow, neg_bow, pos_boc, neg_boc, pos_boh, neg_boh, all_sideways_emo_pos_bow, \
all_sideways_emo_neg_bow, all_emojis_pos, all_emojis_neg = export_material()


def create_collocations(tweet_collection):
    tweet_collocations = BigramCollocationFinder.from_words(TweetTokenizer().tokenize(tweet_collection)). \
        nbest(BigramAssocMeasures.likelihood_ratio, 1000)
    return tweet_collocations


def word_analysis(tweet_text):
    no_of_pos_bow, no_of_neg_bow = 0, 0
    for token in perform_preprocessing_2(tweet_text):
        if token in pos_bow:
            no_of_pos_bow += 1
        if token in neg_bow:
            no_of_neg_bow += 1
    return no_of_pos_bow, no_of_neg_bow


def col_analysis(col_list):
    no_of_pos_colloc, no_of_neg_colloc = 0, 0
    for col in col_list:
        if col in pos_boc:
            no_of_pos_colloc += 1
        if col in neg_boc:
            no_of_neg_colloc += 1
    return no_of_pos_colloc, no_of_neg_colloc


def hashtag_analysis(tweet_text):
    no_of_pos_hashtags, no_of_neg_hashtags = 0, 0
    tokens = tokens_in_tweet_collection(tweet_text)
    hashtags = [hashtag for hashtag in tokens if text_has_hashtag(hashtag)]
    for hashtag in hashtags:
        if hashtag in pos_boh:
            no_of_pos_hashtags += 1
        if hashtag in neg_boh:
            no_of_neg_hashtags += 1
    return no_of_pos_hashtags, no_of_neg_hashtags


def sideways_emo_analysis(tweet_text):
    no_of_pos_sideways_emo, no_of_neg_sideways_emo = 0, 0
    tokens = tokens_in_tweet_collection(tweet_text)
    sideways_emojis = [sideway_emo for sideway_emo in tokens if sideway_emo in (sideways_emoticons_neg +
                                                                                sideways_emoticons_pos)]
    for sideway_emo in sideways_emojis:
        if sideway_emo in all_sideways_emo_pos_bow:
            no_of_pos_sideways_emo += 1
        if sideway_emo in all_sideways_emo_neg_bow:
            no_of_neg_sideways_emo += 1
    return no_of_pos_sideways_emo, no_of_neg_sideways_emo


def emoji_analysis(tweet_text):
    no_of_pos_emoji, no_of_neg_emoji = 0, 0
    tokens = tokens_in_tweet_collection(tweet_text)
    emojis = [emoji for emoji in tokens if text_has_emoji(emoji)]
    for emoji in emojis:
        if emoji in all_emojis_pos:
            no_of_pos_emoji += 1
        if emoji in all_emojis_neg:
            no_of_neg_emoji = + 1
    return no_of_pos_emoji, no_of_neg_emoji


def twitter_samples_metas_analysis(tweet_object):
    mentions, no_of_urls, retweets_count, no_of_hashtags = 0, 0, 0, 0
    mentions += len(tweet_object.get("entities").get("user_mentions"))
    no_of_urls += len(tweet_object.get("entities").get("urls"))
    retweets_count += tweet_object.get("retweet_count")
    no_of_hashtags += len(tweet_object.get("entities").get("hashtags"))
    return mentions, no_of_urls, retweets_count, no_of_hashtags


def corona_metas_analysis(tweet_object):
    mentions, no_of_urls, retweets_count, no_of_hashtags, tag = 0, 0, 0, 0, 0
    mentions += len(tweet_object.get("mentions"))
    no_of_urls += len(tweet_object.get("urls"))
    retweets_count += tweet_object.get("retweets_count")
    no_of_hashtags += len(tweet_object.get("hashtags"))
    if tweet_object.get("tag") == "pos":
        tag = 1
    return mentions, no_of_urls, retweets_count, no_of_hashtags, tag


corpus_data = []

for tweet in tweets_pos:  # pos -> tag = 1
    tweet_data_lang = []
    tweet_data_metas = []
    tweet_text = tweet.get("text")
    no_of_pos_bow, no_of_neg_bow = word_analysis(tweet_text)
    no_of_pos_colloc, no_of_neg_colloc = col_analysis(create_collocations(tweet_text))
    no_of_pos_hashtags, no_of_neg_hashtags = hashtag_analysis(tweet_text)
    no_of_pos_sideways_emo, no_of_neg_sideways_emo = sideways_emo_analysis(tweet_text)
    no_of_pos_emoji, no_of_neg_emoji = emoji_analysis(tweet_text)

    mentions, no_of_urls, retweets_count, no_of_hashtags = twitter_samples_metas_analysis(tweet)
    tweet_data_lang.append(len(tweet_text))
    tweet_data_lang.append(no_of_pos_bow)
    tweet_data_lang.append(no_of_neg_bow)
    tweet_data_lang.append(no_of_pos_colloc)
    tweet_data_lang.append(no_of_neg_colloc)
    tweet_data_lang.append(no_of_pos_hashtags)
    tweet_data_lang.append(no_of_neg_hashtags)
    tweet_data_lang.append(no_of_pos_sideways_emo)
    tweet_data_lang.append(no_of_neg_sideways_emo)
    tweet_data_lang.append(no_of_pos_emoji)
    tweet_data_lang.append(no_of_neg_emoji)

    tweet_data_metas.append(mentions)
    tweet_data_metas.append(no_of_urls)
    tweet_data_metas.append(retweets_count)
    tweet_data_metas.append(no_of_hashtags)
    tweet_data_metas.append(1)

    tweet_data = tweet_data_lang + tweet_data_metas
    corpus_data.append(tweet_data)

for tweet in tweets_neg:  # pos -> tag = 0
    tweet_data_lang = []
    tweet_data_metas = []
    tweet_text = tweet.get("text")
    no_of_pos_bow, no_of_neg_bow = word_analysis(tweet_text)
    no_of_pos_colloc, no_of_neg_colloc = col_analysis(create_collocations(tweet_text))
    no_of_pos_hashtags, no_of_neg_hashtags = hashtag_analysis(tweet_text)
    no_of_pos_sideways_emo, no_of_neg_sideways_emo = sideways_emo_analysis(tweet_text)
    no_of_pos_emoji, no_of_neg_emoji = emoji_analysis(tweet_text)

    mentions, no_of_urls, retweets_count, no_of_hashtags = twitter_samples_metas_analysis(tweet)
    tweet_data_lang.append(len(tweet_text))
    tweet_data_lang.append(no_of_pos_bow)
    tweet_data_lang.append(no_of_neg_bow)
    tweet_data_lang.append(no_of_pos_colloc)
    tweet_data_lang.append(no_of_neg_colloc)
    tweet_data_lang.append(no_of_pos_hashtags)
    tweet_data_lang.append(no_of_neg_hashtags)
    tweet_data_lang.append(no_of_pos_sideways_emo)
    tweet_data_lang.append(no_of_neg_sideways_emo)
    tweet_data_lang.append(no_of_pos_emoji)
    tweet_data_lang.append(no_of_neg_emoji)

    tweet_data_metas.append(mentions)
    tweet_data_metas.append(no_of_urls)
    tweet_data_metas.append(retweets_count)
    tweet_data_metas.append(no_of_hashtags)
    tweet_data_metas.append(0)

    tweet_data = tweet_data_lang + tweet_data_metas
    corpus_data.append(tweet_data)

tweet_data_df = pd.DataFrame(corpus_data, columns=["tweet_length",
                                                   "no_of_pos_bow",
                                                   "no_of_neg_bow",
                                                   "no_of_pos_colloc",
                                                   "no_of_neg_colloc",
                                                   "no_of_pos_hashtags",
                                                   "no_of_neg_hashtags",
                                                   "no_of_pos_sideways_emo",
                                                   "no_of_neg_sideways_emo",
                                                   "no_of_pos_emoji",
                                                   "no_of_neg_emoji",
                                                   "mentions",
                                                   "no_of_urls",
                                                   "retweets_count",
                                                   "no_of_hashtags",
                                                   "tag"])

print("Dataframe shape is: ", tweet_data_df.shape)

X = tweet_data_df.drop("tag", 1)
X = X.astype(float)

y = tweet_data_df["tag"]
y = y.astype(int)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clfDT = tree.DecisionTreeClassifier()

clfDT.fit(x_train, y_train)
y_test_pred_DT = clfDT.predict(x_test)

confMatrixTestDT = confusion_matrix(y_test, y_test_pred_DT, labels=None)

print(confMatrixTestDT)
print('Tree: Macro Precision, recall, f1-score')
print(precision_recall_fscore_support(y_test, y_test_pred_DT, average='macro'))
print(f1_score(y_test, y_test_pred_DT))

corpus_covid_data = []

for tweet in tweet_objects:
    tweet_data_lang = []
    tweet_data_metas = []
    tweet_text = tweet.get('text').encode('utf-8').decode("utf-8")
    no_of_pos_bow, no_of_neg_bow = word_analysis(tweet_text)
    no_of_pos_colloc, no_of_neg_colloc = col_analysis(create_collocations(tweet_text))
    no_of_pos_hashtags, no_of_neg_hashtags = hashtag_analysis(tweet_text)
    no_of_pos_sideways_emo, no_of_neg_sideways_emo = sideways_emo_analysis(tweet_text)
    no_of_pos_emoji, no_of_neg_emoji = emoji_analysis(tweet_text)

    mentions, no_of_urls, retweets_count, no_of_hashtags, tag = corona_metas_analysis(tweet)
    tweet_data_lang.append(len(tweet_text))
    tweet_data_lang.append(no_of_pos_bow)
    tweet_data_lang.append(no_of_neg_bow)
    tweet_data_lang.append(no_of_pos_colloc)
    tweet_data_lang.append(no_of_neg_colloc)
    tweet_data_lang.append(no_of_pos_hashtags)
    tweet_data_lang.append(no_of_neg_hashtags)
    tweet_data_lang.append(no_of_pos_sideways_emo)
    tweet_data_lang.append(no_of_neg_sideways_emo)
    tweet_data_lang.append(no_of_pos_emoji)
    tweet_data_lang.append(no_of_neg_emoji)

    tweet_data_metas.append(mentions)
    tweet_data_metas.append(no_of_urls)
    tweet_data_metas.append(retweets_count)
    tweet_data_metas.append(no_of_hashtags)
    tweet_data_metas.append(tag)

    tweet_data = tweet_data_lang + tweet_data_metas
    corpus_covid_data.append(tweet_data)

tweet_data_covid_df = pd.DataFrame(corpus_covid_data, columns=["tweet_length",
                                                               "no_of_pos_bow",
                                                               "no_of_neg_bow",
                                                               "no_of_pos_colloc",
                                                               "no_of_neg_colloc",
                                                               "no_of_pos_hashtags",
                                                               "no_of_neg_hashtags",
                                                               "no_of_pos_sideways_emo",
                                                               "no_of_neg_sideways_emo",
                                                               "no_of_pos_emoji",
                                                               "no_of_neg_emoji",
                                                               "mentions",
                                                               "no_of_urls",
                                                               "retweets_count",
                                                               "no_of_hashtags",
                                                               "tag"])

print("=================================")
print("=================================")
print("=================================")
print("=================================")
print("=================================")
print("Dataframe shape is: ", tweet_data_covid_df.shape)

XC = tweet_data_covid_df.drop("tag", 1)
XC = XC.astype(float)

yC_true = tweet_data_covid_df["tag"]
yC_true = yC_true.astype(int)

y_test_pred_covid_DT = clfDT.predict(XC)

confMatrixTestDT = confusion_matrix(yC_true, y_test_pred_covid_DT, labels=None)

print(confMatrixTestDT)
print('Tree: Macro Precision, recall, f1-score')
print(precision_recall_fscore_support(yC_true, y_test_pred_covid_DT, average='macro'))
print(f1_score(yC_true, y_test_pred_covid_DT))
