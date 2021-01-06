from nltk.corpus import TwitterCorpusReader
from nltk.tokenize import TweetTokenizer
import datetime
from nltk.tokenize import TweetTokenizer
from Sandbox import tweet_corpus_report
import re
import sys


def string_to_date(string):
    ymd = string.split("-")
    date = datetime.date(int(ymd[0]), int(ymd[1]), int(ymd[2]))
    return date


def sort_array_of_tweet_dictionaries_by_date(initial_array, order=False):
    sorted_list = sorted(initial_array, key=lambda k: string_to_date(k["date"]),
                         reverse=order)  # reverse=True -> Descending
    return sorted_list


def diff_dates(date1, date2):
    return abs(date2 - date1).days

sys.stdout = open("./reports/corona_tagged_report.txt", "w", encoding="utf8")

tknzr = TweetTokenizer()
print("ok")
reader = TwitterCorpusReader("./covid19-tweetCorpus", "corona_tagged.json", encoding="utf-8")
tweet_docs = reader.docs()
print(type(tweet_docs))
tweet_texts = []
for tw_obj in tweet_docs:
    tweet_texts.append(tw_obj.get('text').encode('utf-8').decode("utf-8"))

tweet_corpus_report(tweet_texts)
sys.stdout.close()
#
sys.stdout = open("./reports/covid_report.txt", "w", encoding="utf8")

tknzr = TweetTokenizer()
print("ok")
reader = TwitterCorpusReader("./covid19-tweetCorpus", "corona.json", encoding="utf-8")
tweet_docs = reader.docs()
print(type(tweet_docs))
tweet_texts = []
for tw_obj in tweet_docs:
    tweet_texts.append(tw_obj.get('text').encode('utf-8').decode("utf-8"))

tweet_corpus_report(tweet_texts)
sys.stdout.close()