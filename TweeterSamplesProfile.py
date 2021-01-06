import sys
from nltk.corpus import twitter_samples
from Sandbox import tweet_corpus_report

tc = twitter_samples.strings("positive_tweets.json")
sys.stdout = open("./reports/positive_tweets_report.txt", "w", encoding="utf8")
tweet_corpus_report(tc)
sys.stdout.close()

tc = twitter_samples.strings("negative_tweets.json")
sys.stdout = open("./reports/negative_tweets_report.txt", "w", encoding="utf8")
tweet_corpus_report(tc)
sys.stdout.close()