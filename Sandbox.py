import os
import nltk
from nltk import FreqDist
import re
import emoji
from nltk.corpus import twitter_samples
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, TwitterCorpusReader
from Util import sideways_emoticons_neg, sideways_emoticons_pos
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def number_of_tweets(tweet_collection):
    """Accepts as argument the tweet texts of a collection
    i.e. twitter_samples_pos_texts = twitter_samples.strings("positive_tweets.json")"""
    return len(tweet_collection)


def sentences_in_tweet_collection(tweet_collection):
    """Accepts as argument the tweet texts of a collection
    i.e. twitter_samples_pos_texts = twitter_samples.strings("positive_tweets.json")"""
    all_sentences = []  # array of arrays - 2D
    all_sentences_flat = []  # array 1D
    for tweet in tweet_collection:
        all_sentences.append(sent_tokenize(tweet))
    for tweet in all_sentences:
        for tweet_sentence in tweet:
            all_sentences_flat.append(tweet_sentence)
    return all_sentences_flat


def array_of_tokens_in_tweets(tweet_collection):
    """Accepts as argument the tweet texts of a collection
        i.e. twitter_samples_pos_texts = twitter_samples.strings("positive_tweets.json")"""
    tknzr = TweetTokenizer()
    all_sentences = []
    all_sentences_tokenized = []
    tokens_in_tweets = []
    for tweet in tweet_collection:
        all_sentences.append(sent_tokenize(tweet))
    for tweet in all_sentences:
        sent_tokens = []
        for sent in tweet:
            sent_tokens.append(tknzr.tokenize(sent))
        all_sentences_tokenized.append(sent_tokens)
    for tweet in all_sentences_tokenized:
        token_tweet_bucket = []
        for sent in tweet:
            for token in sent:
                token_tweet_bucket.append(token)
        tokens_in_tweets.append(token_tweet_bucket)
    return tokens_in_tweets


def number_of_sentences_in_tweet_collection(tweet_collection):
    """Accepts as argument the tweet texts of a collection
    i.e. twitter_samples_pos_texts = twitter_samples.strings("positive_tweets.json")"""
    return len(sentences_in_tweet_collection(tweet_collection))


def tokens_in_tweet_collection(tweet_collection):
    """Accepts as argument the tweet texts of a collection
    i.e. twitter_samples_pos_texts = twitter_samples.strings("positive_tweets.json")"""
    tknzr = TweetTokenizer()
    collection_sentences_flat = sentences_in_tweet_collection(tweet_collection)
    collection_tokens = []
    collection_tokens_flat = []
    for sentence in collection_sentences_flat:
        collection_tokens.append(tknzr.tokenize(sentence))
    for sentence_tokens in collection_tokens:
        for token in sentence_tokens:
            collection_tokens_flat.append(token)
    return collection_tokens_flat


def get_words_per_sentence(tweet_collection):
    """Accepts as argument the tweet texts of a collection
    i.e. twitter_samples_pos_texts = twitter_samples.strings("positive_tweets.json")"""
    return round(
        len(tokens_in_tweet_collection(tweet_collection)) / number_of_sentences_in_tweet_collection(tweet_collection),
        2)


def get_lexical_richness(tweet_collection):
    """Accepts as argument the tweet texts of a collection
    i.e. twitter_samples_pos_texts = twitter_samples.strings("positive_tweets.json")"""
    tokens_lower = [tokens.lower() for tokens in tokens_in_tweet_collection(tweet_collection)]
    vocabulary = set(tokens_lower)
    return round(len(tokens_lower) / len(vocabulary), 2)


def text_has_emoji(text):
    """
    :param text: sentence or token to be checked if it contains emoji character
    :return: boolean value
    """
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return True
    return False


def get_number_of_emojis(tweet_collection):
    number_of_emojis_in_tweet_collection = 0
    tokens = tokens_in_tweet_collection(tweet_collection)
    for token in tokens:
        if text_has_emoji(token):
            number_of_emojis_in_tweet_collection += 1
    return number_of_emojis_in_tweet_collection


def text_has_hyperlinks(token):
    urls_in_token = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', token)
    if len(urls_in_token) != 0:
        return True
    else:
        return False


def get_number_of_hyperlinks(tweet_collection):
    number_of_hyperlinks_in_tweet_collection = 0
    tokens = tokens_in_tweet_collection(tweet_collection)
    for token in tokens:
        if text_has_hyperlinks(token):
            number_of_hyperlinks_in_tweet_collection += 1
    return number_of_hyperlinks_in_tweet_collection


def text_has_hashtag(token):
    hashtags_in_token = re.findall(r"#(\w+)", token)
    if len(hashtags_in_token) != 0:
        return True
    else:
        return False


def remove_hastags(tokens):
    return [filtered_tokens for filtered_tokens in tokens if not text_has_hashtag(filtered_tokens)]


def get_number_of_hashtags(tweet_collection):
    number_of_hashtags_in_tweet_collection = 0
    tokens = tokens_in_tweet_collection(tweet_collection)
    for token in tokens:
        if text_has_hashtag(token):
            number_of_hashtags_in_tweet_collection += 1
    return number_of_hashtags_in_tweet_collection


def to_lower(tokens):
    return [filtered_tokens.lower() for filtered_tokens in tokens]  # tokens to lowerscase


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [filtered_tokens for filtered_tokens in tokens if filtered_tokens not in stop_words]  # stopword extraction


def remove_hyperlinks(tokens):
    return [filtered_tokens for filtered_tokens in tokens if
            not text_has_hyperlinks(filtered_tokens)]  # hyperlink extraction


def remove_emoji(tokens):
    return [filtered_tokens for filtered_tokens in tokens if not text_has_emoji(filtered_tokens)]  # emoji extraction


def remove_sideways_emoticons(tokens):
    return [filtered_tokens for filtered_tokens in tokens if filtered_tokens
            not in (sideways_emoticons_neg + sideways_emoticons_pos)]  # emoji sideways extraction


def keep_hashtags(tokens):
    hashtags = []
    for token in tokens:
        if text_has_hashtag(token):
            hashtags.append(token)
    return hashtags


def remove_punctuation(tokens):
    return [filtered_tokens for filtered_tokens in tokens if filtered_tokens.isalpha()] + keep_hashtags(
        tokens)  # punctuation extraction


def perform_lemmatization(tokens):
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    return [wordnet_lemmatizer.lemmatize(filtered_tokens) for filtered_tokens in tokens]  # Lemmatization applied


def get_collocations(tweet_collection, nbest, length_filter=0):
    collocations_finder = BigramCollocationFinder.from_words(tokens_in_tweet_collection(tweet_collection))
    n_best_collocates = collocations_finder.nbest(BigramAssocMeasures.likelihood_ratio, nbest)
    # say you want to keep collocates where each word of the bigram pair needs to have a certain length (e.g. >2)
    # then call method with length_filter = 2
    filtered_collocations = [(one, two) for one, two in
                             collocations_finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)
                             if (len(one) > length_filter and len(two) > length_filter)]
    print("Size of pos/neg collocates list is: ", str(len(n_best_collocates)))
    return n_best_collocates, filtered_collocations


def get_Freq_Dist(tweet_collection):
    processed_tokens = tokens_in_tweet_collection(tweet_collection)
    processed_tokens = to_lower(processed_tokens)
    processed_tokens = remove_hyperlinks(processed_tokens)
    processed_tokens = remove_emoji(processed_tokens)
    processed_tokens = remove_sideways_emoticons(processed_tokens)
    processed_tokens = remove_punctuation(processed_tokens)
    processed_tokens = remove_stopwords(processed_tokens)
    freq_dist = FreqDist(processed_tokens)
    freq_dist.plot(50, cumulative=True)
    freq_dist_keys = freq_dist.keys()
    freq_dist_most_common_tokens = freq_dist.most_common(50)
    freq_dist_hapaxes = freq_dist.hapaxes()
    return freq_dist_keys, freq_dist_most_common_tokens, freq_dist_hapaxes


def get_hashtag_freq_dist(tweet_collection):
    tokens = tokens_in_tweet_collection(tweet_collection)
    hashtag_tokens = keep_hashtags(tokens)
    freq_dist = FreqDist(hashtag_tokens)
    freq_dist.plot(50, cumulative=True)
    freq_dist_keys_hashtags = freq_dist.keys()
    freq_dist_most_common_hashtags = freq_dist.most_common(50)
    freq_dist_hapaxes_hashtags = freq_dist.hapaxes()
    return freq_dist_keys_hashtags, freq_dist_most_common_hashtags, freq_dist_hapaxes_hashtags


def tweet_corpus_report(tweet_collection):
    # file = open("CovidReport.txt", "w+")
    """
    Method Call as: tweet_corpus_report(twitter_samples.strings("positive_tweets.json"))
    :param tweet_collection: The tweet collection to be processed
    :return:
    """
    newLine = os.linesep
    keys, most_common_tokens, hapaxes = get_Freq_Dist(tweet_collection)
    freq_dist_keys_hashtags, freq_dist_most_common_hashtags, freq_dist_hapaxes_hashtags = get_hashtag_freq_dist(
        tweet_collection)
    n_best_collocates, filtered_collocations = get_collocations(tweet_collection, 100)
    print("1. Number of tweets:")
    no_of_tweets = number_of_tweets(tweet_collection)
    print(number_of_tweets(tweet_collection))
    print("2. Number of sentences:")
    # file.write("2. Number of sentences:" + newLine)
    no_of_sentences = number_of_sentences_in_tweet_collection(tweet_collection)
    print(no_of_sentences)
    # file.write(str(no_of_sentences) + newLine)
    print("3. Number of words:")
    # file.write("3. Number of words:" + newLine)
    no_of_words = len(tokens_in_tweet_collection(tweet_collection))
    print(no_of_words)
    print("4. Words per sentence:")
    # file.write("4. Words per sentence:" + newLine)
    words_per_sentence = str(get_words_per_sentence(tweet_collection))
    print(words_per_sentence)
    # file.write(str(words_per_sentence) + newLine)
    print("5. Lexical Richness:")
    lexical_richness = get_lexical_richness(tweet_collection)
    print(lexical_richness)
    print("6. Sentences/tweets:")
    print(round(no_of_sentences / no_of_tweets, 2))
    print("7. Number of emojis:")
    # file.write(str(no_of_tweets) + newLine)
    # file.write(str(lexical_richness) + newLine)
    # file.write("6. Sentences/tweets:" + newLine)
    # file.write("5. Lexical Richness:" + newLine)
    # file.write(str(round(no_of_sentences / no_of_tweets, 2)) + newLine)
    # file.write("7. Number of emojis:" + newLine)
    # file.write(str(no_of_words) + newLine)
    no_of_emojis = get_number_of_emojis(tweet_collection)
    print(no_of_emojis)
    # file.write(str(no_of_emojis) + newLine)
    print("8. Number of emojis/tweet:")
    # file.write("8. Number of emojis/tweet:" + newLine)
    print(round(no_of_emojis / no_of_tweets, 2))
    # file.write(str(round(no_of_emojis / no_of_tweets, 2)) + newLine)
    print("10. Number of hyperlinks:")
    # file.write("10. Number of hyperlinks:" + newLine)
    # file.write("1. Number of tweets:" + newLine)
    no_of_hyperlinks = get_number_of_hyperlinks(tweet_collection)
    print(no_of_hyperlinks)
    # file.write(str(no_of_hyperlinks) + newLine)
    print("11. Number of hyperlinks per tweet:")
    print(round(no_of_hyperlinks / no_of_tweets, 2))
    print("16. Number of hashtags:")
    no_of_hashtags = get_number_of_hashtags(tweet_collection)
    print(no_of_hashtags)
    print("17. Number of hashtags per tweet:")
    print(round(no_of_hashtags / no_of_tweets, 2))
    print("12. FreqDistStaff")
    print(len(get_Freq_Dist(tweet_collection)))
    print("12.1. Keys")
    # print(keys)
    print("12.2. Most common tokens")
    print(most_common_tokens)
    print("12.3. Hapaxes")
    print(hapaxes)
    print("22. Collocations")
    print("22.1. N-best collocates")
    print(n_best_collocates)
    print("23. Most frequent hashtags")
    print(freq_dist_most_common_hashtags)
    print("23.1. hashtag keys")
    print(freq_dist_keys_hashtags)
    print("23.2. Most common hashtags")
    print(freq_dist_most_common_hashtags)
    print("23.3. Hashtag hapaxes")
    print(freq_dist_hapaxes_hashtags)
    # file.write(str(n_best_collocates) + newLine)
    # file.write("23. Most frequent hashtags" + newLine)
    # file.write(str(freq_dist_hapaxes_hashtags) + newLine)
    # file.write(str(freq_dist_most_common_hashtags) + newLine)
    # file.write("12.2. Most common tokens")
    # file.write(str(round(no_of_hyperlinks / no_of_tweets, 2)) + newLine)
    # file.write("22.1. N-best collocates" + newLine)
    # file.write("12.3. Hapaxes")
    # file.write("16. Number of hashtags:" + newLine)
    # file.write(str(most_common_tokens) + newLine)
    # file.write("23.3. Hashtag hapaxes" + newLine)
    # file.write("11. Number of hyperlinks per tweet:" + newLine)
    # file.write(str(no_of_hashtags) + newLine)
    # file.write(str(round(no_of_hashtags / no_of_tweets, 2)) + newLine)
    # file.write("17. Number of hashtags per tweet:" + newLine)

# def tweet_corpus_report(tweet_collection):
#     """
#     Method Call as: tweet_corpus_report(twitter_samples.strings("positive_tweets.json"))
#     :param tweet_collection: The tweet collection to be processed
#     :return:
#     """
#     keys, most_common_tokens, hapaxes = get_Freq_Dist(tweet_collection)
#     freq_dist_keys_hashtags, freq_dist_most_common_hashtags, freq_dist_hapaxes_hashtags = get_hashtag_freq_dist(
#         tweet_collection)
#     n_best_collocates, filtered_collocations = get_collocations(tweet_collection, 50)
#     no_of_tweets = number_of_tweets(tweet_collection)
#     no_of_sentences = number_of_sentences_in_tweet_collection(tweet_collection)
#     tokens = tokens_in_tweet_collection(tweet_collection)
#     vocabulary = set(tokens)
#     no_of_emojis = get_number_of_emojis(tweet_collection)
#     print("1. Number of tweets:")
#     print(no_of_tweets)
#     print("2. Number of sentences:")
#     print(no_of_sentences)
#     print("3. Number of words:")
#     print(len(tokens))
#     print("4. Words per sentence:")
#     print(str(len(tokens) / no_of_sentences))
#     print("5. Lexical Richness:")
#     print(len(tokens) / len(vocabulary))
#     print("6. Sentences/tweets:")
#     print(round(no_of_sentences / no_of_tweets, 2))
#     print("7. Number of emojis:")
#     print(no_of_emojis)
#     print("8. Number of emojis/tweet:")
#     print(round(no_of_emojis / no_of_tweets, 2))
#     print("10. Number of hyperlinks:")
#     print(get_number_of_hyperlinks(tweet_collection))
#     print("11. Number of hyperlinks per tweet:")
#     print(round(get_number_of_hyperlinks(tweet_collection) / number_of_tweets(tweet_collection), 2))
#     print("16. Number of hashtags:")
#     print(get_number_of_hashtags(tweet_collection))
#     print("17. Number of hashtags per tweet:")
#     print(round(get_number_of_hashtags(tweet_collection) / number_of_tweets(tweet_collection), 2))
#     print(len(get_Freq_Dist(tc)))
#     print("12. FreqDistStaff")
#     print("12.1. Keys")
#     print(keys)
#     print("12.2. Most common tokens")
#     print(most_common_tokens)
#     print("12.3. Hapaxes")
#     print(hapaxes)
#     print("22. Collocations")
#     print("22.1. N-best collocates")
#     print(n_best_collocates)
#     print("23. Most frequent hashtags")
#     print("23.1. hashtag keys")
#     print(freq_dist_keys_hashtags)
#     print("23.2. Most common hashtags")
#     print(freq_dist_most_common_hashtags)
#     print("23.3. Hashtag hapaxes")
#     print(freq_dist_hapaxes_hashtags)


# tc = twitter_samples.strings("negative_tweets.json")
# tweet_corpus_report(tc)

# reader = TwitterCorpusReader("./covid19-tweetCorpus", "corona.json", word_tokenizer=TweetTokenizer(), encoding="utf8")
# reader = TwitterCorpusReader("./covid19-tweetCorpus", "corona.json", word_tokenizer=TweetTokenizer(), encoding="cp437")
# tweet_array = reader.docs()
# corona_tweets = []
# for tweet_object in tweet_array:
#     corona_tweets.append(tweet_object["text"])
# tweet_corpus_report_covid(corona_tweets)


