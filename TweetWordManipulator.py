import nltk
from nltk.corpus import stopwords
from Sandbox import text_has_hyperlinks, text_has_emoji, keep_hashtags, text_has_hashtag
from Util import sideways_emoticons_neg, sideways_emoticons_pos


class TweetWordManipulator:

    def __init__(self, token_array):
        self.token_array = token_array

    def to_lower(self):
        self.token_array = [filtered_tokens.lower() for filtered_tokens in self.token_array]
        return self  # tokens to lowerscase

    def remove_stopwords(self):
        stop_words = set(stopwords.words('english'))
        self.token_array = [filtered_tokens for filtered_tokens in self.token_array if
                            filtered_tokens not in stop_words]
        return self  # stopword extraction

    def remove_hyperlinks(self):
        self.token_array = [filtered_tokens for filtered_tokens in self.token_array if
                            not text_has_hyperlinks(filtered_tokens)]
        return self  # hyperlink extraction

    def remove_emoji(self):
        self.token_array = [filtered_tokens for filtered_tokens in self.token_array if
                            not text_has_emoji(filtered_tokens)]
        return self  # emoji extraction

    def remove_sideways_emoticons(self):
        self.token_array = [filtered_tokens for filtered_tokens in self.token_array if filtered_tokens
                            not in (
                                    sideways_emoticons_neg + sideways_emoticons_pos)]
        return self  # emoji sideways extraction

    def remove_punctuation(self):
        self.token_array = [filtered_tokens for filtered_tokens in self.token_array if
                            filtered_tokens.isalpha()] + keep_hashtags(
            self.token_array)
        return self  # punctuation extraction

    def perform_lemmatization(self):
        wordnet_lemmatizer = nltk.WordNetLemmatizer()
        self.token_array = [wordnet_lemmatizer.lemmatize(filtered_tokens)
                            for filtered_tokens in self.token_array]
        return self  # Lemmatization applied

    def remove_hastags(self):
        self.token_array = [filtered_tokens for filtered_tokens in self.token_array if
                            not text_has_hashtag(filtered_tokens)]
        return self

    def keep_pos_tags(self, pos_selector=0):
        """
        :param pos_selector: int value that designates what set of POS are used
        0: default -> no POS tags selected
        1:  verbs & adjectives_pronouns
        2: 1 + nouns_pronouns
        3: 1 + 2 + adverbs
        :return:
        """
        verbs = ["VB", "VBD", "VBZ", "VBP", "VBN", "VBG"]
        adjectives_pronouns = ["JJ", "JJR", "JJS", "TO"]
        nouns_pronouns = ["NN", "NNS", "NNPS", "NNP", "PDT", "POS", "PRP", "PRP$"]
        adverbs = ["RB", "RBR", "RBS", "RP"]
        tokens_list = []
        tagged_tokens = nltk.pos_tag(self.token_array)
        for w, t in tagged_tokens:
            if pos_selector == 0:
                tokens_list.append(w)
            if pos_selector == 1:
                if t in verbs + adjectives_pronouns:
                    tokens_list.append(w)
            if pos_selector == 2:
                if t in verbs + adjectives_pronouns + nouns_pronouns:
                    tokens_list.append(w)
            if pos_selector == 3:
                if t in verbs + adjectives_pronouns + nouns_pronouns + adverbs:
                    tokens_list.append(w)
            self.token_array = tokens_list
        return self

    def get_tokens(self):
        return self.token_array
