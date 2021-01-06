class TweetObject:

    def __init__(self, tweet_length, no_of_pos_bow, no_of_neg_bow, no_of_pos_colloc, no_of_neg_colloc,
                 no_of_pos_hashtags, no_of_neg_hashtags, no_of_pos_sideways_emo, no_of_neg_sideways_emo,
                 no_of_pos_emoji, no_of_neg_emoji, mentions, no_of_urls, retweets_count, no_of_hashtags,
                 tag):

        self.tweet_length = tweet_length            #OK
        self.no_of_pos_bow = no_of_pos_bow          #OK
        self.no_of_neg_bow = no_of_neg_bow          #OK
        self.no_of_pos_colloc = no_of_pos_colloc    #OK
        self.no_of_neg_colloc = no_of_neg_colloc    #OK
        self.no_of_pos_hashtags = no_of_pos_hashtags #OK
        self.no_of_neg_hashtags = no_of_neg_hashtags #OK
        self.no_of_pos_sideways_emo = no_of_pos_sideways_emo #OK
        self.no_of_neg_sideways_emo = no_of_neg_sideways_emo #OK
        self.no_of_pos_emoji = no_of_pos_emoji #OK
        self.no_of_neg_emoji = no_of_neg_emoji #OK
        self.mentions = mentions
        self.no_of_urls = no_of_urls
        self.retweets_count = retweets_count
        self.no_of_hashtags = no_of_hashtags
        self.tag = tag






