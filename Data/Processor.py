import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sklearn

# -*- coding: utf-8 -*-


# noinspection PyMethodMayBeStatic

class Processor:

    def __init__(self):
        pass

    def processTweetText(self, tweets):

        for tweet in tweets:
            # Converting some values into lists
            tweets[tweet]['hashtags'] = re.findall("\".*\"", tweets[tweet]['hashtags'].lower())
            tweets[tweet]['mentions'] = re.findall("\".*\"", tweets[tweet]['mentions'])

            # Processing tweet text
            tweet_text = tweets[tweet]['text']

            # Low-casing and tokenizing words
            tweet_text = tweet_text.lower()
            tweet_text = nltk.word_tokenize(tweet_text) + tweets[tweet]['hashtags']

            # Removing non-textual content
            tweet_text = list(filter(lambda x: x.isalnum() and x != "https", tweet_text))

            # Stopwords removal
            stop_words = set(stopwords.words('english'))
            tweet_text = list(filter(lambda x: x not in stop_words, tweet_text))

            # Stemming
            ps = PorterStemmer()
            tweet_text = list(map(lambda x: ps.stem(x), tweet_text))

            # Testing
            tweets[tweet]['text'] = tweet_text
            # print(tweet_text)

        return tweets


    def processTweetMeta(self, tweets):

        for tweet in tweets:

            try:
                tweets[tweet]['retweet_count'] = int(tweets[tweet]['retweet_count'])
            except:
                tweets[tweet]['retweet_count'] = 0

            try:
                tweets[tweet]['favorite_count'] = int(tweets[tweet]['favorite_count'])
            except:
                tweets[tweet]['favorite_count'] = 0

        return tweets

    def processUserMeta(self, users):
        minMax = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)

        numeric_data = getRetweetCount(users)
        numeric_data = minMax.fit(numeric_data)

        for user_id in users:

            if users[user_id]['location'] is None:
                users[user_id]['location'] = 'Unknown'

            if users[user_id]['timezone'] is None:
                users[user_id]['timezone'] = 'Unknown'

        return users





