import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sklearn
import numpy as np

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


        for user_id in users:

            if users[user_id]['location'] is None:
                users[user_id]['location'] = 'Unknown'

            if users[user_id]['time_zone'] is None:
                users[user_id]['time_zone'] = 'Unknown'

            users[user_id]['followers_count'] = self.replaceNumeric(users[user_id]['followers_count'])
            users[user_id]['statuses_count'] = self.replaceNumeric(users[user_id]['statuses_count'])
            users[user_id]['favourites_count'] = self.replaceNumeric(users[user_id]['favourites_count'])
            users[user_id]['friends_count'] = self.replaceNumeric(users[user_id]['friends_count'])
            users[user_id]['listed_count'] = self.replaceNumeric(users[user_id]['listed_count'])



        # statuses_count, favourites_count, friends_count, listed_count

        minMax = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))

        numeric_data = [np.array(self.getColumn(users, 'favourites_count'))]
        print(numeric_data)
        numeric_data = minMax.fit_transform(numeric_data)
        print(numeric_data)

        return users

    def replaceNumeric(self, attrib):
        try:
            convert = int(attrib)
        except:
            convert = 0
        return convert

    def getColumn(self, data, name):
        col = []
        for key in data:
            col.append(data[key][name])
        return col





