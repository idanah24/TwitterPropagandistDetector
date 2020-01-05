import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# -*- coding: utf-8 -*-


# noinspection PyMethodMayBeStatic

class Processor:

    def __init__(self):
        pass

    # This method performs standard NLP processing on text
    # Input: dictionary of tweet data
    # Output: dictionary of tweet data with processed text
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

            tweets[tweet]['text'] = tweet_text

        return tweets

    # This method processes non-text related tweet data
    # Input: a dictionary of tweet data
    # Output: a dictionary of processed tweet metadata
    def processTweetMeta(self, tweets):
        for tweet in tweets:
            if not tweets[tweet]['retweeted']:
                tweets[tweet]['retweeted'] = 'false'

            tweets[tweet]['retweet_count'] = self.replaceNumeric(tweets[tweet]['retweet_count'])
            tweets[tweet]['favorite_count'] = self.replaceNumeric(tweets[tweet]['favorite_count'])

        self.applyNormalization(tweets, 'retweet_count')
        self.applyNormalization(tweets, 'favorite_count')

        return tweets

    # This method processes a user data
    # Input: a dictionary of users data
    # Output: a dictionary of processed user data
    # TODO: Find a way to normalize location
    def processUserMeta(self, users):
        for user_id in users:

            if not users[user_id]['location']:
                users[user_id]['location'] = 'Unknown'

            if not users[user_id]['time_zone']:
                users[user_id]['time_zone'] = 'Unknown'

            users[user_id]['followers_count'] = self.replaceNumeric(users[user_id]['followers_count'])
            users[user_id]['statuses_count'] = self.replaceNumeric(users[user_id]['statuses_count'])
            users[user_id]['favourites_count'] = self.replaceNumeric(users[user_id]['favourites_count'])
            users[user_id]['friends_count'] = self.replaceNumeric(users[user_id]['friends_count'])
            users[user_id]['listed_count'] = self.replaceNumeric(users[user_id]['listed_count'])

        self.applyNormalization(users, 'followers_count')
        self.applyNormalization(users, 'statuses_count')
        self.applyNormalization(users, 'favourites_count')
        self.applyNormalization(users, 'friends_count')
        self.applyNormalization(users, 'listed_count')

        return users

    # This method applies min-max normalization on a given column of numeric data
    # Input: a data dictionary(users or tweets) and a string representing the numeric data column
    # Output: normalized column
    def applyNormalization(self, data, column):
        # Normalizing column
        scaled = self.minMaxNormalization(self.getColumn(data, column))

        # Replacing column to normalized
        i = 0
        for key in data:
            data[key][column] = scaled[i]
            i += 1

        return data


    # This method normalizes a given column of numeric data
    # Input: array of numeric data
    # Output: a scaled array of numeric data
    def minMaxNormalization(self, data):
        minimum = min(data)
        maximum = max(data)
        scaled = list(map(lambda x: (x - minimum) / (maximum-minimum), data))
        return scaled

    # This method attempts to convert string value to numeric value
    # Input: string representing a number
    # Output: numeric value of the given string, 0 for empty strings
    def replaceNumeric(self, attrib):
        try:
            convert = int(attrib)
        except:
            convert = 0
        return convert

    # This method pulls a column from a given dataset
    # Input: a dictionary of data(users or tweets) and a column name
    # Output: an array representing the column
    def getColumn(self, data, name):
        col = []
        for key in data:
            col.append(data[key][name])
        return col





