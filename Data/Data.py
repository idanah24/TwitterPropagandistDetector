import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import preprocessing
# -*- coding: utf-8 -*-
# noinspection PyMethodMayBeStatic
import os
import pathlib


class Data:

    def __init__(self):
        self.tweets, self.users = None, None
        path = pathlib.Path(os.getcwd()).parent / 'Data'
        self.RAW_VER_DATA = str(path / 'VerifiedUsers.csv')
        self.RAW_PROP_USERS = str(path / 'prop_users.csv')
        self.RAW_PROP_TWEETS = str(path / 'prop_tweets.csv')
        self.READY_USERS = str(path / 'USERS.csv')
        self.READY_TWEETS = str(path / 'TWEETS.csv')

    # This is the class's main method, processing all data
    def process(self):
        # Gathering propagandist info
        prop_users = self.getPropUsers()

        # Reading verified data
        ver_data = pd.read_csv(self.RAW_VER_DATA)

        # Extracting user information from verified data
        ver_users = self.getVerifiedUsers(ver_data)

        # Extracting tweet information from verified data
        ver_tweets = self.getTweets(data=ver_data)

        # Extracting propaganda tweets
        prop_tweets = self.getTweets()

        # Putting together and processing all users data
        users = self.processUserData(ver_users, prop_users)

        # Putting together and processing all tweet data
        tweets = self.processTweetData(users, prop_tweets, ver_tweets)

        self.users = users
        self.tweets = tweets
        self.users.to_csv(self.READY_USERS)
        self.tweets.to_csv(self.READY_TWEETS)


    # This method extracts propagandist information and performs some pre-filtering
    def getPropUsers(self):
        prop_users = pd.read_csv(self.RAW_PROP_USERS)
        # Adding class label
        prop_users['class'] = 'Propaganda'
        # Dropping unnecessary columns
        prop_users.drop(labels=['verified', 'time_zone', 'created_at'], axis='columns', inplace=True)

        # Removing non-english users (also dropping rows with critical data missing as a result)
        prop_users.drop(prop_users[prop_users['lang'] != 'en'].index, axis='rows', inplace=True)
        prop_users.drop(labels=['lang'], axis='columns', inplace=True)

        # Filling default values for missing data
        prop_users['location'].fillna(value='Unknown', inplace=True)
        prop_users['description'].fillna(value='NO_DESC', inplace=True)

        return prop_users

    # This method extracts verified user's information and performs some pre-filtering
    def getVerifiedUsers(self, data):
        # Pulling user's information from data
        columns = ['user_id', 'location', 'user_key', 'followers_count', 'statuses_count', 'screen_name', 'description',
                   'favourites_count', 'friends_count', 'listed_count', 'lang']
        verified_users = pd.DataFrame(data[columns])

        # Renaming some columns to create uniform set
        verified_users.rename(mapper={'user_id': 'id', 'user_key': 'name'}, axis='columns', inplace=True)

        # Making id column unique
        verified_users.drop_duplicates(subset='id', inplace=True)

        # Removing non-english users (also dropping rows with critical data missing as a result)
        verified_users.drop(verified_users[verified_users['lang'] != 'en'].index, axis='rows', inplace=True)
        verified_users.drop(labels=['lang'], axis='columns', inplace=True)

        # Adding class label
        verified_users['class'] = 'Not Propaganda'

        # Filling missing data with missing values
        verified_users['location'].fillna(value='Unknown', inplace=True)
        verified_users['description'].fillna(value='NO_DESC', inplace=True)

        return verified_users

    # This method performs processing of user's metadata
    def processUserData(self, verified, prop):
        # TODO: consider dropping more verified users
        users = verified.append(prop, ignore_index=True)
        scaler = preprocessing.MinMaxScaler()

        # Min-Max normalization on numerical columns
        # TODO: consider using binning on some columns
        users['followers_count'] = scaler.fit_transform(users['followers_count'].values.reshape(-1, 1))
        users['statuses_count'] = scaler.fit_transform(users['statuses_count'].values.reshape(-1, 1))
        users['favourites_count'] = scaler.fit_transform(users['favourites_count'].values.reshape(-1, 1))
        users['friends_count'] = scaler.fit_transform(users['friends_count'].values.reshape(-1, 1))
        users['listed_count'] = scaler.fit_transform(users['listed_count'].values.reshape(-1, 1))

        # Mapping locations to numerical values
        # TODO: find a way to normalize this column
        users['location'] = users['location'].astype("category").cat.codes

        # TODO: normalize location
        # users['location'] =

        return users

    # This method extracts tweet information
    # if data is not given, reading and returning propaganda tweets dataframe, otherwise returns verified tweets
    def getTweets(self, data=None):
        if data is None:
            data = pd.read_csv(self.RAW_PROP_TWEETS)

        columns = ['user_id', 'user_key', 'created_at', 'retweet_count', 'retweeted', 'favorite_count',
                   'text', 'tweet_id',  'hashtags', 'mentions', 'retweeted_status_id',
                   'in_reply_to_status_id']

        tweets = data[columns]
        return tweets

    # This method performs processing of tweet data
    def processTweetData(self, users, prop_tweets, ver_tweets):

        tweets = ver_tweets.append(prop_tweets, ignore_index=True)

        # Dropping tweets with no user information
        tweets_to_drop = tweets['user_id'].isin(users['id']) | tweets['user_key'].isin(users['name'])
        tweets_to_drop = tweets_to_drop.index[tweets_to_drop == False]
        tweets.drop(tweets_to_drop, axis='rows', inplace=True)

        # Dropping columns with insufficent data (around 41% of data is missing)
        # TODO: consider filling in values with default value
        tweets.drop(['created_at', 'retweet_count', 'retweeted', 'favorite_count'], axis='columns', inplace=True)

        # Dropping columns with almost no data
        tweets.drop(['retweeted_status_id', 'in_reply_to_status_id'], axis='columns', inplace=True)

        # Dropping tweets with no text data
        tweets.dropna(subset=['text'], axis='rows', inplace=True)

        # Filling in some missing values
        tweets['mentions'].fillna(value='[]', inplace=True)

        # TODO: combine all text data before processing
        # text_data = tweets['text'].combine(tweets['hashtags'].combine(tweets['mentions'], lambda x, y: x + y), lambda x, y: x + y)

        tweets['text'] = self.processText(corpus=tweets['text'])

        return tweets

    # This method performs standard NLP on tweet text
    # Input: text corpus
    # Output: processed corpus
    def processText(self, corpus):
        print(corpus)
        print("[INFO] Start text processing...")
        # Low-casing and tokenizing words
        corpus = corpus.map(lambda x: x.lower())
        corpus = corpus.map(lambda x: word_tokenize(x))

        # Removing non-textual content
        corpus = corpus.map(lambda val: list(filter(lambda x: x.isalnum() and x != "https", val)))

        # Stopwords removal
        stop_words = set(stopwords.words('english'))
        corpus = corpus.map(lambda val: list(filter(lambda x: x not in stop_words, val)))

        # Stemming
        ps = PorterStemmer()
        corpus = corpus.map(lambda val: list(map(lambda x: ps.stem(x), val)))

        print("[INFO] Done!")
        return corpus

    # This method loads processed and ready data
    def loadData(self):
        print("[INFO] Loading data...")
        self.tweets = pd.read_csv(self.READY_TWEETS, index_col=[0])
        self.users = pd.read_csv(self.READY_USERS, index_col=[0])
        print("[INFO] Done!")

