import pandas as pd
import csv
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sklearn
import tensorflow
# -*- coding: utf-8 -*-


# This function reads the propagandist user dataset
# returns a dictionary of users



def readUsers():
    users = dict()
    with open('users.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for line in reader:
            users[line[0]] = dict()
            users[line[0]]['location'] = line[1]
            users[line[0]]['name'] = line[2]
            users[line[0]]['followers_count'] = line[3]
            users[line[0]]['statuses_count'] = line[4]
            users[line[0]]['time_zone'] = line[5]
            users[line[0]]['verified'] = line[6]
            users[line[0]]['lang'] = line[7]
            users[line[0]]['screen_name'] = line[8]
            users[line[0]]['description'] = line[9]
            users[line[0]]['created_at'] = line[10]
            users[line[0]]['favourites_count'] = line[11]
            users[line[0]]['friends_count'] = line[12]
            users[line[0]]['listed_count'] = line[13]
    return users

# This function reads the propaganda tweets dataset
# returns a dictionary of tweets
def readTweets():
    tweets = dict()
    with open('tweets.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for line in reader:
            tweets[line[8]] = dict()
            tweets[line[8]]['user_id'] = line[0]
            tweets[line[8]]['user_key'] = line[1]
            tweets[line[8]]['created_at'] = line[2]
            tweets[line[8]]['created_str'] = line[3]
            tweets[line[8]]['retweet_count'] = line[4]
            tweets[line[8]]['retweeted'] = line[5]
            tweets[line[8]]['favorite_count'] = line[6]
            tweets[line[8]]['text'] = line[7]
            tweets[line[8]]['source'] = line[9]
            tweets[line[8]]['hashtags'] = line[10]
            tweets[line[8]]['expanded_urls'] = line[11]
            tweets[line[8]]['posted'] = line[12]
            tweets[line[8]]['mentions'] = line[13]
            tweets[line[8]]['retweeted_status_id'] = line[14]
            tweets[line[8]]['in_reply_to_status_id'] = line[15]
    return tweets

# This function searches for a user's posted tweets
# Input: a user id string, tweet data dictionary
# Output: a list of tweets (empty if user not found)
def getTweets(userId, tweets):
    userTweets = []
    for key in tweets:
        if tweets[key]['user_id'] == userId:
            userTweets.append(tweets[key])
    return userTweets


def processTweetText(tweets):

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

def getRetweetCount(users):
    return users

def processTweetMeta(tweets):

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

def processUserMeta(users):
    minMax = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)

    numeric_data = getRetweetCount(users)
    numeric_data = minMax.fit(numeric_data)

    for user_id in users:

        if users[user_id]['location'] is None:
            users[user_id]['location'] = 'Unknown'

        if users[user_id]['timezone'] is None:
            users[user_id]['timezone'] = 'Unknown'

    return users



# Reading user and tweet data
print("Reading user and tweet data...")
users = readUsers()
tweets = readTweets()
print("Done!")

# Processing tweet textual data
print("Processing tweet text...")
tweets = processTweetText(tweets)
print("Done!")

# Processing and normalizing the rest of the tweet data
print("Processing tweet data...")
tweets = processTweetMeta(tweets)
print("Done!")

print("Printing tweet data:")
for tweet in tweets:
    print(tweets[tweet])
print("Done!")

