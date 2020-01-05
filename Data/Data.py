import csv
# -*- coding: utf-8 -*-
# noinspection PyMethodMayBeStatic

class Data:

    def __init__(self):
        pass

    # This function reads the propagandist user dataset
    # returns a dictionary of users
    def readUsers(self, path):
        users = dict()
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            flag = False
            for line in reader:
                if flag is not False:
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
                flag = True
        return users

    # This function reads the propaganda tweets dataset
    # returns a dictionary of tweets
    def readTweets(self, path):
        tweets = dict()
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            flag = False
            for line in reader:
                if flag is not False:
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
                flag = True
        return tweets

    # This function searches for a user's posted tweets
    # Input: a user id string, tweet data dictionary
    # Output: a list of tweets (empty if user not found)
    def getTweets(self, userId, tweets):
        userTweets = []
        for key in tweets:
            if tweets[key]['user_id'] == userId:
                userTweets.append(tweets[key])
        return userTweets

    # This function reads the verified tweets dataset
    # returns a dictionary of tweets
    def readVerifiedTweets(self, path):
        tweets = dict()
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            flag = False
            for line in reader:
                if flag is not False:
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
                    tweets[line[8]]['text_date'] = line[16]
                    tweets[line[8]]['location'] = line[17]
                    tweets[line[8]]['followers_count'] = line[18]
                    tweets[line[8]]['statuses_count'] = line[19]
                    tweets[line[8]]['lang'] = line[20]
                    tweets[line[8]]['screen_name'] = line[21]
                    tweets[line[8]]['favourites_count'] = line[22]
                    tweets[line[8]]['friends_count'] = line[23]
                    tweets[line[8]]['listed_count'] = line[24]
                flag = True
        return tweets

    # This function reads the verified user dataset
    # returns a dictionary of users
    def readVerifiedUsers(self, path):
        users = dict()
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            flag = False
            for line in reader:
                if flag is not False:
                    users[line[8]] = dict()
                    users[line[8]]['location'] = line[17]
                    users[line[8]]['name'] = line[1]
                    users[line[8]]['followers_count'] = line[18]
                    users[line[8]]['statuses_count'] = line[19]
                    users[line[8]]['time_zone'] = ""
                    users[line[8]]['verified'] = 'TRUE'
                    users[line[8]]['lang'] = line[20]
                    users[line[8]]['screen_name'] = line[21]
                    users[line[8]]['description'] = line[9]
                    users[line[8]]['created_at'] = line[10]
                    users[line[8]]['favourites_count'] = line[11]
                    users[line[8]]['friends_count'] = line[12]
                    users[line[8]]['listed_count'] = line[13]
                flag = True
        return users