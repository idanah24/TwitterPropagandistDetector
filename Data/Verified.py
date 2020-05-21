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
                users[line[8]]['description'] = line[25]
                users[line[8]]['created_at'] = line[2]
                users[line[8]]['favourites_count'] = line[22]
                users[line[8]]['friends_count'] = line[23]
                users[line[8]]['listed_count'] = line[24]
            flag = True
    return users