from Data.Data import Data
from Data.Processor import Processor


# Reading user and tweet data
print("Reading user and tweet data...")
dt = Data()
users = dt.readUsers('C:\\Users\\Idan\\PycharmProjects\\TwitterPropagandistDetector\\Data\\prop_users.csv')
tweets = dt.readTweets('C:\\Users\\Idan\\PycharmProjects\\TwitterPropagandistDetector\\Data\\prop_tweets.csv')
print("Done!")

# Processing tweet textual data
pr = Processor()
# print("Processing tweet text...")
# tweets = pr.processTweetText(tweets)
# print("Done!")

# Processing and normalizing the rest of the tweet data
print("Processing tweet data...")
tweets = pr.processTweetMeta(tweets)
print("Done!")

#  Processing user meta data
print("Processing user meta data")
users = pr.processUserMeta(users)
print("Done!")

# print("Printing tweet data:")
# for tweet in tweets:
#     print(tweets[tweet])
# print("Done!")

