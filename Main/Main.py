from Data.Data import Data
from Models.NeuralNet import NeuralNet
from Models.TextModel import TextModel
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

def prepareData(data):

    # Preparing text vectors
    tm = TextModel(data.tweets).loadModel()
    text_vectors = tm.getVectors(load=True)

    # Merging by id
    merged_by_id = pd.merge(left=dt.tweets, right=dt.users, left_on='user_id', right_on='id')

    # Merging missing user id's tweets by user name
    missing_id = dt.tweets[~dt.tweets['user_id'].isin(dt.users['id'])]
    merged_by_name = pd.merge(left=missing_id, left_on='user_key', right=dt.users, right_on='name', left_index=True). \
        drop_duplicates(subset=['tweet_id'])

    # Merging all data
    merged = merged_by_id.append(merged_by_name).sort_index()


    # Taking out relevant features
    selected_columns = ['followers_count', 'statuses_count', 'favourites_count', 'friends_count', 'listed_count', 'class']
    merged = merged[selected_columns]

    # Preparing target vector
    target_vector = merged['class'].map(lambda x: 1 if x == 'Propaganda' else 0).to_numpy()

    # Preparing user vectors
    merged.drop(labels=['class'], axis='columns', inplace=True)
    user_vectors = merged.to_numpy()

    return [user_vectors, text_vectors, target_vector]


def calcMemoryUsage(vectors):
    sum = 0
    for vec in vectors:
        sum += vec.itemsize * vec.size

    return sum


dt = Data()
dt.loadData()

user_vectors, tweet_vectors, target_vector = prepareData(dt)


# Creating model
network = NeuralNet(user_vectors, tweet_vectors, target_vector)

network.train()













