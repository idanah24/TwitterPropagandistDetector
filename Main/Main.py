import sys
import pathlib
import os

# This part sets project modules to system path

def setSystemPaths():
    main_path = str(pathlib.Path(os.getcwd()))
    data_path = str(pathlib.Path(os.getcwd()).parent / 'Data')
    models_path = str(pathlib.Path(os.getcwd()).parent / 'Models')
    experiments_path = str(pathlib.Path(os.getcwd()).parent / 'Experiments')
    engine_path = str(pathlib.Path(os.getcwd()).parent / 'Engine')
    # Check if necessary
    # keras_path = str(pathlib.Path(os.getcwd()).parent / 'Experiments')
    # sys.path.append('/home/sce-twitter/TwitterPropagandistDetector/venv/Lib/site-packages/keras')
    paths = [main_path, data_path, models_path, experiments_path, engine_path]
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)


setSystemPaths()

from Engine.Predictor import Predictor
from Data.Data import Data
from Models.NeuralNet import NeuralNet
from Models.TextModel import TextModel
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval
import pandas as pd

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

def prepareData(data, text_input='Doc2Vec'):

    # Preparing text vectors


    # Merging by id
    merged_by_id = pd.merge(left=dt.tweets, right=dt.users, left_on='user_id', right_on='id')

    # Merging missing user id's tweets by user name
    missing_id = dt.tweets[~dt.tweets['user_id'].isin(dt.users['id'])]
    merged_by_name = pd.merge(left=missing_id, left_on='user_key', right=dt.users, right_on='name', left_index=True). \
        drop_duplicates(subset=['tweet_id'])

    # Merging all data

    merged = merged_by_id.append(merged_by_name).sort_index()

    # Dropping prop data to make even number of samples
    # merged.drop(index=merged[merged['class'] == 'Propaganda'].sample(n=73125).index, inplace=True)

    # Get text vectors

    if text_input == 'Doc2Vec':
        tm = TextModel(text=merged).buildModel().saveModel()
        text_vectors = tm.getVectors(generate=True, save=True)

    elif text_input == 'Tf-Idf':
        vectorizer = TfidfVectorizer(lowercase=False, tokenizer=lambda x: x)
        merged['text'] = merged['text'].map(lambda x: literal_eval(x))
        text_vectors = vectorizer.fit_transform(merged['text'])

    # Taking out relevant features
    selected_columns = ['followers_count', 'statuses_count', 'favourites_count', 'friends_count', 'listed_count', 'class']
    merged = merged[selected_columns]

    # Preparing target vector
    target_vector = merged['class'].map(lambda x: 1 if x == 'Propaganda' else 0).to_numpy()

    # Preparing user vectors
    merged.drop(labels=['class'], axis='columns', inplace=True)
    user_vectors = merged.to_numpy()

    return [user_vectors, text_vectors, target_vector]


# dt = Data()
# dt.loadData()



# user_vectors, tweet_vectors, target_vector = prepareData(dt)



# Creating model
# network = NeuralNet(user_vectors, tweet_vectors, target_vector)

# history = network.train()
# predictions = network.test()
# network.outputResults(history, predictions)



# Creating data to test predictor

data = pd.read_csv('C:\\Users\\Idan\\PycharmProjects\\TwitterPropagandistDetector\\Data\\verified_tweets_2.csv')
data = data[data['user_key'] == 'Alex']
columns = ['user_key', 'created_at', 'location', 'followers_count'
           , 'statuses_count', 'screen_name', 'favourites_count', 'friends_count'
           , 'listed_count', 'description']
user_info = pd.DataFrame(data[columns]).drop_duplicates(subset=['user_key']).to_dict()
columns = ['created_at', 'retweet_count', 'retweeted', 'favorite_count', 'text',
           'tweet_id', 'source', 'hashtags',  'mentions',
             'location', 'followers_count', 'statuses_count', 'lang']

tweets_info = pd.DataFrame(data[columns]).to_dict()



pred = Predictor()
pred.setInput(user_info, tweets_info)
pred.process()


predictions, classes = pred.predict()

