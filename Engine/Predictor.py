import pandas as pd
import numpy as np
import os
import json
import pathlib
from keras.models import load_model
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize





class Predictor:

    def __init__(self):
        self.user_info, self.tweet_info = None, None

        path = pathlib.Path(os.getcwd()).parent / 'Engine'
        self.NN_MODEL_PATH = str(path / 'model.h5')
        self.TEXT_MODEL_PATH = str(path / 'new_text_model')

        self.threshold = 0.5

        with open(path / 'min-max-values.json', 'r') as file:
            self.min_max_values = json.load(file)


        print("[INFO] Loading prediction models...")
        self.nn_model = load_model(self.NN_MODEL_PATH)
        self.text_model = Doc2Vec.load(self.TEXT_MODEL_PATH)
        print("[INFO] Done!")

    def setInput(self, user_info, tweet_info):
        self.user_info = pd.DataFrame(user_info)
        self.tweet_info = pd.DataFrame(tweet_info)

    def process(self):
        self.processTextData()
        self.processUserData()


    def predict(self):
        print("[INFO] Making predictions...")
        predictions = self.nn_model.predict([self.user_info, self.tweet_info])
        classes = np.where(predictions <= self.threshold, 0, 1)
        print("[INFO] Done!")
        return [predictions, classes]



    def processUserData(self):
        def min_max_normalize(feature, feature_name):
            return (feature - self.min_max_values[feature_name][0]) / (self.min_max_values[feature_name][1] - self.min_max_values[feature_name][0])
        columns = ['followers_count', 'statuses_count', 'favourites_count', 'friends_count', 'listed_count']

        self.user_info = self.user_info[columns]
        self.user_info['followers_count'] = self.user_info['followers_count'].map(
            lambda x: min_max_normalize(x, 'followers_count'))

        self.user_info['statuses_count'] = self.user_info['statuses_count'].map(
            lambda x: min_max_normalize(x, 'statuses_count'))

        self.user_info['favourites_count'] = self.user_info['favourites_count'].map(
            lambda x: min_max_normalize(x, 'favourites_count'))

        self.user_info['friends_count'] = self.user_info['friends_count'].map(
            lambda x: min_max_normalize(x, 'friends_count'))

        self.user_info['listed_count'] = self.user_info['listed_count'].map(
            lambda x: min_max_normalize(x, 'listed_count'))

        self.user_info = pd.concat([self.user_info] * len(self.tweet_info))

        self.user_info = self.user_info.to_numpy()





    def processTextData(self):
        print("[INFO] Start text processing...")
        corpus = self.tweet_info['text']
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

        print("[INFO] Activating text model...")
        corpus = corpus.map(lambda x: self.text_model.infer_vector(x))
        print("[INFO] Done!")
        self.tweet_info['text'] = corpus

        self.tweet_info = self.tweet_info['text'].to_numpy()
        vectors = []
        for vec in self.tweet_info:
            vectors.append(vec)
        self.tweet_info = np.array(vectors)
        self.tweet_info = np.reshape(self.tweet_info, newshape=(self.tweet_info.shape[0],
                                                                      self.tweet_info.shape[1], -1))




