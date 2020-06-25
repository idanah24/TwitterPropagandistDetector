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



# This class is the deployment of the classification model
class Predictor:

    # Class constructor loads Doc2Vec model, NN model, min-max values for processing, and threshold for predicting
    def __init__(self, classifier):

        # Determining type of classification
        self.classifier = classifier

        # Setting paths for models
        path = pathlib.Path(os.getcwd()).parent / 'Engine'

        if self.classifier == 'Tweets':
            self.NN_MODEL_PATH = str(path / 'TC_model.h5')
            self.TEXT_MODEL_PATH = str(path / 'TC_text_model')
        elif self.classifier == 'Accounts':
            self.NN_MODEL_PATH = str(path / 'AC_model.h5')
            self.TEXT_MODEL_PATH = str(path / 'AC_new_text_model')

        # Setting class variables
        self.user_info, self.tweet_info = None, None
        self.threshold = 0.5

        # Loading min-max values for user metadata
        with open(path / 'min-max-values.json', 'r') as file:
            self.min_max_values = json.load(file)

        # Loading classification models
        print("[INFO] Loading prediction models...")
        self.nn_model = load_model(self.NN_MODEL_PATH)
        self.text_model = Doc2Vec.load(self.TEXT_MODEL_PATH)
        print("[INFO] Done!")


    # This method takes in user and tweet info as dictionaries and converts to dataframe class variable
    def setInput(self, user_info, tweet_info):
        self.user_info = pd.DataFrame(user_info)
        self.tweet_info = pd.DataFrame(tweet_info)


    # This method performs processing to data before class prediction
    # calls text processing and user meta processing separately
    # must feed input before calling this method
    def process(self):
        self.processTextData()
        self.processUserData()

    # This method makes prediction on a pre-given input
    # must feed data and process before calling this method
    def predict(self):
        print("[INFO] Making predictions...")

        # Activating model, returns numpy array where each value is between 0-1
        predictions = self.nn_model.predict([self.user_info, self.tweet_info])

        # Classifying according to threshold
        classes = np.where(predictions <= self.threshold, 0, 1)

        # Rounding predictions for a more friendly view
        predictions = np.round(predictions, 3)

        # Determining final prediction
        prop = np.count_nonzero(classes == 1)
        non_prop = len(classes) - prop
        final = 'Propaganda' if prop > non_prop else 'Not Propaganda'

        print("[INFO] Done!")
        return [predictions, self.threshold, classes, final]


    # This method prepares user meta-data for classification
    def processUserData(self):
        print("[INFO] Processing user meta data...")
        # Helper function to calculate scaled feature values
        def min_max_normalize(feature, feature_name):
            return (feature - self.min_max_values[feature_name][0]) / (self.min_max_values[feature_name][1] - self.min_max_values[feature_name][0])

        # Taking out relevant features
        columns = ['followers_count', 'statuses_count', 'favourites_count', 'friends_count', 'listed_count']
        self.user_info = self.user_info[columns]

        # Min-Max normalizing features
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

        if self.classifier == 'Tweets':
            # Replicating user meta data vector to be the same amount as the tweets
            self.user_info = pd.concat([self.user_info] * len(self.tweet_info))

        # Converting to numpy array
        self.user_info = self.user_info.to_numpy()
        print("[INFO] Done!")



    # This method processes tweet text data for classification
    def processTextData(self):
        print("[INFO] Processing tweet text data...")
        if self.classifier == 'Accounts':
            self.tweet_info['text'] = self.tweet_info['text'].apply(lambda x: "%s" % ' '.join(x))
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

        # Getting vectors for each tweet from the Doc2Vec model
        print("[INFO] Activating text model...")
        corpus = corpus.map(lambda x: self.text_model.infer_vector(x))
        print("[INFO] Vectors generated!")
        self.tweet_info['text'] = corpus

        # Reforming text vectors shape to match model's LSTM layer
        self.tweet_info = self.tweet_info['text'].to_numpy()
        vectors = []
        for vec in self.tweet_info:
            vectors.append(vec)
        self.tweet_info = np.array(vectors)
        self.tweet_info = np.reshape(self.tweet_info, newshape=(self.tweet_info.shape[0],
                                                                      self.tweet_info.shape[1], -1))

        print("[INFO] Text processing complete!")



