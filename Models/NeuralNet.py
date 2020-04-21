import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import concatenate
from keras.utils import plot_model
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import numpy as np



class NeuralNet:

    def __init__(self, user_vectors, tweet_vectors, target_vector):
        self.user_vectors = user_vectors
        self.tweet_vectors = tweet_vectors
        self.target_vector = target_vector

        # Creating mlp and rnn branches
        self.user_model = self.createUserModel()
        self.tweet_model = self.createTweetModel()

        # Concatenating tensor
        combinedInput = concatenate([self.user_model.output, self.tweet_model.output])

        # Adding final layers
        final_output = Dense(4, activation="relu")(combinedInput)
        final_output = Dense(1, activation="linear")(final_output)

        self.model = Model(inputs=[self.user_model.input, self.tweet_model.input], outputs=final_output)

        # plot_model(self.model, to_file='model.png', show_layer_names=True, show_shapes=True,
        #            expand_nested=True)

        # print(self.model.summary())




    # This method defines mlp for the user vectors
    def createUserModel(self):

        model = Sequential()

        model.add(Dense(8, input_shape=(self.user_vectors.shape[1], ), activation='relu'))

        # model.add(Dense(8, input_shape=self.user_vectors.shape, activation='relu'))

        model.add(Dense(4, activation='relu'))

        plot_model(model, to_file='user_model.png', show_layer_names=True, show_shapes=True,
                   expand_nested=True)

        return model

    def createTweetModel(self):
        model = Sequential()

        model.add(Dense(16, activation='relu', input_shape=(self.tweet_vectors.shape[1], 1)))

        model.add(LSTM(8))


        model.add(Dense(4))
        plot_model(model, to_file='text_model.png', show_layer_names=True, show_shapes=True,
                   expand_nested=True)

        return model

    

    def train(self):
        # Splitting train & test data
        X_user_train, X_user_test, X_tweet_train, X_tweet_test, Y_train, Y_test = train_test_split(self.user_vectors,
                                                                                                   self.tweet_vectors,
                                                                                                   self.target_vector,
                                                                                                   test_size=0.2)


        # Reshaping text vectors for LSTM input
        X_tweet_train = np.reshape(X_tweet_train, newshape=(X_tweet_train.shape[0], X_tweet_train.shape[1], -1))
        X_tweet_test = np.reshape(X_tweet_test, newshape=(X_tweet_test.shape[0], X_tweet_test.shape[1], -1))



        # Creating optimizer
        opt = Adam(lr=1e-3, decay=1e-3 / 200)

        # Compiling model
        self.model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

        # Training model
        print("[INFO] training model...")
        history = self.model.fit(
            [X_user_train, X_tweet_train], Y_train,
            validation_data=([X_user_test, X_tweet_test], Y_test),
            epochs=10, batch_size=8)

        print(history)

        # Make predictions on the testing data
        print("[INFO] predicting...")
        predictions = self.model.predict([X_user_test, X_tweet_test])

        print(predictions)



