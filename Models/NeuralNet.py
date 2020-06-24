import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers import concatenate
from keras.utils import plot_model
from keras.models import Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import os
import pathlib
import json

from datetime import datetime
import numpy as np



class NeuralNet:

    def __init__(self, user_vectors, tweet_vectors, target_vector):
        self.user_vectors = user_vectors
        self.tweet_vectors = tweet_vectors
        self.target_vector = target_vector

        # Vectors for train & test process
        self.X_user_train, self.X_user_test, self.X_tweet_train, self.X_tweet_test, self.Y_train, self.Y_test = \
            None, None, None, None, None, None

        # Amount per batch and number of epochs for training
        self.batch_size, self.epochs = 100, 8

        # Threshold for classifying
        self.threshold = 0.5

        print("Building network...")
        # Creating mlp and rnn branches
        self.user_model = self.createUserModel()
        self.tweet_model = self.createTweetModel()

        # Concatenating tensor
        combinedInput = concatenate([self.user_model.output, self.tweet_model.output])

        # Adding final layers
        final_output = Dense(4, activation="relu")(combinedInput)
        final_output = Dense(1, activation="sigmoid")(final_output)

        self.model = Model(inputs=[self.user_model.input, self.tweet_model.input], outputs=final_output)

        print("Done!")

    # This method defines mlp for the user vectors
    def createUserModel(self):
        model = Sequential()
        model.add(Dense(100, input_shape=(self.user_vectors.shape[1], ), activation='relu'))
        model.add(Dense(100, activation='relu'))
        return model

    # This method defines rnn for the text vectors
    def createTweetModel(self):
        model = Sequential()
        model.add(Dense(100, activation='relu', input_shape=(self.tweet_vectors.shape[1], 1)))
        model.add(LSTM(100))
        model.add(Dense(100))
        return model

    # This method defines compiling and training the model
    def train(self):

        # Splitting train & test data
        self.X_user_train, self.X_user_test, self.X_tweet_train, self.X_tweet_test, self.Y_train, self.Y_test = \
            train_test_split(self.user_vectors, self.tweet_vectors, self.target_vector, test_size=0.2)

        # Reshaping text vectors for LSTM input
        self.X_tweet_train = np.reshape(self.X_tweet_train, newshape=(self.X_tweet_train.shape[0],
                                                                      self.X_tweet_train.shape[1], -1))
        self.X_tweet_test = np.reshape(self.X_tweet_test, newshape=(self.X_tweet_test.shape[0],
                                                                    self.X_tweet_test.shape[1], -1))


        # Creating optimizer
        opt = Adam(lr=1e-3, decay=1e-3 / 200)

        # Compiling model
        self.model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['binary_accuracy', 'accuracy'])

        # Training model
        print("[INFO] Training model...")
        history = self.model.fit(
            [self.X_user_train, self.X_tweet_train], self.Y_train,
            validation_data=([self.X_user_test, self.X_tweet_test], self.Y_test),
            epochs=self.epochs, batch_size=self.batch_size)

        print("[INFO] Training complete!")

        return history

    # This method defines testing the model
    def test(self):
        # Make predictions on the testing data
        print("[INFO] Testing model...")
        predictions = self.model.predict([self.X_user_test, self.X_tweet_test])
        # Classifying according to threshold
        predictions = np.where(predictions <= self.threshold, 0, 1)
        print("[INFO] Test complete!")
        return predictions

    # This method outputs train & test results
    def outputResults(self, history, predictions):
        print("[INFO] Evaluating results...")
        path = pathlib.Path(os.getcwd()).parent / 'Experiments' / datetime.now().strftime("%d.%m.%Y %H.%M")
        os.mkdir(path)

        # Function that plots given confusion matrix
        def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
            accuracy = np.trace(cm) / float(np.sum(cm))
            misclass = 1 - accuracy

            if cmap is None:
                cmap = plt.get_cmap('Blues')

            plt.figure(figsize=(8, 8))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('Actual')
            plt.xlabel('Predicted\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
            plt.savefig(path / 'confusion_matrix.png')
            plt.clf()

        # Function that plots accuracy and loss over training epochs
        def plot_training_results():

            # Summarize history for accuracy
            fig = plt.figure(figsize=[8, 6])
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig(path / 'model_accuracy.png')
            plt.clf()

            # Summarize history for loss
            fig = plt.figure(figsize=[8, 6])
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig(path / 'model_loss.png')

        # This method defines and calculates the metrics of evaluating for the test results
        def calculateMetrics(cm):
            metrics = dict()
            # Calculating evaluation metrics
            tn, fp, fn, tp = cm.ravel()
            metrics['False Positive Rate'] = round(fp / (fp + tn), 3)
            metrics['False Negative Rate'] = round(fn / (tp + fn), 3)
            metrics['True Negative Rate | Specificity'] = round(tn / (tn + fp), 3)
            metrics['Negative Predictive Value'] = round(tn / (tn + fn), 3)
            metrics['False Discovery Rate'] = round(fp / (tp + fp), 3)
            metrics['True Positive Rate | Recall | Sensitivity'] = round(tp / (tp + fn), 3)
            metrics['Positive Predictive Value | Precision'] = round(tp / (tp + fp), 3)
            metrics['Accuracy'] = round((tp + tn) / (tp + fp + fn + tn), 3)
            metrics['F1-Score'] = round((2 * metrics['Positive Predictive Value | Precision'] *
                                         metrics['True Positive Rate | Recall | Sensitivity'])
                / (metrics['Positive Predictive Value | Precision'] +
                   metrics['True Positive Rate | Recall | Sensitivity']), 3)
            return metrics

        # Plotting model overview
        plot_model(self.model, to_file=path / 'model_overview.png', show_layer_names=True, show_shapes=True,
                   expand_nested=True)

        # Plotting training history
        plot_training_results()

        # Creating and plotting confusion matrix
        cm = confusion_matrix(y_true=self.Y_test, y_pred=predictions)
        plot_confusion_matrix(cm=cm, target_names=['Not Propaganda', 'Propaganda'], normalize=False)

        # Calculating and outputting metrics
        metrics = calculateMetrics(cm)
        with open(path / 'metrics.json', 'w') as file:
            json.dump(metrics, file)

        with open(path / 'info.txt', 'w') as file:
            file.write("Training parameters:")
            file.write('\n')
            file.write("Batch size: {0}".format(self.batch_size))
            file.write('\n')
            file.write("Epochs: {0}".format(self.epochs))
            file.write('\n')
            file.write("Testing parameters:")
            file.write('\n')
            file.write("Threshold: {0}".format(self.threshold))
            file.write('\n')

        print("[INFO] Evaluation complete!")
        self.model.save(path / 'AC_model.h5')
