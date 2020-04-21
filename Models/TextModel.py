from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from ast import literal_eval
import pandas as pd
import numpy as np

class TextModel:


    def __init__(self, text=None):
        self.text = text
        self.model = None
        self.vectors = None
        self.MODEL_PATH = 'C:\\Users\\Idan\\PycharmProjects\\TwitterPropagandistDetector\\Models\\text_model'
        self.VECTORS_PATH = 'C:\\Users\\Idan\\PycharmProjects\\TwitterPropagandistDetector\\Models\\vectors.npy'


    def buildModel(self):

        print("Building model..")
        # Creating tagged document list for Doc2Vec model
        documents = []
        self.text.apply(lambda row: documents.append(TaggedDocument(words=row['text'], tags=[row['tweet_id']])), axis='columns')

        model = Doc2Vec(vector_size=50, min_count=2, epochs=60)
        model.build_vocab(documents)
        print("Done!")

        print("Training model...")
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        print("Done!")

        self.model = model
        return self

    # This method saves trained model
    def saveModel(self):
        self.model.save(self.MODEL_PATH)
        return self

    # This method loads a trained and ready model
    def loadModel(self):
        self.model = Doc2Vec.load(self.MODEL_PATH)
        return self


    # This method returns all vectors created by the model
    def getVectors(self, load=True):
        if load:
            print("Loading file...")
            self.vectors = np.load(self.VECTORS_PATH)
            print("Done!")


        else:
            self.vectors = []
            self.text.apply(lambda row: self.vectors.append(self.model.infer_vector(literal_eval(row['text']))), axis='columns')
            self.vectors = np.array(self.vectors)
            np.save(self.VECTORS_PATH, self.vectors)

        return self.vectors



