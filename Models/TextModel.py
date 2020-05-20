from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from ast import literal_eval
import numpy as np
import os
import pathlib

class TextModel:


    def __init__(self, text=None):
        self.text = text
        self.model = None
        self.vectors = None
        path = pathlib.Path(os.getcwd()).parent / 'Models'
        # self.MODEL_PATH = str(path / 'text_model')
        # self.VECTORS_PATH = str(path / 'vectors_reduced.npy')
        self.MODEL_PATH = str(path / 'new_text_model')
        self.VECTORS_PATH = str(path / 'new_doc_vectors.npy')


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
    def getVectors(self, generate=True, save=False):
        if not generate:
            print("[INFO] Loading text vectors...")
            self.vectors = np.load(self.VECTORS_PATH)
            print("[INFO] Done!")


        else:
            print("[INFO] Generating text vectors...")
            self.vectors = []
            self.text.apply(lambda row: self.vectors.append(self.model.infer_vector(literal_eval(row['text']))), axis='columns')
            self.vectors = np.array(self.vectors)
            if save:
                np.save(self.VECTORS_PATH, self.vectors)
            print("[INFO] Done!")

        return self.vectors



