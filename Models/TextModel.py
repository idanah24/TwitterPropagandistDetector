from Data.Data import Data
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class TextModel:


    def __init__(self, text=None):
        self.text = text
        self.model = None
        self.MODEL_PATH = 'C:\\Users\\Idan\\PycharmProjects\\TwitterPropagandistDetector\\Models\\text_model'


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

    # This method saves trained model
    def saveModel(self):
        self.model.save(self.MODEL_PATH)

    # This method loads a trained and ready model
    def loadModel(self):
        self.model = Doc2Vec.load(self.MODEL_PATH)


