from Data.Data import Data
from ast import literal_eval
from Models.TextModel import TextModel
import pandas as pd

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


dt = Data()
dt.loadData()

tm = TextModel(text=dt.tweets).buildModel().saveModel()









