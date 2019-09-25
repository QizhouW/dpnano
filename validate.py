"""
Basically just runs a test generator on
about the same number of videos as we have in our test set.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
from data import DataSet2 as DataSet
import os
from keras.metrics import top_k_categorical_accuracy
import sys
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt

data_dir = os.environ['DATADIR'] + 'deepnano/'

def validate(model, saved_model, npoints=20,**kargs):
    data = DataSet(npoints=npoints, **kargs)
    rm = ResearchModels(model, npoints=npoints, saved_model=saved_model)
    X, y = data.get_all_sequences_in_memory('test')
    eval = rm.model.evaluate(X,y)
    pred = rm.model.predict(X)
    print(eval)
    np.save('prediction',pred)
    np.save('true',y)

if __name__=='__main__':
    model_name, model_file = sys.argv[1:]
    model = model_name
    saved_model = os.path.join(data_dir,'checkpoints/', model, model_file)
    validate(model, saved_model=saved_model)