from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet2 as DataSet
import time
import os.path
import argparse
from keras.callbacks import Callback
import numpy as np
import sys
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
#进行配置，使用30%的GPU
conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=conf)

# 设置session
KTF.set_session(session )


data_dir = os.environ['DATADIR'] + 'deepnano/'


def train(model, load_to_memory=True, batch_size=None, nb_epoch=100, npoints=40, **kargs):
    # Helper: Save the model.
    if not os.path.isdir(os.path.join(data_dir,'checkpoints', model)):
        os.mkdir(os.path.join(data_dir,'checkpoints/', model))

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(data_dir, 'checkpoints/', model, 'saved_9_25.hdf5'),
        verbose=1,
        save_best_only=True)
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join(data_dir, 'logs', model))
    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)
    # Helper: Save results.
    t=time.localtime(time.time())
    timestamp=str(t.tm_mon)+'-'+str(t.tm_mday)+':'+str(t.tm_hour)+'-'+str(t.tm_min)
    csv_logger = CSVLogger(os.path.join(data_dir, 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    data = DataSet(npoints=npoints, **kargs)
    rm = ResearchModels(model, npoints=npoints)

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train')
        X_val, y_val = data.get_all_sequences_in_memory('val')
    else:
        # Get generators.
        steps_per_epoch = len(data.train) // batch_size
        generator = data.frame_generator(batch_size, 'train')
        val_generator = data.frame_generator(batch_size, 'val')

    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40)

if __name__=='__main__':
    model = sys.argv[1]
    if model == 'primanet':
        one_thickness = True
    elif model == 'alexnet':
        one_thickness = False

    elif model == 'primanet2':
        one_thickness = False
    train(model, one_thickness=one_thickness, load_to_memory=False, batch_size=32)
