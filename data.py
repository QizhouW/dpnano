"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from processor import process_image
from keras.utils import to_categorical
import pandas as pd
from scipy.interpolate import interp1d

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():
    def __init__(self, one_thickness=True, npoints=40, xmin=0.3, xmax=1.0, spectra_type=1,input_shape=(250,250)):
        self.data_dir = os.environ['DATADIR'] + 'deepnano/transmissions/'
        self.project_dir = os.environ['PROJECT'] + 'deepnano/'
        self.one_thickness=one_thickness
        self.npoints = npoints
        self.xmin = xmin
        self.xmax = xmax
        self.spectra_type = spectra_type
        self.input_shape=input_shape
        self.data = self.get_data()
        #split train val test
        self.train = self.data[self.data.status == 'train']
        self.test = self.data[self.data.status == 'test']
        self.val = self.data[self.data.status == 'val']

    def get_data(self):
        """Load our data from file."""
        path = os.path.join(self.project_dir, 'clean_data.csv')
        dataset = pd.read_csv(path, index_col=0)
        if self.one_thickness:
            dataset = dataset[dataset.thick_idx==0]
        self.number_of_thick = dataset.thick_idx.unique().__len__()
        return dataset

    def get_geometry(self, item):
        fname = os.path.join(self.data_dir,item.geometry)
        e = np.fromfile(fname, dtype=np.int32).reshape(*self.input_shape)
        if self.one_thickness==True:
            idx = e == 2
            e[idx] = 1
            e = np.expand_dims(e,axis=-1)
        else:
            # encode thickness
            idx = e == 2
            e = np.empty((*self.input_shape,self.number_of_thick))
            e[idx] = self.get_thick_one_hot(item.thick_idx)
        return e

    def get_spectra(self, item):
        fname = os.path.join(self.data_dir, item.spectra)
        s = np.fromfile(fname).reshape(-1,5)
        wl = s[:,0]
        t = s[:,self.spectra_type]
        idx = (wl + 0.1 > self.xmin) * (wl - 0.3 < self.xmax)
        f = interp1d(wl[idx],t[idx])
        wl = np.linspace(self.xmin,self.xmax,self.npoints)
        return f(wl)

    def clean_data(self):
        pass

    def get_thick_one_hot(self,thick_id):
        thick = np.zeros(self.number_of_thick)
        thick[thick_id] = 1.0
        return thick

    def get_all_sequences_in_memory(self, train_val_test):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        if train_val_test == 'train':
            data = self.train
        elif train_val_test == 'val':
            data = self.val
        elif train_val_test == 'test':
            data = self.test

        print("Loading %d samples into memory for %sing." % (len(data), train_val_test))

        X, y = [], []
        for idx, row in data.iterrows():
            geometry = self.get_geometry(row)
            spectra = self.get_spectra(row)
            X.append(geometry)
            y.append(spectra)

        return np.array(X), np.array(y)

    @threadsafe_generator
    def frame_generator(self, batch_size, train_val_test):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset.
        if train_val_test == 'train':
            data = self.train
        elif train_val_test == 'val':
            data = self.val
        elif train_val_test == 'test':
            data = self.test

        print("Creating %s generator with %d samples." % (train_val_test, len(data)))
        while 1:
            X, y = [], []

            # Generate batch_size samples.
            sample = data.sample(batch_size)
            for i,row in sample.iterrows():
                # Reset to be safe.
                sequence = None
                geometry = self.get_geometry(row)
                spectra = self.get_spectra(row)
                X.append(geometry)
                y.append(spectra)

            yield np.array(X), np.array(y)

class DataSet2(DataSet):
    def __init__(self, one_thickness=True, npoints=40, xmin=0.3, xmax=1.0, spectra_type=1,input_shape=(250,250)):
        super(DataSet2, self).__init__(one_thickness=one_thickness, npoints=npoints, xmin=xmin, xmax=xmax, spectra_type=spectra_type,input_shape=input_shape)
        self.data_dir = os.environ['DATADIR'] + 'deepnano/data/'

    def get_data(self, fname='clean_data2.csv'):
        """Load our data from file."""
        path = os.path.join(self.project_dir, fname)
        dataset = pd.read_csv(path, index_col=0)
        self.number_of_thick = dataset.dz.unique().__len__()
        return dataset

    def get_geometry(self, item):
        fname = os.path.join(self.data_dir,item.name + '-mask.bin')
        dimx = int(item.nx)
        dimy = int(item.ny)
        e = np.fromfile(fname, dtype=np.int32).reshape(dimx, dimy)
        mask = np.pad(e,pad_width=[(0,self.input_shape[0]-e.shape[0]),
                                   (0,self.input_shape[0]-e.shape[1])],
                      mode='constant', constant_values=-1)
        mask = np.expand_dims(mask,axis=-1)
        return mask

    def get_spectra(self, item):
        fname = os.path.join(self.data_dir, item.name + '-spectra.bin')
        s = np.fromfile(fname).reshape(-1, 5)
        wl = s[:, 0]
        t = s[:, self.spectra_type]
        idx = (wl + 0.1 > self.xmin) * (wl - 0.3 < self.xmax)
        f = interp1d(wl[idx], t[idx])
        wl = np.linspace(self.xmin, self.xmax, self.npoints)
        return f(wl)