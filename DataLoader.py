import Dataset as D
from tensorflow.keras.utils import Sequence
import random
import numpy as np
import albumentations as A
import cv2


class DataLoader(Sequence):
    def __len__(self):

        return int(np.ceil(len(self.X) / self.batch_size))

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.indices = list(range(len(self.X)))
        self.length = len(self.X)

        self.batch_size = batch_size
        random.shuffle(self.indices)

        self.xrange = 0
        if self.length % batch_size == 0:
            self.xrange = self.length // batch_size
        else:
            self.xrange = self.length // batch_size + 1

        self.batches_indices = [ self.indices[a:]
             if (self.length - a < self.batch_size) else self.indices[a:a + self.batch_size] for a in
            range(0, self.length, self.batch_size)]


        self.res = A.Compose([
            A.Resize(width=256, height=256, p=1),
        ])

    def __getitem__(self, idx):

        self.batch_train = []
        self.batch_test = []

        for i in range(len(self.batches_indices[idx])):
            self.batch_train.append(np.load(self.X[self.batches_indices[idx][i]]))
            self.batch_test.append(np.load(self.y[self.batches_indices[idx][i]]))

        self.data = [self.res(image=i)['image'] for i in self.batch_train]
        self.mask = [self.res(image=i)['image'] for i in self.batch_test]
        self.data = np.array(self.data).astype(np.float32)
        self.mask = np.array(self.mask).astype(np.float32)
        self.data = self.data / 255.
        self.data = self.data.reshape((len(self.data), 256, 256, 1))
        self.mask = self.mask.reshape((len(self.mask), 256, 256, 1))
        return self.data, self.mask


def get_generator(X_train, y_train, batch_size):
    return DataLoader(X_train, y_train, batch_size)
