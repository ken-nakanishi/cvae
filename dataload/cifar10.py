import numpy as np
from sklearn.model_selection import train_test_split
import pickle

def prepare_dataset(debug_test = False, y_onehot = False):

    def unpickle(file):
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
        return data

    trn = [unpickle("./data/cifar-10-batches-py/data_batch_%d" % i) for i in range(1, 6)]
    cifar_X_1 = np.concatenate([d['data'] for d in trn]).astype('float32')
    cifar_y_1 = np.concatenate([d['labels'] for d in trn]).astype('int32')

    tst = unpickle("./data/cifar-10-batches-py/test_batch")
    cifar_X_2 = tst['data'].astype('float32')
    cifar_y_2 = np.array(tst['labels'], dtype='int32')

    cifar_X = np.r_[cifar_X_1, cifar_X_2]
    cifar_y = np.r_[cifar_y_1, cifar_y_2]

    if debug_test:
        cifar_X = cifar_X[:1000]
        cifar_y = cifar_y[:1000]

    cifar_X /= 255.

    if y_onehot:
        cifar_y = np.eye(10)[cifar_y]

    train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y, test_size=0.2, random_state=135)

    data_shape = (1, 3, 32, 32)

    return train_X, test_X, train_y, test_y, data_shape