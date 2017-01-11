import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata

def prepare_dataset(debug_test = False, y_onehot = False):

    mnist = fetch_mldata('MNIST original')
    mnist_X, mnist_y = shuffle(mnist.data.astype('float32'), mnist.target.astype('int32'), random_state=42)

    if debug_test:
        mnist_X = mnist_X[:1000]
        mnist_y = mnist_y[:1000]

    mnist_X /= 255.0

    if y_onehot:
        mnist_y = np.eye(10)[mnist_y]

    train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y, test_size=0.2, random_state=135)

    data_shape = (28, 28)

    return train_X, test_X, train_y, test_y, data_shape