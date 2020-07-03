import numpy as np
import tensorflow as tf
import gzip
import os
import platform
import pickle


class DataSet(object):
    def __init__(self, dataSetName, is_IID, dtype=tf.float32):
        dype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype {}, expected uint8 or float32'.format(dtype))

        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.train_data_size = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'mnist':
            self.mnist_dataset_construct(is_IID, dtype)
        elif self.name == 'cifar10':
            self.cifar10_dataset_construct(is_IID)
        else:
            pass


    def mnist_dataset_construct(self, is_IID, dtype):
        data_dir = r'./data/MNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        if dtype == tf.float32:
            train_images = train_images.astype(np.float32)
            train_images = np.multiply(train_images, 1.0 / 255.0)
            test_images = test_images.astype(np.float32)
            test_images = np.multiply(test_images, 1.0 / 255.0)

        if is_IID == 1:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels


    def cifar10_dataset_construct(self, is_IID):
        images, labels = [], []
        for filename in ['./data/CIFAR-10/cifar-10-batches-py/data_batch_{}'.format(i) for i in range(1, 6)]:
            with open(filename, 'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
            for i in range(len(cifar10[b'labels'])):
                image = np.reshape(cifar10[b'data'][i], (3, 32, 32))
                image = np.transpose(image, (1, 2, 0))
                image = image.astype(float)
                images.append(image)
            labels += cifar10[b'labels']
        images = np.array(images, dtype='float')
        labels = np.array(labels, dtype='int')
        # self.train_data, self.train_label = images, labels
        if is_IID == 1:
            order = np.arange(images.shape[0])
            np.random.shuffle(order)
            self.train_data = images[order]
            self.train_label = dense_to_one_hot(labels[order])
        else:
            order = np.argsort(labels)
            self.train_data = images[order]
            self.train_label = dense_to_one_hot(labels[order])

        images, labels = [], []
        with open(r'./data//CIFAR-10/cifar-10-batches-py/test_batch', 'rb') as fo:
            if 'Windows' in platform.platform():
                cifar10 = pickle.load(fo, encoding='bytes')
            elif 'Linux' in platform.platform():
                cifar10 = pickle.load(fo, encoding='bytes')
        for i in range(len(cifar10[b'labels'])):
            image = np.reshape(cifar10[b'data'][i], (3, 32, 32))
            image = np.transpose(image, (1, 2, 0))
            image = image.astype(float)
            images.append(image)
        labels += cifar10[b'labels']
        images = np.array(images, dtype='float')
        labels = np.array(labels, dtype='int')
        self.test_label = dense_to_one_hot(labels)
        self.test_data = []
        shape = (24, 24, 3)
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            old_image = np.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
            left = int((old_image.shape[0] - shape[0]) / 2)
            top = int((old_image.shape[1] - shape[1]) / 2)
            old_image = old_image[left: left + shape[0], top: top + shape[1], :]

            mean = np.mean(old_image)
            std = np.max([np.std(old_image),
                          1.0 / np.sqrt(images.shape[1] * images.shape[2] * images.shape[3])])
            new_image = (old_image - mean) / std

            self.test_data.append(new_image)

        self.test_data = np.array(self.test_data, dtype='float')
        self.train_data_size = self.train_data.shape[0]
        self.test_data_size = self.test_data.shape[0]


    def next_batch(self, batch_size):
        start = self._index_in_train_epoch
        self._index_in_train_epoch += batch_size
        if self._index_in_train_epoch > self.train_data_size:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = self.train_data[order]
            self.train_label = self.train_label[order]

            start = 0
            self._index_in_train_epoch = batch_size
            assert batch_size <= self.train_data_size
        end = self._index_in_train_epoch
        return self.train_data[start: end], self.train_label[start: end]


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)
