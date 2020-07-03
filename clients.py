import numpy as np
import tensorflow as tf
from dataSets import DataSet
import cv2


class user(object):
    def __init__(self, localData, localLabel, isToPreprocess):
        self.dataset = localData
        self.label = localLabel
        self.train_dataset = None
        self.train_label = None
        self.isToPreprocess = isToPreprocess

        self.dataset_size = localData.shape[0]
        self._index_in_train_epoch = 0
        self.parameters = {}

        self.train_dataset = self.dataset
        self.train_labellabel = self.label
        if self.isToPreprocess == 1:
            self.preprocess()


    def next_batch(self, batchsize):
        start = self._index_in_train_epoch
        self._index_in_train_epoch += batchsize
        if self._index_in_train_epoch > self.dataset_size:
            order = np.arange(self.dataset_size)
            np.random.shuffle(order)
            self.train_dataset = self.dataset[order]
            self.train_label = self.label[order]
            if self.isToPreprocess == 1:
                self.preprocess()
            start = 0
            self._index_in_train_epoch = batchsize
        end = self._index_in_train_epoch
        return self.train_dataset[start:end], self.train_label[start:end]

    def preprocess(self):
        new_images = []
        shape = (24, 24, 3)
        for i in range(self.dataset_size):
            old_image = self.train_dataset[i, :, :, :]
            old_image = np.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
            left = np.random.randint(old_image.shape[0] - shape[0] + 1)
            top = np.random.randint(old_image.shape[1] - shape[1] + 1)
            new_image = old_image[left: left + shape[0], top: top + shape[1], :]

            if np.random.random() < 0.5:
                new_image = cv2.flip(new_image, 1)

            mean = np.mean(new_image)
            std = np.max([np.std(new_image),
                          1.0 / np.sqrt(self.train_dataset.shape[1] * self.train_dataset.shape[2] * self.train_dataset.shape[3])])
            new_image = (new_image - mean) / std

            new_images.append(new_image)

        self.train_dataset = new_images



class clients(object):
    def __init__(self, numOfClients, dataSetName, bLocalBatchSize,
                 eLocalEpoch, sess, train, inputsx, inputsy, is_IID):
        self.num_of_clients = numOfClients
        self.dataset_name = dataSetName
        self.dataset_size = None
        self.test_data = None
        self.test_label = None
        self.B = bLocalBatchSize
        self.E = eLocalEpoch
        self.session = sess
        self.train = train
        self.inputsx = inputsx
        self.inputsy = inputsy
        self.IID = is_IID
        self.clientsSet = {}

        self.dataset_balance_allocation()


    def dataset_balance_allocation(self):
        dataset = DataSet(self.dataset_name, self.IID)
        self.dataset_size = dataset.train_data_size
        self.test_data = dataset.test_data
        self.test_label = dataset.test_label

        localDataSize = self.dataset_size // self.num_of_clients
        shard_size = localDataSize // 2
        shards_id = np.random.permutation(self.dataset_size // shard_size)
        preprocess = 1 if self.dataset_name == 'cifar10' else 0
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = dataset.train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = dataset.train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = dataset.train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = dataset.train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            someone = user(np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2)), preprocess)
            self.clientsSet['client{}'.format(i)] = someone


    def ClientUpdate(self, client, global_vars):
        all_vars = tf.trainable_variables()
        for variable, value in zip(all_vars, global_vars):
            variable.load(value, self.session)

        for i in range(self.E):
            for j in range(self.clientsSet[client].dataset_size // self.B):
                train_data, train_label = self.clientsSet[client].next_batch(self.B)
                self.session.run(self.train, feed_dict={self.inputsx: train_data, self.inputsy: train_label})

        return self.session.run(tf.trainable_variables())
