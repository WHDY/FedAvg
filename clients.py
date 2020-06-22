import numpy as np
import tensorflow as tf
from dataSets import DataSet


class user(object):
    def __init__(self, localData, localLabel):
        self.dataset = localData
        self.label = localLabel

        self.dataset_size = localData.shape[0]
        self._index_in_train_epoch = 0
        self.parameters = {}

    def next_batch(self, batchsize):
        start = self._index_in_train_epoch
        self._index_in_train_epoch += batchsize
        if self._index_in_train_epoch > self.dataset_size:
            order = np.arange(self.dataset_size)
            np.random.shuffle(order)
            self.dataset = self.dataset[order]
            self.label = self.label[order]

            start = 0
            self._index_in_train_epoch = batchsize
        end = self._index_in_train_epoch
        return self.dataset[start:end], self.label[start:end]


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
        print(self.IID)
        dataset = DataSet(self.dataset_name, self.IID)
        self.dataset_size = dataset.train_data_size
        self.test_data = dataset.test_data
        self.test_label = dataset.test_label

        localDataSize = self.dataset_size // self.num_of_clients
        shard_size = localDataSize // 2
        shards_id = np.random.permutation(self.dataset_size // shard_size)
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = dataset.train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = dataset.train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = dataset.train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = dataset.train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            someone = user(np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2)))
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
