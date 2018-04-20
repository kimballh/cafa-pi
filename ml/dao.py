import h5py
import numpy as np
from .embeddings import seq_from_matrix

class HDF5TargetDao(object):
    def __init__(self, h5_path):
        f = h5py.File(h5_path, "r")
        self.data = f['embeddings/sequence']
        self.n = self.data.shape[0]

    def get_data_chunked(self, size=25):
        for i in range(0, self.n, size):
            # If we're going to overflow, then take a few steps back.
            # Maybe necessary because tf takes in fixed batch sizes.
            if i + size > self.n:
                i = self.n - size
            yield self.data[i:i+size, :, :]

class HDF5Dao(object):
    def __init__(self, h5_path, label_type="binary/motility", pct_test=0.1):
        f = h5py.File(h5_path, "r")
        self.data = f['embeddings/sequence']
        self.n = self.data.shape[0]
        self.labels = f['labels/{}'.format(label_type)]
        self.train_test_split(pct_test)
        self.__n_train_retrieved = 0

    def train_test_split(self, pct_test):
        n_test = int(pct_test * self.n)
        indices = np.array(range(self.n))
        np.random.shuffle(indices)
        self.test_indices = indices[:n_test]
        self.train_indices = indices[n_test:]

    def __batch(self, n, test_or_train):
        if test_or_train == "test":
            ixs = self.test_indices
        else:
            ixs = self.train_indices
        batch_ixs = sorted(np.random.choice(ixs, size=n, replace=False))
        x, y = self.data[batch_ixs,:,:], self.labels[batch_ixs]
        return x, y

    def get_batch_train(self, size=25):
        self.__n_train_retrieved += size
        return self.__batch(size, "train")

    def get_batch_test(self, size=25):
        return self.__batch(size, "test")

    @property
    def epochs(self):
        return self.__n_train_retrieved / self.n

if __name__ == "__main__":
    h5_path = "./data/parsed/cafa3/train.h5"
    target_dao = HDF5TargetDao(h5_path)
    # Prime chunk size 17 because maybe it'll help test the whole
    # overflow avoidance thing.
    chunk_size = 17
    for chunk in target_dao.get_data_chunked(size=chunk_size):
        assert (len(chunk) == chunk_size,
            "Chunks are the wrong size: {} != {}".format(
                len(chunk), chunk_size
            ))
    print(seq_from_matrix(chunk[-1]))
        # for mtx in chunk:
        #     print(seq_from_matrix(mtx))
    # dao = HDF5Dao(h5_path, label_type="multi_hot")
    # for _ in range(100):
    #     x, y = dao.get_batch_train(100)
    # print(dao.epochs)