import h5py
HDF5_FILENAME = "data/stonks.hdf5"

class H5PY_CONTROLLER:
    def __init__(self, filename=HDF5_FILENAME):
        self.filename = filename

    def get_dataset(self, dataset_name):
        file = h5py.File(self.filename, 'r')
        try:
            dataset = file[dataset_name]
            array = dataset[:]
            file.close()
            return array
        except KeyError:
            file.close()
            return None

    def set_dataset(self, dataset_name, vals):
        file = h5py.File(self.filename, 'a')
        try:
            del file[dataset_name]
        except KeyError:
            pass
        file.create_dataset(dataset_name, data=vals, dtype='float32')
        file.close()

    def get_dataset_at_i(self, dataset_name, i):
        file = h5py.File(self.filename, 'r')
        try:
            dataset = file[dataset_name]
            val = dataset[i]
            file.close()
            return val
        except KeyError:
            file.close()
            return None

    def set_dataset_at_i(self, dataset_name, i, val):
        file = h5py.File(self.filename, 'a')
        try:
            file[dataset_name][i] = val
            file.close()
            return True
        except KeyError:
            file.close()
            return False

