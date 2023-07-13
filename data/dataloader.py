from common.common_imports import *
from data.history import HISTORY_DATA
from data.info import INFO_DATA
from data.h5py import H5PY_CONTROLLER

class ForecasterDataset(Dataset):
    def __init__(self, start_index, end_index):
        super().__init__()
        self.start_index = start_index
        self.end_index = end_index
        self.history_data = HISTORY_DATA()
        self.info_data = INFO_DATA()
        self.h5py = H5PY_CONTROLLER()
        self.len = len(self.history_data.tickers_idx)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.check_progress(index): return None, None
        self.mark_progress(index)

        # get history array according to start/end limits
        if self.end_index:
            history = self.history_data.history_array_from_index(index)[self.start_index:self.end_index]
        else:
            history = self.history_data.history_array_from_index(index)[self.start_index:]

        info = self.info_data.info_array_from_index(index)
        info_padded = np.pad(info, (0, history.shape[1]-info.shape[0]), mode='constant', constant_values=0)

        # return training_data, data_len
        return torch.tensor(np.vstack((info_padded, history)), dtype=torch.float)


    def check_progress(self, index):
        """return true if we have already trained on this index"""
        return self.h5py.get_dataset_at_i("progress", index) == 1

    def mark_progress(self, index):
        self.h5py.set_dataset_at_i("progress", index, 1.)

    def reset_progress(self):
        progress_array = np.zeros((self.len))
        self.h5py.set_dataset("progress", progress_array)

# TODO create custom dataloader that does get_inputs as part of collate_fn

def create_loader(batch_size, start_index=0, end_index=-1, reset_progress=False):
    """end_index should be negative"""
    dataset = ForecasterDataset(start_index, end_index)
    if reset_progress: dataset.reset_progress()
    return DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)

def create_splits(batch_size, test_size, reset_progress=False):
    train_loader = create_loader(
        batch_size=batch_size,
        start_index=0,
        end_index=0-test_size,
        reset_progress=reset_progress)
    test_loader = create_loader(
        batch_size=batch_size,
        start_index=0-test_size,
        end_index=None,
        reset_progress=reset_progress)
    return train_loader, test_loader

def collate_fn(data):
    """recieves list of dataset.__getitem__() outputs\n
       returns batch collated to max_len, tensor of lengths for each item in batch
    """
    batch_size = len(data)
    feature_size = data[0].shape[1]
    lengths = torch.empty((batch_size), dtype=int)
    for i, array in enumerate(data):
        lengths[i] = array.shape[0]
    maxLen = torch.max(lengths).item()
    collated = torch.zeros((batch_size, maxLen, feature_size))
    for i, array in enumerate(data):
        collated[i][:array.shape[0]] = array
    return collated, lengths


