from common.imports import *
from common.const import *
from data.history import HISTORY_DATA

class ForecasterDataset(Dataset):
    def __init__(self, start_index, end_index):
        super().__init__()
        self.start_index = start_index
        self.end_index = end_index
        self.history_data = HISTORY_DATA()

    def __len__(self):
        return len(self.history_data.tickers_idx)

    def __getitem__(self, index):
        history = self.history_data.history_array_from_index(index)
        end_index =  self.end_index if self.end_index else history.shape[0]
        history = history[self.start_index:end_index, :D_MODEL]
        return torch.tensor(history, dtype=torch.float32)



# standardized_data = (data - mean(data)) / std_dev(data)

# To reverse standardization, use the inverse transformation:

# original_data = standardized_data * std_dev(data) + mean(data)

def create_splits(
        batch_size, test_size, num_workers):

    train_loader = DataLoader(
        dataset=ForecasterDataset(0, -test_size),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        dataset=ForecasterDataset(0, None),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers
    )

    return train_loader, test_loader


def collate(data: list[torch.Tensor]):
    """recieves list of dataset.__getitem__() outputs\n
    returns batch collated to max_len, tensor of lengths for each item in batch
    """
    collated = pad_sequence(data, batch_first=True)
    lengths = torch.tensor([array.shape[0] for array in data])
    return collated, lengths

def get_tgt_mask(size) -> torch.Tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
    return mask
