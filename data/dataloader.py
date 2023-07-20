from common.common_imports import *
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
        history = self.history_data.history_array_from_index(index)[self.start_index:self.end_index+1]
        return torch.tensor(history, dtype=torch.float32)

class BatchProcessor:
    def __init__(self, max_src_window: int):
        self.max_src_window = max_src_window

    def get_src_len(self, lengths, max_src_window, days_pred):
        batch_size = lengths.shape[0]
        return torch.min(
            torch.full((batch_size,), max_src_window),
            lengths - days_pred
        )

    def get_inputs(self, batch: torch.Tensor, lengths, days_pred: int):
        """returns src, src_kay_padding_mask, tgt, exp"""
        batch_size = batch.shape[0]
        num_feats = batch.shape[2]
        cur_max_src_window = min(self.max_src_window, batch.shape[1])
        src = batch[:, :cur_max_src_window, :]

        # src[:] <= src_window_size && src[:] <= lengths[:] - days_pred - 1 && src[i] >= min_src_window
        src_len = self.get_src_len(lengths, cur_max_src_window, days_pred)

        # create src_padding_mask, tgt, exp using src_len
        src_padding_mask = torch.full((batch_size, cur_max_src_window), True)
        tgt = torch.empty((batch_size, days_pred, num_feats))
        exp = torch.empty((batch_size, days_pred, num_feats))

        # The following is equivalent to this
        # for i in range(batch_size):
        #     src_padding_mask[i, :src_len[i]] = False
        #     tgt[i] = batch[i, src_len[i]-1:src_len[i]-1+days_pred]
        #     exp[i] = batch[i, src_len[i]:src_len[i]+days_pred]

        # Expand dimensions for correct broadcasting
        src_len_expanded = src_len.view(-1, 1)
        # Create a range tensor: 1 x seq_len
        range_tensor = torch.arange(src_padding_mask.size(1)).unsqueeze(0)
        # Update src_padding_mask
        src_padding_mask = range_tensor >= src_len_expanded
        # Create range tensors for correct slicing
        range_tensor_2 = torch.arange(batch.shape[1]).unsqueeze(0).expand(batch_size, -1)
        # Adjust indices for correct slicing
        # start_indices = src_len_expanded - 1
        # end_indices = start_indices + days_pred
        # Create masks for tgt and exp
        tgt_mask = (range_tensor_2 >= src_len_expanded - 1) & (range_tensor_2 < src_len_expanded - 1 + days_pred)
        exp_mask = (range_tensor_2 >= src_len_expanded) & (range_tensor_2 < (src_len_expanded + days_pred))
        # Mask tgt and exp
        tgt = batch[tgt_mask].reshape(tgt.shape)
        exp = batch[exp_mask].reshape(exp.shape)

        return src, src_padding_mask, tgt, exp

    def validate_batch(
            self, batch, lengths, src, src_padding_mask, tgt, exp, days_pred
    ):
        src_len = self.get_src_len(lengths, self.max_src_window, days_pred)

        # ex:
        # batch = 0,1,2,3,4,5,6,7,8,9
        # batch_size = 10
        # src_len = 5
        # src = [0,1,2,3,4],5,6,7,8,9
        # tgt = 0,1,2,3,[4,5,6,7,8],9
        # exp = 0,1,2,3,4,[5,6,7,8,9]
        for i in range(len(batch)):
            assert src_len[i] + days_pred <= len(batch[i])
            assert src_len[i] <= self.max_src_window
            for j in range(len(batch[i])):
                # src (valid src)
                if j < src_len[i]:
                    assert torch.sum(src[i, j]) > 0
                    assert torch.equal(src[i, j], batch[i, j])
                    assert src_padding_mask[i, j] == False
                elif j < src_len[i] + days_pred:
                    # src (invalid src)
                    if j < self.max_src_window:
                        assert src_padding_mask[i, j] == True
                    # tgt // tgt[:,0] == batch[:, src_len-1]
                    assert torch.equal(tgt[i, j-src_len[i]], batch[i, j-1])
                    # exp // exp[:,0] == batch[:, src_len]
                    assert torch.equal(exp[i, j-src_len[i]], batch[i, j])

    def standardize(self, batch: torch.Tensor, lengths: torch.Tensor):
        batch_size = batch.shape[0]
        prices = batch[:,:,:4]
        vols = batch[:,:,4]
        range_tensor = torch.arange(batch.shape[1]).unsqueeze(0).expand(batch_size, -1)
        mask = (range_tensor < lengths.view(-1, 1))
        prices = prices[mask]
        vols = vols[mask]
        mean_price = torch.mean(prices)
        mean_vol = torch.mean(vols)
        std_dev_price = torch.std(prices)
        std_dev_vol = torch.std(vols)

        batch[:,:,:4] = (batch[:,:,:4] - mean_price) / std_dev_price
        batch[:,:,4] = (batch[:,:,4] - mean_vol) / std_dev_vol
        return mean_price, mean_vol, std_dev_price, std_dev_vol

    def unstandardize(self, data: torch.Tensor, mean_price, mean_vol, std_dev_price, std_dev_vol):
        data[:,:,:4] = data[:,:,:4] * std_dev_price + mean_price
        data[:,:,4] = data[:,:,4] * std_dev_vol + mean_vol


# standardized_data = (data - mean(data)) / std_dev(data)

# To reverse standardization, use the inverse transformation:

# original_data = standardized_data * std_dev(data) + mean(data)

def create_splits(
        batch_size, test_size, num_workers):

    train_loader = DataLoader(
        dataset=ForecasterDataset(0, 0-test_size),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        dataset=ForecasterDataset(1-test_size, -1),
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
