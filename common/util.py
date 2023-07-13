from common.common_imports import *

def get_tgt_mask(size) -> torch.tensor:
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

def get_src_len(lengths, min_src_window, max_src_window, days_pred):

    batch_size = lengths.shape[0]
    return torch.min(
        torch.full((batch_size,), max_src_window),
        lengths - days_pred
    )

def get_inputs(batch: torch.Tensor, lengths, max_src_window, days_pred):
    """returns src, src_kay_padding_mask, tgt, exp"""
    batch_size = batch.shape[0]
    num_feats = batch.shape[2]
    max_src_window = min(max_src_window, batch.shape[1])
    min_src_window = 5
    # src[:] <= src_window_size && src[:] <= lengths[:] - days_pred - 1 && src[i] >= min_src_window
    src_len = get_src_len(lengths, min_src_window, max_src_window, days_pred)

    # create src_padding_mask, tgt, exp using src_len
    src_padding_mask = torch.full((batch_size, max_src_window), True)
    tgt = torch.empty((batch_size, days_pred, num_feats))
    exp = torch.empty((batch_size, days_pred, num_feats))
    for i in range(batch_size):
        src_padding_mask[i, :src_len[i]] = False
        tgt[i] = batch[i, src_len[i]-1:src_len[i]-1+days_pred]
        exp[i] = batch[i, src_len[i]:src_len[i]+days_pred]

    # src is just batch
    src = batch[:, :max_src_window, :]
    return src, src_padding_mask, tgt, exp

def validate_batch(
        batch,
        lengths,
        max_src_window,
        days_pred
):
    src_len = get_src_len(lengths, 5, max_src_window, days_pred)

    src, src_padding_mask, tgt, exp = get_inputs(
        batch, lengths, max_src_window, days_pred
    )
    # ex:
    # batch = 0,1,2,3,4,5,6,7,8,9
    # batch_size = 10
    # src_len = 5
    # src = [0,1,2,3,4],5,6,7,8,9
    # tgt = 0,1,2,3,[4,5,6,7,8],9
    # exp = 0,1,2,3,4,[5,6,7,8,9]
    for i in range(len(batch)):
        assert src_len[i] + days_pred <= len(batch[i])
        assert src_len[i] <= max_src_window
        for j in range(len(batch[i])):
            # src (valid src)
            if j < src_len[i]:
                assert sum(src[i, j]) > 0
                assert torch.equal(src[i, j], batch[i, j])
                assert src_padding_mask[i, j] == False
            elif j < src_len[i] + days_pred:
                # src (invalid src)
                if j < max_src_window:
                    assert src_padding_mask[i, j] == True
                # tgt // tgt[:,0] == batch[:, src_len-1]
                assert torch.equal(tgt[i, j-src_len[i]], batch[i, j-1])
                # exp // exp[:,0] == batch[:, src_len]
                assert torch.equal(exp[i, j-src_len[i]], batch[i, j])








def get_grad_mean(model: torch.nn.Module, device):
    means = torch.empty((0,), dtype=float).to(device)
    for param in model.parameters():
        mean = torch.mean(torch.abs(param.grad)).unsqueeze(0)
        means = torch.cat((means, mean))

    return torch.mean(means)

def get_grad_max(model: torch.nn.Module, device):
    maxes = torch.empty((0,), dtype=float).to(device)
    for param in model.parameters():
        max = torch.max(torch.abs(param.grad)).unsqueeze(0)
        maxes = torch.cat((maxes, max))

    return torch.max(maxes)

