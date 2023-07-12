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

def get_inputs(batch: torch.Tensor, lengths, max_src_window, days_pred):
    """returns src, src_kay_padding_mask, tgt, exp"""
    batch_size = batch.shape[0]
    num_feats = batch.shape[2]
    max_src_window = min(max_src_window, batch.shape[1])
    min_src_window = 5
    # src[:] <= src_window_size && src[:] <= lengths[:] - days_pred - 1 && src[i] >= min_src_window
    src_len = torch.min(
        torch.full_like(lengths, max_src_window),
        torch.max(torch.full_like(lengths, min_src_window), lengths - days_pred - 1)
    )

    # create src_padding_mask, tgt, exp using src_len
    src_padding_mask = torch.full((batch_size, max_src_window), True)
    tgt = torch.empty((batch_size, days_pred, num_feats))
    exp = torch.empty((batch_size, days_pred, num_feats))
    for i in range(batch_size):
        src_padding_mask[i, :src_len[i]] = False
        tgt[i] = batch[i, src_len[i]-1:src_len[i]-1+days_pred]
        exp[i] = batch[i, src_len[i]:src_len[i]+days_pred]
        # validate
        assert sum(batch[i, src_len[i]-1]) > 0
        # for each elt in tgt and exp, sum of elt > 0

    # src is just batch
    src = batch[:, :max_src_window, :]
    return src, src_padding_mask, tgt, exp

def validate_inputs(
        batch,
        lengths,
        max_src_window,
        days_pred,
        src,
        src_padding_mask,
        tgt,
        exp,
):
    pass
