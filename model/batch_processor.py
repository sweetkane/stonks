from common.imports import *

class BatchProcessor:
    def __init__(self, max_src_window: int = 6000):
        self.max_src_window = max_src_window

    def get_src_len(self, lengths, max_src_window, days_pred) -> torch.Tensor:
        batch_size = lengths.shape[0]
        return torch.min(
            torch.full((batch_size,), max_src_window),
            lengths - days_pred
        )

    def get_exp(self, batch: torch.Tensor, lengths, days_pred: int):
        """returns src, src_kay_padding_mask, tgt, exp"""
        batch_size = batch.shape[0]
        num_feats = batch.shape[2]
        cur_max_src_window = min(self.max_src_window, batch.shape[1])

        src_len = self.get_src_len(lengths, cur_max_src_window, days_pred)

        exp = torch.empty((batch_size, days_pred, num_feats))

        src_len_expanded = src_len.view(-1, 1)
        range_tensor_2 = torch.arange(batch.shape[1]).unsqueeze(0).expand(batch_size, -1)
        exp_mask = (range_tensor_2 >= src_len_expanded) & (range_tensor_2 < (src_len_expanded + days_pred))

        exp = batch[exp_mask].reshape(exp.shape)

        return exp

    def get_inputs(self, batch: torch.Tensor, lengths, days_pred: int):
        """returns src, src_kay_padding_mask, tgt, exp"""
        batch_size = batch.shape[0]
        num_feats = batch.shape[2]
        cur_max_src_window = min(self.max_src_window, batch.shape[1])
        src = batch[:, :cur_max_src_window, :]

        src_len = self.get_src_len(lengths, cur_max_src_window, days_pred)

        src_padding_mask = torch.full((batch_size, cur_max_src_window), True)
        tgt = torch.empty((batch_size, days_pred, num_feats))

        src_len_expanded = src_len.view(-1, 1)
        range_tensor = torch.arange(src_padding_mask.size(1)).unsqueeze(0)
        src_padding_mask = range_tensor >= src_len_expanded
        range_tensor_2 = torch.arange(batch.shape[1]).unsqueeze(0).expand(batch_size, -1)
        tgt_mask = (range_tensor_2 >= src_len_expanded - 1) & (range_tensor_2 < src_len_expanded - 1 + days_pred)

        tgt = batch[tgt_mask].reshape(tgt.shape)

        return src, src_padding_mask, tgt

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
