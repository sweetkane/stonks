from common.imports import *

def standardize(batch: torch.Tensor, lengths: torch.Tensor):
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
        return batch, (mean_price, mean_vol, std_dev_price, std_dev_vol)

def unstandardize(data: torch.Tensor, mean_price, mean_vol, std_dev_price, std_dev_vol):
    data[:,:,:4] = data[:,:,:4] * std_dev_price + mean_price
    data[:,:,4] = data[:,:,4] * std_dev_vol + mean_vol
    return data
