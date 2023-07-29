from common.imports import *
from common.util import *
from data.dataloader import *
from model.positonal_encoding import *
from model.standardization import *
from model.batch_processor import *

def train_stonks_transformer(
    model: torch.nn.Transformer,
    learning_rate: float,
    max_norm: float,
    days_pred: int,
    train_loader: DataLoader,
    batch_processor: BatchProcessor,
    device: torch.device,
    pbar = None # can't find type name for this
):
    losses = []

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss().to(device)
    plus_positional_encoding = Summer(PositionalEncoding1D(model.d_model))
    tgt_mask = get_tgt_mask(days_pred).to(device)
    for batch, lengths in train_loader:
        if pbar: pbar.update(1)

        batch, _ = standardize(batch, lengths)
        exp = batch_processor.get_exp(batch, lengths, days_pred).to(device)
        batch = plus_positional_encoding(batch)
        data = batch_processor.get_inputs(batch, lengths, days_pred)
        src, src_padding_mask, tgt = tuple(tensor.to(device) for tensor in data)
        out = model(src=src, src_key_padding_mask=src_padding_mask, tgt=tgt, tgt_mask=tgt_mask)

        loss: torch.Tensor = loss_fn(out, exp)
        if not loss < float("inf"):
            # print("bad loss")
            continue
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        opt.step()

    return np.array(losses)

def test_stonks_transformer(
    model: torch.nn.Transformer,
    days_pred: int,
    test_loader: DataLoader,
    batch_processor: BatchProcessor,
    device: torch.device,
    pbar = None
):
    model.eval()
    loss_fn = torch.nn.MSELoss()
    losses = []

    tgt_mask = get_tgt_mask(days_pred).to(device)
    plus_positional_encoding = Summer(PositionalEncoding1D(model.d_model))

    for batch, lengths in test_loader:
        if pbar: pbar.update(1)

        batch, _ = standardize(batch, lengths)
        exp = batch_processor.get_exp(batch, lengths, days_pred).to(device)
        batch = plus_positional_encoding(batch)
        data = batch_processor.get_inputs(batch, lengths, days_pred)
        src, src_padding_mask, tgt = tuple(tensor.to(device) for tensor in data)
        out = model(src=src, src_key_padding_mask=src_padding_mask, tgt=tgt, tgt_mask=tgt_mask)

        loss: torch.Tensor = loss_fn(out, exp)
        if not loss < float("inf"): continue
        losses.append(loss.item())


    return np.array(losses)

def inference_stonks_transformer(
    model: torch.nn.Transformer,
    days_pred: float,
    src: torch.Tensor,
    device: torch.device
):
    model.eval()

    plus_positional_encoding = Summer(PositionalEncoding1D(model.d_model))
    src = src.to(device)
    src_len = src.shape[1]
    src, std_f = standardize(src, torch.Tensor([[src_len]]))
    #print("src:",src,"\n________________________")
    for _ in range(days_pred):
        src_p = plus_positional_encoding(src)
        tgt = src_p[:,-1:,:]
        #print("tgt:",tgt.shape, "\n", tgt[:,-1,:], "\n____________________________")
        print("src:",src_p.shape, "\n", src_p[:,-5:,:], "\n____________________________")
        out = model(src=src_p, tgt=tgt)
        print("out:", out.shape, "\n", out.clone().detach(), "\n____________________________")
        out = out[:,-1,:].unsqueeze(0)
        src = torch.cat((src, out), dim=1)
    res = src[:,-days_pred:, :5]
    #print(res)
    return unstandardize(res, *std_f).detach().to("cpu")

