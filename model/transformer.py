from common.imports import *

def stonks_transformer_model(
    d_model,
    nhead,
    num_encoder_layers,
    num_decoder_layers,
    dropout
):
    return torch.nn.Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        batch_first=True,
    )
