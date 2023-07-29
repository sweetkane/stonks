from common.imports import *
from common.const import *

def stonks_transformer_model(
    d_model=D_MODEL,
    nhead=NUM_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dropout=DROPOUT
):
    return torch.nn.Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        batch_first=True,
    )
