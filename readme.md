#### Scratchpad

##### pretty solid explainer on transformers
https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
http://jalammar.github.io/illustrated-transformer/
https://medium.com/p/69e073d4061e
https://towardsdatascience.com/how-to-run-inference-with-a-pytorch-time-series-transformer-394fd6cbe16c#:~:text=tgt%20is%20another%20input%20required,value%20of%20the%20target%20sequence.

##### potentially useful libs
- datasource: https://github.com/quandl/quandl-python
- lots of info and good docs: https://quantlib-python-docs.readthedocs.io/en/latest/index.html
- pytorch models for timeseries: https://github.com/cure-lab/LTSF-Linear/tree/main
- time series transformer 1D: https://github.com/KasperGroesLudvigsen/influenza_transformer/blob/main/transformer_timeseries.py

##### time series with transformer, lstm, linear
https://github.com/cure-lab/ltsf-linear
- transformer: https://github.com/cure-lab/LTSF-Linear/blob/main/models/Transformer.py
- training: https://github.com/cure-lab/LTSF-Linear/blob/main/exp/exp_main.py

##### gpu profiler
https://gist.github.com/MInner/8968b3b120c95d3f50b8a22a74bf66bc

##### TODO
- convert info to embeddings. create a function in info_data class and call in random.ipynb
- move consts to a common.const file
- add some helpful data analysis and vizualization methods
- d_model should only be 5. I can add the rest in during inference
