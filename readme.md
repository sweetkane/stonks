# 📉 STONKS.0.2 📈

This is a highly informal pet project of mine, which I used to experiment with a ton of concepts such as
- transformers
- pytorch
- cuda
- numpy
- pandas
- hdf5

This project was an abject failure in terms of deliverable haha. 
The goal was to create a stock price predictor that can predict n days in the future, and the result was a model that predicts the same price every day for n days.
The price wasn't even close to the one before it! (see the bottom of [inference.ipynb](https://github.com/sweetkane/stonks.0.2/blob/main/inference.ipynb))

But I learned a lot! 
I spent a lot of time building custom dataloaders that would efficiently clean my fairly large (for a home PC) dataset. 
- This taught me a lot about the standard libraries for numpy, pandas, and pytorch. 
- I learned about what form my data should be in for different tasks, and how to [process](https://github.com/sweetkane/stonks.0.2/blob/main/model/batch_processor.py) it into various forms. 
  
I also learned a lot about transformers during the many hours spent trying to decrease my loss.
- saw firsthand the importance of things like time encoding and normalization.

---
#### My Scratchpad

##### pretty solid explainer on transformers
https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
http://jalammar.github.io/illustrated-transformer/
https://medium.com/p/69e073d4061e
https://towardsdatascience.com/how-to-run-inference-with-a-pytorch-time-series-transformer-394fd6cbe16c#:~:text=tgt%20is%20another%20input%20required,value%20of%20the%20target%20sequence.

##### potentially useful libs
- datasource: https://github.com/quandl/quandl-python
- lots of crap and good docs: https://quantlib-python-docs.readthedocs.io/en/latest/index.html
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
