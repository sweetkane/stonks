from common.imports import *
from data.h5py import H5PY_CONTROLLER

class HISTORY_DATA:
    def __init__(self):
        self.tickers_idx = pd.read_csv("data/csv/tickers_idx.csv")
        self.h5py = H5PY_CONTROLLER()

    def history_array_from_index(self, i) -> np.ndarray:
        symbol = self.ticker_itos(i)
        return self.history_array_from_ticker(symbol)

    def history_array_from_ticker(self, ticker) -> np.ndarray:
        return self.h5py.get_dataset(ticker+"_history")

    def ticker_itos(self, i):
        return self.tickers_idx.iloc[i]["ticker"]

    def ticker_stoi(self, ticker):
        return self.tickers_idx.index[self.tickers_idx['ticker'] == ticker].tolist()[0]

    @staticmethod
    def array_to_dataframe(array):
        columns = [
            'Open',
            'High',
            'Low',
            'Close',
            'Volume'
        ]
        if array.shape[1] == 8:
            columns += ['year','month','day']
        return pd.DataFrame(array, columns=columns)
