from common.common_imports import *
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

    def array_to_dataframe(self, array):
        columns = [
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'year',
            'month',
            'day',
            'month_sin',
            'month_cos',
            'day_sin',
            'day_cos'
            ]
        if array.shape[1] == 15:
            columns += ['country', 'sector', 'industry']

        return pd.DataFrame(array, columns=columns)
