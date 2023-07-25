from common.imports import *
from data.history import HISTORY_DATA

class INFO_DATA:
    def __init__(self):
        self.history_data = HISTORY_DATA()
        self.info_df = pd.read_csv("data/csv/info.csv", na_filter=False)
        self.countries_idx = pd.read_csv("data/csv/countries_idx.csv", na_filter=False)
        self.sectors_idx = pd.read_csv("data/csv/sectors_idx.csv", na_filter=False)
        self.industries_idx = pd.read_csv("data/csv/industries_idx.csv", na_filter=False)


    def info_tuple_from_index(self, i):
        ticker = self.history_data.ticker_itos(i)
        row = self.info_df[self.info_df['Symbol'] == ticker].iloc[0]

        info_tuple = (
            row['Country'],
            row['Sector'],
            row['Industry']
        )
        return info_tuple

    def info_tuple_stoi(self, info_tuple):
        return (
            self.countries_idx.index[self.countries_idx['country'] == info_tuple[0]].tolist()[0],
            self.sectors_idx.index[self.sectors_idx['sector'] == info_tuple[1]].tolist()[0],
            self.industries_idx.index[self.industries_idx['industry'] == info_tuple[2]].tolist()[0]
        )

    def info_array_from_index(self, i) -> np.ndarray:
        info_tuple = self.info_tuple_from_index(i)
        info_tuple_idx = self.info_tuple_stoi(info_tuple)
        return np.array(list(info_tuple_idx), dtype=np.float32)

    # def add_info_columns_numpy(self, data):
    #     info_tuple = self.get_info_tuple()
    #     info_tuple_idx = self.info_tuple_stoi(info_tuple)
    #     info_data = np.empty(shape=(data.shape[0], 3))
    #     info_data[:,0] = info_tuple_idx[0]
    #     info_data[:,1] = info_tuple_idx[1]
    #     info_data[:,2] = info_tuple_idx[2]
    #     data = np.concatenate((data, info_data), axis=1)

