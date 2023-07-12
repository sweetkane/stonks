from common.common_imports import *

class TIME_ENCODER:
    def __init__(self, data, date_col="Date"):
        self.data = data
        self.date_col = date_col

    def separate_dates(self):
        self.data['year'] = self.data[self.date_col].dt.year
        self.data['month'] = self.data[self.date_col].dt.month
        self.data['day'] = self.data[self.date_col].dt.day

    def encode(self):
        self.separate_dates()
        self.encode_column('month',12)
        self.encode_column('day', 31)
        self.data = self.data.drop(columns=[self.date_col])
        return self.data

    def encode_column(self, col, max_val):
        self.data[col + '_sin'] = np.sin(2 * np.pi * self.data[col]/max_val)
        self.data[col + '_cos'] = np.cos(2 * np.pi * self.data[col]/max_val)
        return self.data
