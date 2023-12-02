import pandas as pd
import shared_vars as sv
from datetime import datetime
import numpy as np

def get_csv_data(path):
    data = np.genfromtxt(path, delimiter=',')
    return data

def load_data_sets(start: datetime, finish: datetime):
    d = get_csv_data(f'_crypto_data/{sv.settings.coin}/{sv.settings.coin}_{sv.settings.time}m.csv')

    filtered_data = d[(d[:, 0] / 1000 >= start.timestamp()) & (d[:, 0] / 1000 <= finish.timestamp())]

    data = filtered_data[np.argsort(filtered_data[:, 0])]

    print(f'Data {sv.settings.coin} 1m downloaded successfuly')
    sv.data = data




