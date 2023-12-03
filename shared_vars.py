from models.settings import Settings
import numpy as np
from models.signal import Signal


path_to_save_examples = '_classif_train_data/'


telegram_api = 'API_TOKEN_1'

signal: Signal = Signal()
counter = 0

data: np.ndarray = None
i = 0
start_date = None
finish_date = None

settings: Settings = None
model_1 = None
model_2 = None

prep_data = 'A'
category = 0

profit = 0

s = None

max_percent = 0

pos_1m = []

global_pos_1m = []

price_of_open = 0

short_long_stat = {
    'short_plus': 0,
    'short_minus': 0,
    'long_plus': 0,
    'long_minus': 0
}
global_short_long_stat = {
    'short_plus': 0,
    'short_minus': 0,
    'long_plus': 0,
    'long_minus': 0
}

the_worth_cases = {
    'model_1_long': 0,
    'model_1_short': 0,
    'model_2_long': 0,
    'model_2_short': 0,
}
counter = 0

indicators = {
    'rsi': 0,
    'stochastic_slowk': 0,
    'stochastic_slowd': 0,
    'natr': 0,
    'plus_di': 0,
    'minus_di': 0,
    'mfi': 0,
    'sma': 0,
    'macd': 0,
    'macd_signal': 0,
    'obv1': 0,
    'obv2': 0,
    'bolinger': 0,
    'average_true_range': 0,
    'commodity_channel_index': 0,
    'rate_of_change': 0,
}

signals_names_dict = {
    'rsi_plus':0,
    'dem_plus':0,
    'adx_plus':0,
    'rsi_minus':0,
    'dem_minus':0,
    'adx_minus':0,
}
type_os_signal = ''
cand_month = 0
deep_open_try = [0.006, 0.004, 0.002,]
period_open_try = [2, 4, 6]

# target_len = [1, 2, 3, 4, 5]
stop_loss = [0.015, 0.01, 0.006, 0.002]
take_profit = [0.006, 0.01, 0.016]

path_num_patterns_24 = '_classif_train_data_1/patterns_24.csv'
path_num_patterns_234 = '_classif_train_data_1/patterns_234.csv'

path_num_patterns_234_regress = '_classif_train_data_2/patterns_234.csv'

gaf = None
pca = None
scaler = None

array_perc_diff = []