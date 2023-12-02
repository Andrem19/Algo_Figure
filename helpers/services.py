import os
from datetime import datetime
from keras.models import load_model
from joblib import load
import colorama
import ML.train as tr_cl
from colorama import Fore, Style
import statistics
import helpers.profit as pr
from catboost import CatBoostClassifier
import catboost
import shared_vars as sv
from models.position import Position
from models.settings import Settings
import ML.train_regrs as tr_rg

def load_catboost_model(model_path: str):
    # Load the model
    cat_model = catboost.CatBoostClassifier()
    cat_model.load_model(model_path)

    return cat_model

def print_position(open_time: datetime, close_time: datetime, settings: Settings, profit: float, saldo: list, type_close: str, open_price: float, close_price: float): 
    pos = Position(open_time.strftime("%Y-%m-%d %H:%M:%S"), settings.amount)
    pos.finish_time = close_time.strftime("%Y-%m-%d %H:%M:%S")
    pos.duration = close_time - open_time
    pos.profit = profit
    pos.saldo = saldo[-1]
    pos.type_close = type_close
    pos.price_open = open_price
    pos.price_close = close_price



    if type_close == 'antitarget' and profit > 0:
        color = Fore.GREEN
    elif type_close == 'antitarget' and profit < 0:
        color = Fore.RED
    elif type_close == 'timefinish' and profit > 0:
        color = Fore.CYAN
    elif type_close == 'timefinish' and profit < 0:
        color = Fore.YELLOW
    elif type_close == 'target' and profit > 0:
        color = Fore.MAGENTA
    else:
        color = Fore.BLUE
    colorama.init()
    print(color + str(pos) + Style.RESET_ALL)
    colorama.deinit()

def short_long_statistic(profit: float):
    if sv.signal.signal == 1:
        if profit > 0:
            sv.short_long_stat['long_plus'] +=1
        else:
            sv.short_long_stat['long_minus'] +=1
    elif sv.signal.signal == 2:
        if profit > 0:
            sv.short_long_stat['short_plus'] +=1
        else:
            sv.short_long_stat['short_minus'] +=1

def the_worth_cases():
    if sv.signal.signal == 1 and sv.signal.data == 1:
        sv.the_worth_cases['model_1_long']+=1
    elif sv.signal.signal == 2 and sv.signal.data == 1:
        sv.the_worth_cases['model_1_short']+=1
    elif sv.signal.signal == 1 and sv.signal.data == 2:
        sv.the_worth_cases['model_2_long']+=1
    elif sv.signal.signal == 2 and sv.signal.data == 2:
        sv.the_worth_cases['model_2_short']+=1

def process_profit(open_time: datetime, profit_list: list, type_close: str, open_price: float, cursor: list, saldos_list: list, price_close: float, type_of_event: list):
    taker = False
    if type_close == 'timefinish':
        taker = True
    buy = True if sv.signal.signal == 1 else False
    prof = pr.profit_counter(taker, open_price, buy, price_close)
    profit_list.append([open_time.timestamp()*1000, prof])
    close_time = datetime.fromtimestamp(cursor[0]/1000)

    saldo = saldos_list[-1][1] + prof
    saldos_list.append([cursor[0], saldo])

    side = 'Buy' if sv.signal.signal == 1 else 'Sell'
    level_up = (price_close - open_price) / open_price
    pln = '+' if prof > 0 else '-'
    type_of_event.append(f'{side}_{type_close}_{pln}{round(abs(level_up), 3)}')
    short_long_statistic(prof)
    if level_up < -0.006:
        the_worth_cases()
    return prof, type_close, close_time, price_close

def remove_files(directory):
    file_list = os.listdir(directory)
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def remove_one_file(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)

def save_list(my_list, path):
    with open(path, 'a') as file:
        for item in my_list:
            item[1] = round(item[1], 3)
            file.write(f'{item[0]},{item[1]}' + "\n")

def load_saldo_profit(folder_path: str):
    data = []
    
    file_list = os.listdir(folder_path)
    
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line != '':
                    parts = line.strip().split(',')
                    timestamp = float(parts[0])
                    value = float(parts[1])
                    data.append([timestamp, value])
    
    sorted_data = sorted(data, key=lambda x: x[0])
    return sorted_data

def count_strings(lst):
    counts = {}
    for string in lst:
        if string in counts:
            counts[string] += 1
        else:
            counts[string] = 1
    return counts

def calculate_result(saldos: list, type_of_event: list, duration_list: list) -> dict:
    positive_cases = 0 if saldos[0][1] < 0 else 1
    negative_cases = 0 if saldos[0][1] > 0 else 1
    for i in range(1, len(saldos)):
        if saldos[i][1]>saldos[i-1][1]:
            positive_cases+=1
        else:
            negative_cases+=1
            percentage_cases = 0
    if positive_cases == 0:
        percentage_cases = 0
    percentage_cases = (positive_cases / (positive_cases + negative_cases)) * 100
    result = {
        'positive_cases': positive_cases,
        'negative_cases': negative_cases,
        'duration_avarage': round(sum(duration_list) / len(duration_list), 4) if len(duration_list) > 0 else 0,
        'duration_median': round(statistics.median(duration_list), 4) if len(duration_list) > 0 else 0,
        'all_cases': len(saldos),
        'percentage_cases': round(percentage_cases, 3),
        'saldo': saldos[-1][1],
        'events': count_strings(type_of_event),
    }
    return result

def train_models():
    # sv.pca = load('pca.joblib')
    sv.gaf = load('gaf.joblib')
    sv.scaler = load('scaler.joblib')
    # sv.model_1 = tr_cl.load_model('_models/my_model_2.keras')
    if sv.settings.prep_data == 'B':
        print('train_models()')
        # sv.scaler = load('scaler.joblib')

        # sv.model_long_1 = tr_cl.train_model_CatBoost(sv.path_to_save_model_long_1, 20)
        # sv.model_short_1 = tr_cl.train_model_CatBoost(sv.path_to_save_model_short_1, 20)
        # sv.model_long_2 = tr_cl.train_model_CatBoost(sv.path_to_save_model_long_2, 30)
        # sv.model_short_2 = tr_cl.train_model_CatBoost(sv.path_to_save_model_short_2, 30)

        # sv.model_long_1 = tr_cl.train_model_XGBoost(sv.path_to_save_model_long_1)
        # sv.model_short_1 = tr_cl.train_model_XGBoost(sv.path_to_save_model_short_1)
        # sv.model_long_2 = tr_cl.train_model_XGBoost(sv.path_to_save_model_long_2)
        # sv.model_short_2 = tr_cl.train_model_XGBoost(sv.path_to_save_model_short_2)

        # sv.model_2 = tr_cl.load_model('_models/my_model.keras')
        # sv.model_1 = tr_cl.load_model('_models/my_model_3.keras')
        # sv.model_1 = tr_cl.train_2Dpic_model_2(3, "_classif_train_data/")
        # sv.model_1 = tr_cl.train_1D_model(3, "_classif_train_data_1/patterns_234.csv")
        # sv.model_1 = tr_cl.train_GAF_model(3, "_classif_train_data_1/patterns_234.csv")
        # sv.model_1 = tr_rg.train_LSTM_Regression(sv.path_num_patterns_234_regress)

        sv.model_1 = tr_cl.train_model_CatBoost("_classif_train_data_1/patterns_234.csv")

        # sv.model_1 = tr_cl.train_LSTM_classif("_classif_train_data_1/patterns_234.csv")

        

        # sv.model_long_1 = load_model('_models/long_model_1.keras')
        # sv.model_short_1 = load_model('_models/short_model_1.keras')
        # sv.model_long_2 = load_model('_models/long_model_2.keras')
        # sv.model_short_2 = load_model('_models/short_model_2.keras')

        # sv.model_long_1 = load_catboost_model('_models/long_model_1.keras')
        # sv.model_short_1 = load_catboost_model('_models/short_model_1.keras')
        # sv.model_long_2 = load_catboost_model('_models/long_model_2.keras')

        # sv.model_1 = load_catboost_model('_models/my_model_3.keras')

# def find_candle_index(timestamp, candles):
#     for i, candle in enumerate(candles):
#         if candle[0] == timestamp:
#             return i
#         elif candle[0] > timestamp:
#             return -1
#     return -1
def find_candle_index(timestamp, candles):
    start = 0
    end = len(candles) - 1
    
    while start <= end:
        mid = (start + end) // 2
        if candles[mid][0] == timestamp:
            return mid
        elif candles[mid][0] < timestamp:
            start = mid + 1
        else:
            end = mid - 1
    
    return -1

def get_points_value(saldos_list_len: int):
    points = 10
    if saldos_list_len < 1000 and saldos_list_len > 200:
        points = 50
    elif saldos_list_len > 1000 and saldos_list_len < 5000:
        points = 250
    elif saldos_list_len > 5000 and saldos_list_len < 15000:
        points = 500
    elif saldos_list_len > 15000 and saldos_list_len < 25000:
        points = 1000
    elif saldos_list_len > 25000 and saldos_list_len < 40000:
        points = 2000
    elif saldos_list_len > 40000:
        points = 5000
    return points

def get_profit_percent(profit: float):
    profit = profit / sv.settings.amount * 100
    return profit

def create_candle_dict(candles):
    candle_dict = {}
    for i, candle in enumerate(candles):
        candle_dict[candle[0]] = i
    return candle_dict

def get_candel_index(timestamp):
    if timestamp in sv.candel_dict:
        candle_index = sv.candel_dict[timestamp]
        return candle_index
    else: return -1

def delete_folder_contents(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)
    
    # print(f"Deleted files: {path}")

def reverse():
    if sv.signal.signal == 2:
        sv.signal.signal = 1 
    elif sv.signal.signal == 1:
        sv.signal.signal = 2

