import shared_vars as sv
import numpy as np
import Indicators.talibr as ta
import csv
import ML.predict as pred
import helpers.vizualizer as viz
import helpers.relative_1 as rel_1
import helpers.relative_2 as rel_2

def calculate_percent_difference(close, high_or_low):
    return round(((high_or_low - close) / close), 3)

def append_to_csv(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

def chose_arr(start_ind: int, arr: np.ndarray, step: int):
    new_arr = []
    for i in range(start_ind, len(arr), step):
        new_arr.append(arr[i])
    return np.array(new_arr)

def diapason(diff, start, fin):
    if start < 0 and fin != 0:
        if diff < start and diff > fin:
            return True
    elif start > 0 and fin != 0:
        if diff > start and diff < fin:
            return True
    elif fin == 0 and start < 0:
        if diff < start:
            return True
    elif fin == 0 and start > 0:
        if diff > start:
            return True
    return False

def checker(d_1: bool, d_2: bool, d_3: bool) -> bool:
    if d_1 or d_2 or d_3:
        return True
    return False

def map_cases(chunk: np.ndarray, target: np.ndarray):
    close_candels_row = target[:, 4]
    last = chunk[-1][4]
    maximum = target[:, 2].max()
    minimum = target[:, 3].min()
    max_fin = calculate_percent_difference(last, maximum)
    min_fin = calculate_percent_difference(last, minimum)
    close_candels_row = np.insert(close_candels_row, 0, last)
    new_arr_1 = chose_arr(0, close_candels_row, 3)
    new_arr_2 = chose_arr(1, close_candels_row, 3)
    # new_arr_3 = chose_arr(0, close_candels_row, 3)
    incline = chunk[-sv.settings.close_strategy.target_len:, 4]
    incline_res = calculate_percent_difference(incline[0], incline[-1])
    if checker(all(np.diff(new_arr_1) > 0), all(np.diff(new_arr_2) > 0), False) and abs(min_fin) < abs(incline_res):
        # if checker(all(np.diff(new_arr_1) > 0), all(np.diff(new_arr_2) > 0), False) and abs(max_fin) > abs(incline_res):
        # viz.draw_candlesticks(np.concatenate((chunk, target)), f'{1}', sv.settings.chunk_len)
        sv.array_perc_diff.append(max_fin)
        return 1
    elif checker(all(np.diff(new_arr_1) < 0), all(np.diff(new_arr_2) < 0), False)  and abs(min_fin) > abs(incline_res):
        # elif checker(all(np.diff(new_arr_1) < 0), all(np.diff(new_arr_2) < 0), False)  and abs(max_fin) < abs(incline_res):
        # viz.draw_candlesticks(np.concatenate((chunk, target)), f'{2}', sv.settings.chunk_len)
        sv.array_perc_diff.append(min_fin)
        return 2
    else:
        return 0
    
def check_pattern(chunk: np.ndarray):
    closes = chunk[-12:, 4]
    new_arr_1 = chose_arr(0, closes, 3)
    new_arr_2 = chose_arr(1, closes, 3)
    new_arr_3 = chose_arr(2, closes, 3)
    trend_up = checker(all(np.diff(new_arr_1) > 0), all(np.diff(new_arr_2) > 0), all(np.diff(new_arr_3) > 0))
    trend_down = checker(all(np.diff(new_arr_1) < 0), all(np.diff(new_arr_2) < 0), all(np.diff(new_arr_3) < 0))
    if trend_up and not trend_down:
        return 1
    elif trend_down and not trend_up:
        return 2
    else:
        return 0
    
def prep_train_data(chunk: np.ndarray, target: np.ndarray):
    closes = chunk[:,4]
    rsi = ta.rsi(closes)
    if rsi != 3:
        trg = map_cases(chunk, target)
        if trg in (1, 2) and trg == rsi:
            # viz.draw_candlesticks(np.concatenate((chunk, target)), f'{trg}', sv.settings.chunk_len)
            viz.save_candlesticks(chunk, f'{sv.path_to_save_examples}{trg}/{sv.counter}.png')
            sv.i+=5
        elif trg == 0:
            viz.save_candlesticks(chunk, f'{sv.path_to_save_examples}{trg}/{sv.counter}.png')
    sv.counter+=1

def prep_train_data_2(chunk: np.ndarray, target: np.ndarray):
    trg = map_cases(chunk, target)
    print(trg)
    if (trg == 0 and sv.counter%10==0) or trg in (1,2):
        viz.draw_candlesticks(np.concatenate((chunk, target)), f'{trg}', sv.settings.chunk_len)
        try:
            dec = int(input('1-long, 2-short, 0-non:'))
        
            if dec in (0,1,2):
                viz.save_candlesticks(chunk, f'{sv.path_to_save_examples}{dec}/{sv.counter}.png')
            elif dec == 3:
                index = int(input('index from tail:'))
                minus_from_start = sv.settings.close_strategy.target_len - index
                dec_2 = int(input('1-long, 2-short, 0-non:'))
                new_chunk = np.concatenate((chunk[minus_from_start:,:], target[:minus_from_start,:]))

                viz.save_candlesticks(new_chunk, f'{sv.path_to_save_examples}{dec_2}/{sv.counter}.png')
        except ValueError as e:
            print(f'Error: {e}')
        sv.i+=5
    sv.counter+=1

def prep_train_data_3(chunk: np.ndarray, trg: int):
    viz.save_candlesticks(chunk, f'{sv.path_to_save_examples}{trg}/{sv.counter}.png')

def save_example(open_index: int):
    trg = 0
    if sv.signal.signal == 1:
        trg = 0 if sv.profit < 0 else 1
    elif sv.signal.signal == 2:
        trg = 0 if sv.profit < 0 else 2
    prep_train_data_3(sv.data[open_index-(sv.settings.chunk_len*2):open_index], trg)

# def map_cases(chunk: np.ndarray, target: np.ndarray):
#     close_candels_row = target[:, 4]
#     last = chunk[-1][4]
#     maximum = target[:, 2].max()
#     minimum = target[:, 3].min()
#     new_arr_1 = chose_arr(0, close_candels_row, 2)
#     max_fin = calculate_percent_difference(last, maximum)
#     min_fin = calculate_percent_difference(last, minimum)


    # if all(np.diff(new_arr_1) > 0) and max_fin >= sv.settings.expressiveness_targ:
    #     # viz.draw_candlesticks(np.concatenate((chunk, target)), sv.settings.prep_data, sv.settings.chunk_len_1)
    #     return 1
    # elif all(np.diff(new_arr_1) < 0) and min_fin <= -sv.settings.expressiveness_targ:
    #     # viz.draw_candlesticks(np.concatenate((chunk, target)), sv.settings.prep_data, sv.settings.chunk_len_1)
    #     return 2
    # return 0

def prep_train_data_4(chunk: np.ndarray, target: np.ndarray):
    chunk_len = sv.settings.chunk_len
    closes = chunk[:, 4]
    rsi = ta.rsi(closes)
    bol = ta.bollinger(closes)
    if rsi != 3 or bol != 3:
        trg = map_cases(chunk, target)
        cols_1 = [2,4]
        cols_2 = [2,3,4]
        futures_1 = rel_1.prep_rel_data(chunk[-chunk_len:], chunk[-(chunk_len+1)], cols_1)
        futures_1.append(trg)
        append_to_csv(sv.path_num_patterns_24, futures_1)
        futures_2 = rel_2.prep_rel_data(chunk[-chunk_len:], cols_2)
        futures_2.append(trg)
        append_to_csv(sv.path_num_patterns_234, futures_2)

def prep_train_data_5(chunk: np.ndarray, target: np.ndarray):
    closes = chunk[:,4]
    rsi = ta.rsi(closes)
    # print(rsi)
    if rsi != 3:
        highs = chunk[:,2]
        lows = chunk[:,3]
        volumes = chunk[:,5]
        signals = []
        signals.append(ta.rsi(closes))
        signals.append(ta.bollinger(closes))
        signals.append(ta.trend(closes))
        signals.append(ta.natr(highs, lows, closes))
        signals.append(ta.sma_ema(closes))
        signals.append(ta.mfi(highs, lows, closes, volumes))
        signals.append(ta.adx(highs, lows, closes))
        signals.append(ta.support_resistance(highs, lows, closes))
        signals.append(ta.macd(closes))
        signals.append(ta.obv(closes, volumes))
        signals.append(ta.stochastic(highs, lows, closes))
        signals.append(ta.average_true_range(highs, lows, closes))
        signals.append(ta.commodity_channel_index(highs, lows, closes))
        signals.append(ta.rate_of_change(closes))
        features = rel_2.prep_rel_data(chunk, [2,3,4])
        lenth = len(features) //2
        features = features[lenth:]
        for key, value in sv.indicators.items():
                features.append(value)
        trg = map_cases(chunk, target)
        real_trg = 0
        if trg == rsi and trg == 1:
            real_trg = 1 # TODO: change this to 1
        elif trg == rsi and trg == 2:
            real_trg = 2 # TODO: change this to 2
        else: real_trg = 0
        features.append(real_trg)
        append_to_csv(sv.path_num_patterns_234, features)
