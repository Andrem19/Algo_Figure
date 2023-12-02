import numpy as np
import shared_vars as sv

def convert_to_relative(data, col: list):
    return [candel for row in data for candel in convert_one_candle(row[1:], col)]

def convert_one_candle(candle: list, cols: list):
    open_price = candle[0]
    relative_candle = [((price - open_price) / open_price) * 100 for price in candle]
    result = (round(relative_candle[i-1], 3) for i in cols)
    return list(result)

def prep_rel_data(data, cols):
    result_arr = convert_to_relative(data, cols)
    return result_arr