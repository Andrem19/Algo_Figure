import talib
import shared_vars as sv
import numpy as np
import talib

def rsi(closes): 

    rsi = talib.RSI(closes)
    sv.indicators['rsi'] = round(rsi[-1]/100, 4)
    if rsi[-1] > 85:
        return 2
    elif rsi[-1] < 15:
        return 1
    else:
        return 3


def stochastic(highs, lows, closes):

    slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
    sv.indicators['stochastic_slowk'] = round(slowk[-1]/100, 4)
    sv.indicators['stochastic_slowd'] = round(slowd[-1]/100, 4)
    if slowk[-1] < 15 and slowd[-1] < 15:
        return 1
    elif slowk[-1] > 85 and slowd[-1] > 85:
        return 2
    else:
        return 3

def support_resistance(highs, lows, closes):
    support = talib.SMA(lows, 20)
    resistance = talib.SMA(highs, 20)
    if closes[-1] < support[-1] and closes[-2] > support[-2]:
        return 2
    elif closes[-1] > resistance[-1] and closes[-2] < resistance[-2]:
        return 1
    elif closes[-1] <= support[-1]:
        return 1
    elif closes[-1] >= resistance[-1]:
        return 2
    else:
        return 3

def bollinger(closes):
    upper_band, middle_band, lower_band = talib.BBANDS(closes, timeperiod=20)

    last_middle_band = middle_band[-1]
    last_upper_band = upper_band[-1]
    last_lower_band = lower_band[-1]
    last_close = closes[-1]

    if last_close > last_upper_band:
        sv.indicators['bolinger'] = 2
        return 2
    elif last_close < last_lower_band:
        sv.indicators['bolinger'] = 1
        return 1
    else:
        return 3

def natr(high, low, close):
    natr = talib.NATR(high, low, close)
    sv.indicators['natr'] = round(natr[-1], 4)

    if natr[-1] > 0.6:
        return 1
    elif natr[-1] < 0.4:
        return 2
    else:
        return 3

def adx(highs, lows, closes):
    adx = talib.ADX(highs, lows, closes)
    plus_di = talib.PLUS_DI(highs, lows, closes)
    minus_di = talib.MINUS_DI(highs, lows, closes)

    last_idx = len(adx) - 1

    sv.indicators['plus_di'] = round(plus_di[last_idx]/100, 4)
    sv.indicators['minus_di'] = round(minus_di[last_idx]/100, 4)

    if plus_di[last_idx] > minus_di[last_idx]:
        return 1
    elif plus_di[last_idx] < minus_di[last_idx]:
        return 2
    elif plus_di[last_idx] < 20 and minus_di[last_idx] < 20:
        return 3
    
def mfi(highs, lows, closes, volumes):
    mfi = talib.MFI(highs, lows, closes, volumes)

    sv.indicators['mfi'] = round(mfi[-1]/100, 4)

    signal = 3
    if mfi[-1] > 70:
        signal = 2
    elif mfi[-1] < 30:
        signal = 1
    else:
        signal = 3

    return signal

def obv(closes, volumes):
    
    obv = talib.OBV(closes, volumes)
    sv.indicators['obv1'] = round(obv[-1]/1000000, 4)
    sv.indicators['obv2'] = round(obv[-2]/1000000, 4)

    if obv[-1] > obv[-2]:
        return 1
    elif obv[-1] < obv[-2]:
        return 2
    else:
        return 3

def sma_ema(closes):
    sma = talib.SMA(closes, 20)
    ema = talib.EMA(closes, 20)
    
    last_sma = sma[-1]
    last_ema = ema[-1]

    sv.indicators['sma'] = round(last_sma/100, 4)

    if last_sma > last_ema:
        return 1
    elif last_sma < last_ema:
        return 2
    else:
        return 3

def macd(closes):
    
    macd, signal, _ = talib.MACD(closes, fastperiod=7, slowperiod=16, signalperiod=5)
    
    last_macd = macd[-1]
    last_signal = signal[-1]

    sv.indicators['macd'] = round(last_macd*100, 4)
    sv.indicators['macd_signal'] = round(last_signal*100, 4)

    if last_macd > last_signal:
        return 1
    elif last_macd < last_signal:
        return 2
    else:
        return 3
    
def trend(closes):
    def chose_arr(start_ind: int, arr: np.ndarray, step: int):
        new_arr = []
        for i in range(start_ind, len(arr), step):
            new_arr.append(arr[i])
        return np.array(new_arr)
    closes = closes[10:]
    new_arr_1 = chose_arr(0, closes, 10)
    new_arr_2 = chose_arr(2, closes, 10)
    new_arr_3 = chose_arr(4, closes, 10)
    new_arr_4 = chose_arr(6, closes, 10)
    if all(np.diff(new_arr_1) > 0) or all(np.diff(new_arr_2) > 0) or all(np.diff(new_arr_3) > 0) or all(np.diff(new_arr_4) > 0):
        return 3  # тренд вверх
    elif all(np.diff(new_arr_1) < 0) or all(np.diff(new_arr_2) < 0) or all(np.diff(new_arr_3) < 0) or all(np.diff(new_arr_4) < 0):
        return 2  # тренд вниз
    else:
        return 3  # нет тренда
    
def average_true_range(highs, lows, closes):
    atr = talib.ATR(highs, lows, closes, timeperiod=14)
    sv.indicators['average_true_range'] = round(atr[-1], 3)
    return atr

def commodity_channel_index(highs, lows, closes):
    cci = talib.CCI(highs, lows, closes, timeperiod=14)
    sv.indicators['commodity_channel_index'] = round(cci[-1]/100, 3)
    return cci

def rate_of_change(closes):
    roc = talib.ROC(closes, timeperiod=10)
    sv.indicators['rate_of_change'] = round(roc[-1], 3)
    return roc

def pivot_points(highs, lows, closes):
    # Расчет Pivot Point
    pivot = (highs[-1] + lows[-1] + closes[-1]) / 3

    # Расчет уровней поддержки и сопротивления
    r1 = 2 * pivot - lows[-1]
    s1 = 2 * pivot - highs[-1]
    r2 = pivot + (highs[-1] - lows[-1])
    s2 = pivot - (highs[-1] - lows[-1])
    sv.indicators['pivot_points'] = round(pivot, 3)
    sv.indicators['r1'] = round(r1, 3)
    sv.indicators['r2'] = round(r2, 3)
    sv.indicators['s1'] = round(s1, 3)
    sv.indicators['s2'] = round(s2, 3)
    return pivot, r1, s1, r2, s2

