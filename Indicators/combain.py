import numpy as np
import Indicators.talibr as ta
import pandas as pd

def get_indiator_signal(chunk: np.ndarray):
    signals = []
    # volumes = chunk[:, 5]
    closes = chunk[:, 4]
    # highs = chunk[:, 2]
    # lows = chunk[:, 3]


    signals.append(ta.rsi(closes))
    # signals.append(ta.bollinger(closes))
    # signals.append(ta.trend(closes))
    # signals.append(ta.natr(highs, lows, closes))
    # signals.append(ta.sma_ema(closes))
    # signals.append(ta.mfi(highs, lows, closes, volumes))
    # signals.append(ta.adx(highs, lows, closes))
    # signals.append(ta.support_resistance(highs, lows, closes))
    # signals.append(ta.macd(closes))
    # signals.append(ta.obv(closes, volumes))
    # signals.append(ta.stochastic(highs, lows, closes))

    count_1 = signals.count(1)
    count_2 = signals.count(2)
    # count_3 = signals.count(3)
    # count_4 = signals.count(4)
    # count_5 = signals.count(5)

    if count_1 >= 1 and count_2 == 0:
        return 1
    elif count_1 == 0 and count_2 >= 1:
        return 2
    else: return 3