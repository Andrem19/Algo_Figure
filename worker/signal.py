import shared_vars as sv
from models.signal import Signal
import ML.predict as pred
import ML.prep_data as prep
import numpy as np
import helpers.services as serv
import helpers.vizualizer as viz
import ML.prep_data as pred_2
import Indicators.combain as cmb
import Indicators.talibr as ta



def get_signal(i, chunk_len):
    trend = 0
    closes = sv.data[i-(chunk_len):i, 4]
    cl_m_ago_1 = sv.data[-sv.settings.chunk_len*10][4]
    cl_m_ago_2 = sv.data[-int(sv.settings.chunk_len*1.25)][4]
    last_trend = [cl_m_ago_1, cl_m_ago_2, closes[-1]]
    incline_res = prep.calculate_percent_difference(closes[0], closes[-1])
    if all(np.diff(last_trend) > 0):
        trend = 2
        sv.settings.close_strategy.target_len = 320 #320
        sv.settings.rsi_max_border = 84
        sv.settings.rsi_min_border = 16
        sv.settings.timeperiod = 16 #40
    elif all(np.diff(last_trend) < 0):
        trend = 1
        sv.settings.close_strategy.target_len = 320 #320
        sv.settings.rsi_max_border = 85
        sv.settings.rsi_min_border = 15
        sv.settings.timeperiod = 16 #16
    else:
        trend = 1
        sv.settings.close_strategy.target_len = 150 #150
        sv.settings.rsi_max_border = 83 #83
        sv.settings.rsi_min_border = 17 #17
        sv.settings.timeperiod = 16 #16

    close_for_ind = sv.data[i-sv.settings.chunk_len*2:i, 4]
    highs = sv.data[i-sv.settings.chunk_len*2:i, 2]
    lows = sv.data[i-sv.settings.chunk_len*2:i, 3]

    signal = ta.rsi(close_for_ind)
    if signal != 3:
        sv.type_os_signal = 'rsi'
    # if signal == 3:
    #     signal = ta.stochastic(highs, lows, close_for_ind)
    #     if signal != 3:
    #         sv.type_os_signal = 'sto'
    if signal == 3:
        signal = ta.demarker(highs, lows)
        if signal != 3:
            sv.type_os_signal = 'dem'
        if signal == 3:
            adx = ta.adx(highs, lows, close_for_ind, 55) 
            if adx == trend:
                signal = adx 
                if signal != 3:
                    sv.type_os_signal = 'adx'

    if signal in sv.s:
        sv.settings.close_strategy.take_profit = abs(incline_res)
        sv.settings.close_strategy.init_stop_loss = abs(incline_res)*0.333333
        sv.signal.signal = signal
        sv.signal.data = 1
    else:
        sv.signal.signal = 3