import shared_vars as sv
from models.signal import Signal
import ML.predict as pred
import ML.prep_data as prep
import numpy as np
import helpers.vizualizer as viz
import ML.prep_data as pred_2
import Indicators.combain as cmb
import Indicators.talibr as ta



def get_signal(i, chunk_len):
    # trg = target[:, 4]
    # signal = pred.get_signal_LSTM(sv.data[i-(chunk_len):i], trg)
    # indic_signal = cmb.get_indiator_signal(sv.data[i-(chunk_len):i])
    data = sv.data[i-(chunk_len):i]
    closes = data[:,4]
    rsi = ta.rsi(closes)
    signal = 0
    # signal = indic_signal
    
    if rsi != 3: #TODO change to != 3
        signal = pred.get_signal_2(sv.data[i-(chunk_len):i])

        incline = closes[-sv.settings.close_strategy.target_len:]
        incline_res = prep.calculate_percent_difference(incline[0], incline[-1])
        if abs(incline_res) < 0.005:
            sv.signal.signal = 3
            return
        else:
            sv.settings.close_strategy.take_profit = abs(incline_res)
            sv.settings.close_strategy.init_stop_loss = abs(incline_res)/3
            # print(abs(incline_res), abs(incline_res)/2)
    else:
        sv.signal.signal = 3
        return
    if signal != rsi:
        sv.signal.signal = 3
        return

    if signal in sv.s:
        sv.signal.signal = signal
        sv.signal.data = 1
    else:
        sv.signal.signal = 3