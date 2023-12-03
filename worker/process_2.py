import shared_vars as sv
import helpers.vizualizer as viz
from datetime import datetime
import helpers.services as serv
import Indicators.talibr as ta

def position_proccess(i: int, data_len: int, profit_list: list, saldos_list: list, type_of_event: list):
    sv.counter +=1
    s_loss = 0
    stop_loss = 0
    close_time = None
    close_price = 0
    take_profit = 0
    price_open = 0


    s_loss = sv.settings.close_strategy.init_stop_loss
    take_profit = sv.settings.close_strategy.take_profit
    target_len = sv.settings.close_strategy.target_len


    open_time = datetime.fromtimestamp(sv.data[i][0]/1000)
    if sv.signal.signal == 1:
        price_open = sv.data[i][1] * (1 - 0.0001)
        stop_loss = (1 - s_loss) * float(price_open)
        take_profit = (1 + take_profit) * float(price_open)

    elif sv.signal.signal == 2:
        price_open = sv.data[i][1] * (1 + 0.0001)
        stop_loss = (1 + s_loss) * float(price_open)
        take_profit = (1 - take_profit) * float(price_open)

    index = i
    for it in range(index, data_len):
        final_it = it
        # sv.settings.rsi_max_border = 85
        # sv.settings.rsi_min_border = 15
        # sv.settings.timeperiod = 16 #16
        # closes = sv.data[it-sv.settings.chunk_len*2:it, 4]
        # highs = sv.data[it-sv.settings.chunk_len*2:it, 4]
        # lows = sv.data[it-sv.settings.chunk_len*2:it, 4]
        # rsi = ta.rsi(closes)
        # # adx = ta.adx(highs, lows, closes, 60) 
        # if (rsi == 1) and sv.signal.signal == 2:# and sv.data[it][0] > price_open:
        #     type_close = 'timefinish'
        #     close = (1 + 0.0005) * sv.data[it][1]
        #     sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
        #     break
        # if (rsi == 2) and sv.signal.signal == 1:# and sv.data[it][0] < price_open:
        #     type_close = 'timefinish'
        #     close = (1 - 0.0005) * sv.data[it][1]
        #     sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
        #     break

        if sv.settings.close_strategy.target_len != 0:
            duration = (datetime.fromtimestamp(sv.data[it][0] / 1000) - open_time).total_seconds() / 60 /sv.settings.time
            if duration >= target_len:
                type_close = 'timefinish'
                close = 0
                if sv.signal.signal == 1:
                    if sv.data[it][1] > stop_loss:
                        close = (1 - 0.0005) * sv.data[it][1] if sv.data[it][1] < take_profit else take_profit
                    else:
                        close = stop_loss
                elif sv.signal.signal == 2:
                    if sv.data[it][1] < stop_loss:
                        close = (1 + 0.0005) * sv.data[it][1] if sv.data[it][1] > take_profit else take_profit
                    else:
                        close = stop_loss
                sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
                break
        if sv.signal.signal == 1:
            if sv.data[it][3]< stop_loss:
                #   and sv.data[it][2] < take_profit:                        
                close = stop_loss
                type_close = 'antitarget'
                sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
                break
            if sv.data[it][2]> take_profit and sv.data[it][3] > stop_loss:                        
                close = take_profit
                type_close = 'target'
                sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
                break
        elif sv.signal.signal == 2:
            if sv.data[it][2]> stop_loss:
                #   and sv.data[it][3] > take_profit:                        
                close = stop_loss
                type_close = 'antitarget'
                sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
                break
            if sv.data[it][3]< take_profit and sv.data[it][2] < stop_loss:                        
                close = take_profit
                type_close = 'target'
                sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
                break
    period = (final_it - index)+1
    if sv.settings.printer and sv.counter%sv.settings.iter_count==0:
        serv.print_position(open_time, close_time, sv.settings, sv.profit, saldos_list[-1], type_close, price_open, close_price)
        if sv.settings.drawing:
            sett = f'tp: {sv.settings.close_strategy.take_profit} sl: {sv.settings.close_strategy.init_stop_loss}'
            title = f'up {period} - {sett}' if sv.signal.signal == 1 else f'down {period} - {sett}'
            viz.draw_candlesticks(sv.data[i-sv.settings.chunk_len:it+1], title, sv.settings.chunk_len)
    return period