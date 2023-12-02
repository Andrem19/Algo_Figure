import shared_vars as sv
import helpers.vizualizer as viz
from datetime import datetime
import helpers.services as serv

def position_proccess(i: int, data_len: int, profit_list: list, saldos_list: list, type_of_event: list):
    sv.counter +=1
    s_loss = 0
    stop_loss = 0
    close_time = None
    close_price = 0
    take_profit = 0
    last_border_sl = 0
    target_len = 0
    price_open = 0
    distance = 0

    s_loss = sv.settings.close_strategy.init_stop_loss
    take_profit = sv.settings.close_strategy.take_profit
    target_len = sv.settings.close_strategy.target_len
    distance = sv.settings.close_strategy.distance

    open_time = datetime.fromtimestamp(sv.data[i][0]/1000)
    if sv.signal.signal == 1:
        price_open = sv.price_of_open # sv.data[i][1] * (1 - 0.0001)
        stop_loss = (1 - s_loss) * float(price_open)
        take_profit = (1 + take_profit) * float(price_open)

    elif sv.signal.signal == 2:
        price_open = sv.price_of_open # sv.data[i][1] * (1 + 0.0001)
        stop_loss = (1 + s_loss) * float(price_open)
        take_profit = (1 - take_profit) * float(price_open)

    last_border_sl = price_open * (1+0.007) if sv.signal.signal == 1 else price_open * (1-0.007)

    for it in range(i, data_len):
        final_it = it
        # duration = (datetime.fromtimestamp(sv.data[it][0] / 1000) - open_time).total_seconds() / 60
        # if duration < target_len:
        if sv.signal.signal == 1:
            if sv.data[it][3]< stop_loss \
                and sv.data[it][2] < last_border_sl * (1+distance*2):                        
                close = stop_loss
                type_close = 'antitarget'
                sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
                break
            elif sv.data[it][3]< stop_loss \
                and sv.data[it][2] > last_border_sl * (1+distance*2):
                close = stop_loss * (1+distance/2)
                type_close = 'antitarget'
                sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
                break

            if sv.data[it][2] > take_profit and sv.data[it][3] > stop_loss:
                close = take_profit
                type_close = 'target'
                sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
                break

            if sv.data[it][2] > last_border_sl * (1+distance*2):
                stop_loss = (1 + distance) * last_border_sl
                if sv.data[it][2] > last_border_sl * (1+distance*4):
                    stop_loss = (1 + distance*2) * last_border_sl
                last_border_sl = stop_loss

        elif sv.signal.signal == 2:
            if sv.data[it][2]> stop_loss \
                and sv.data[it][3] > last_border_sl * (1-distance*2):                        
                close = stop_loss
                type_close = 'antitarget'
                sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
                break
            elif sv.data[it][2]> stop_loss \
                and sv.data[it][3] < last_border_sl * (1-distance*2):
                close = stop_loss * (1-distance/2)
                type_close = 'antitarget'
                sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
                break

            if sv.data[it][3] < take_profit and sv.data[it][2] < stop_loss:
                close = take_profit
                type_close = 'target'
                sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
                break

            if sv.data[it][3] < last_border_sl * (1-distance*2):
                stop_loss = (1 - distance) * last_border_sl
                if sv.data[it][3] < last_border_sl * (1-distance*4):
                    stop_loss = (1 - distance*2) * last_border_sl
                last_border_sl = stop_loss
                    
        # elif duration >= target_len:
        #     type_close = 'timefinish'
        #     close = 0
        #     if sv.signal.signal == 1:
        #         if sv.data[it][1] > stop_loss:
        #             close = (1 - 0.0005) * sv.data[it][1]
        #         else:
        #             close = stop_loss
        #     elif sv.signal.signal == 2:
        #         if sv.data[it][1] < stop_loss:
        #             close = (1 + 0.0005) * sv.data[it][1]
        #         else:
        #             close = stop_loss
        #     sv.profit, type_close, close_time, close_price = serv.process_profit(open_time, profit_list, type_close, price_open, sv.data[it], saldos_list, close, type_of_event)
        #     break
    if sv.settings.printer and sv.counter%sv.settings.iter_count==0:
        serv.print_position(open_time, close_time, sv.settings, sv.profit, saldos_list[-1], type_close, price_open, close_price)
        if sv.settings.drawing:
            chunk_len = sv.settings.chunk_len_1 if sv.signal.data == 1 else sv.settings.chunk_len_2
            title = 'up' if sv.signal.signal == 1 else 'down'
            viz.draw_candlesticks(sv.data[i-chunk_len:it], title, chunk_len)
    return (final_it - i)+1