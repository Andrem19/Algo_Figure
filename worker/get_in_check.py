import shared_vars as sv
import helpers.profit as pr
from datetime import datetime
import ML.prep_data as prep

def get_in_check(index, perc, time_to_take, profit_list: list, saldos_list: list):
    it = index
    start_price = sv.data[it][1]
    counter = 0
    price = sv.data[it][1] * (1-perc) if sv.signal.signal == 1 else sv.data[it][1] * (1+perc)
    while it < index+time_to_take:
        counter+=1
        if sv.signal.signal == 1:
            if sv.data[it][3] < price and sv.data[it][3] > price * (1-sv.settings.close_strategy.init_stop_loss):
                sv.price_of_open = price
                return counter
            elif sv.data[it][3] < price * (1-sv.settings.close_strategy.init_stop_loss):
                sv.profit = pr.profit_counter(True, price, True, price * (1-sv.settings.close_strategy.init_stop_loss))
                profit_list.append([datetime.fromtimestamp(sv.data[it][0]/1000).timestamp()*1000, sv.profit])
                saldo = saldos_list[-1][1] + sv.profit
                saldos_list.append([sv.data[it][0], saldo])
                res = 1 if sv.profit > 0 else 0
                sv.pos_1m.append(res)
                # prep.save_example(index)
                sv.i+= counter
                return 0
        elif sv.signal.signal == 2:
            if sv.data[it][2] > price and sv.data[it][2] < price * (1+sv.settings.close_strategy.init_stop_loss):
                sv.price_of_open = price
                return counter
            elif sv.data[it][2] > price * (1+sv.settings.close_strategy.init_stop_loss):
                sv.profit = pr.profit_counter(True, price, False, price * (1+sv.settings.close_strategy.init_stop_loss))
                profit_list.append([datetime.fromtimestamp(sv.data[it][0]/1000).timestamp()*1000, sv.profit])
                saldo = saldos_list[-1][1] + sv.profit
                saldos_list.append([sv.data[it][0], saldo])
                res = 1 if sv.profit > 0 else 0
                sv.pos_1m.append(res)
                # prep.save_example(index)
                sv.i+= counter
                return 0

        
        it+=1

    if sv.data[it][1] < start_price and sv.signal.signal == 1:
        sv.price_of_open = sv.data[it][1]
        return counter
    elif sv.data[it][1] > start_price and sv.signal.signal == 2:
        sv.price_of_open = sv.data[it][1]
        return counter
    sv.i+= counter
    return 0
            