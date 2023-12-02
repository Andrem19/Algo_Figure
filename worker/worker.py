import worker.process_2 as prc_2
import worker.process as prc
import helpers.services as serv
import ML.prep_data as prep
import signal as sg
import helpers.tel as tel
import shared_vars as sv
import worker.signal as sg
import worker.get_in_check as gic


async def run():
    try:
        data_len = len(sv.data)
        print(data_len)
        chunk_len = sv.settings.chunk_len
        target_len = sv.settings.close_strategy.target_len
        type_of_event: list = []
        saldos_list: list = []
        profit_list: list = []
        durations_list: list = []
        saldos_list.append([sv.data[0][0],0])
        sv.i = chunk_len*3
        open_index = 0

        if sv.settings.prep_data != 'B':
            while sv.i < data_len - chunk_len*3:
                prep.prep_train_data_5(sv.data[sv.i-(chunk_len):sv.i], sv.data[sv.i:sv.i+target_len])
                sv.i+=2
                if sv.i%2000==0:
                        print(sv.i)
        else:
            while sv.i < data_len - chunk_len*3:
                sg.get_signal(sv.i, chunk_len)
                if sv.settings.revers:
                    serv.reverse()
                # res = 0
                # if sv.signal.signal in (1, 2):
                #     res = gic.get_in_check(sv.i, sv.settings.deep_open_try, sv.settings.period_open_try, profit_list, saldos_list)
                #     sv.i+=res
                if sv.signal.signal in (1, 2):
                    open_index = sv.i
                    tm1 = prc_2.position_proccess(sv.i, data_len, profit_list, saldos_list, type_of_event)
                    durations_list.append(tm1-1)
                    res = 1 if sv.profit > 0 else 0
                    sv.pos_1m.append(res)
                    # prep.save_example(open_index)
                    sv.i+=tm1
                else: 
                    sv.i+=10
                    # if sv.i%100==0:
                    #     print(len(saldos_list), saldos_list[-1][1])
                    #     plus = len([x for x in profit_list if x[1] > 0])
                    #     minus = len([x for x in profit_list if x[1] < 0])
                    #     if plus != 0 and minus != 0:
                    #         pl_perc = plus/((plus + minus)/100)
                    #     else:
                    #         pl_perc = 0
                    #     print(f'plus:{plus} pl_perc:{pl_perc},minus: {minus} mn_perc: {100-pl_perc}')
                    #     print(sv.i)
        result = serv.calculate_result(saldos_list, type_of_event, durations_list)
        return profit_list, saldos_list, result
    except Exception as e:
        print(f'Error: {e}')
        # await tel.send_inform_message(f'Error: {e}', None, False)