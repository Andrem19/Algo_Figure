import os
import worker.worker as w
import statistics
import helpers.get_data as gd
import shared_vars as sv
import helpers.tel as tel
import helpers.vizualizer as viz
import helpers.services as serv
import variants.infrastructure as inf
from collections import Counter


async def avarag_saldo(coins_list: list):
    sv.category = 0
    profit_path = '_profits/profits.txt'
    final_res = {}

    await inf.start_of_program_preparing()

    median_duration = []
    avarage_duration = []
    for c in coins_list:
        try:
            print(c)
            file_coin = f'_crypto_data/{c}/{c}_1m.csv'
            if not os.path.exists(file_coin):
                print(f'{c} doesnt exist')
                await tel.send_inform_message(f'{c} doesnt exist', None, False)
                continue
            sv.settings.coin = c
            gd.load_data_sets(sv.start_date, sv.finish_date)
            profit_list, saldos_list, result = await w.run()

            if sv.settings.prep_data != 'B':
                print(f'len: {len(sv.array_perc_diff)}')
                print(f'max: {max(sv.array_perc_diff)}, min: {min(sv.array_perc_diff)} median: {round(statistics.median(sv.array_perc_diff), 4)}, avarage: {round(statistics.mean(sv.array_perc_diff), 4)}')

            if result != None and profit_list != None and sv.settings.prep_data == 'B':
                pls = 0
                mns = 0
                
                for i in range(1, len(saldos_list)):
                    if saldos_list[i][1] > saldos_list[i-1][1]:
                        pls +=1
                    else:
                        mns +=1
                percentage = pls / (pls + mns)
                if sv.settings.send_pic:
                    points = serv.get_points_value(len(saldos_list))
                    path = viz.plot_time_series(saldos_list, True, points, True)
                    await tel.send_inform_message(f'{c}', path, True)

                sv.global_pos_1m.extend(sv.pos_1m)
                pos_1_count_0 = sv.pos_1m.count(0)
                pos_1_count_1 = sv.pos_1m.count(1)

                sv.pos_1m.clear()

                sg = f'{sv.settings.coin} - Profit: {round(saldos_list[-1][1], 3)} Lenth: {len(saldos_list)}, percent: {round(percentage, 3)},\n1m: {pos_1_count_1}, {pos_1_count_0},\nShort_Long: {sv.short_long_stat}\nDur_med: {result["duration_median"]} Dur_avg: {result["duration_avarage"]}'
                print(sg)
                # await tel.send_inform_message(sg, '', False)
                for key, value in sv.global_short_long_stat.items():
                    sv.global_short_long_stat[key] += sv.short_long_stat[key]
                for key, value in sv.short_long_stat.items():
                    sv.short_long_stat[key] = 0
                for obj_type in ['events']:
                    for key, value in result[obj_type].items():
                        if type(value) is int or type(value) is float:
                            if key in final_res:
                                final_res[key] += value
                            else:
                                final_res[key] = value
                median_duration.append(result['duration_median'])
                avarage_duration.append(result['duration_avarage'])
                serv.save_list(profit_list, profit_path)
                
        except Exception as e:
            print(e)

    if sv.settings.prep_data != 'B':
        await tel.send_inform_message(f'Data collected successfuly!', '', False)
        return
    
    profit_data = serv.load_saldo_profit('_profits')

    # remove position more than two in the same time
    it = 3
    while it < len(profit_data)-1:
        if profit_data[it-1][0] == profit_data[it][0] and profit_data[it-2][0] == profit_data[it][0]:
            profit_data.pop(it)
        it+=1

    # complete saldo list from profit list
    saldos_list = []
    if len(profit_data) > 3:
        saldos_list.append([profit_data[0][0], 0])
        for i in range(1, len(profit_data)):
            saldos_list.append([profit_data[i][0], saldos_list[-1][1]+profit_data[i][1]])
        print(f'saldo list: {len(saldos_list)}')
        print(saldos_list[3][0])

    # plus and minus statistic
    plus = 0
    minus = 0
    for i in range(1, len(saldos_list)):
        if saldos_list[i][1] > saldos_list[i-1][1]:
            plus +=1
        else:
            minus +=1
    
    print(f'plus: {plus} minus: {minus}')
    if plus != 0 and minus != 0:
        percentage = plus / (plus + minus)
    else:
        percentage = 0
    print(f'percentage: {percentage}')
    points = serv.get_points_value(len(saldos_list))
    path = viz.plot_time_series(saldos_list, True, points, True)
    pos_1_count_0 = sv.global_pos_1m.count(0)
    pos_1_count_1 = sv.global_pos_1m.count(1)
    if path is not None:
        if len(avarage_duration) > 0:
            info = {
                'target_len': 'off',
                'rsi': '85-15',
                'only': 'short' if sv.settings.only == 2 else 'long' if sv.settings.only == 1 else 'off',
                'chunk_len': sv.settings.chunk_len,
                'stop_loss': sv.settings.close_strategy.init_stop_loss,
                'take_profit': sv.settings.close_strategy.take_profit,
                'deep_open_try': sv.settings.deep_open_try,
                'period_open_try': sv.settings.period_open_try,
                'duration_avarage': round(sum(avarage_duration) / len(avarage_duration), 4),
                'duration_median': round(statistics.median(median_duration), 4),
                '1m': f'{pos_1_count_1}, {pos_1_count_0}',
            }
        else:
            info = {}
        info.update(sv.global_short_long_stat)
        sorted_dict = dict(sorted(final_res.items(), key=lambda x: x[1], reverse=True))
        sorted_dict = {key: value for key, value in sorted_dict.items() if value != 1}
        for key, value in sorted_dict.items():
            print(key, value)
        print(str(info))
        if len(saldos_list) > 0:
            print(f'all: {plus+minus}, plus: {plus} minus: {minus} saldo: {saldos_list[-1][1]} perc: {percentage}')
            print(str(info))
            await tel.send_inform_message(f'all: {plus+minus}, plus: {plus} minus: {minus} saldo: {saldos_list[-1][1]} perc: {percentage}', path, True)
            await tel.send_inform_message(str(info), '', False)