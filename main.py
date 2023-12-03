import shared_vars as sv
import variants.avarage_saldo as av
import coins as coins
from datetime import datetime
import setup as setup
import helpers.services as serv
from itertools import product
import asyncio
import sys

sv.telegram_api = 'API_TOKEN_1'
coin_list = coins.new_collection + coins.coins_to_add#['LTCUSDT']# coins.all_coins

async def main(args):
    global coin_list
    if len(args) > 1:
        setup.setup_args(args)

    if sv.settings.main_variant == 1:
        prep_data_vars = [
            # 'A',
            # 'A2',
            # 'AA',
            'B'
            ] if sv.prep_data == 'A' else ['B']
        for vars in prep_data_vars:
            print(f'variant - {vars}')

            sv.settings.prep_data = vars
            # serv.train_models()
            if sv.settings.prep_data == 'B':
                # coin_list = coins.test
                # sv.settings.close_strategy.target_len = 0
                coin_list = coins.best_set # coins.new_collection + coins.coins_to_add


            sv.start_date = datetime(2022, 1, 1) if sv.settings.prep_data != 'B' else datetime(2017, 12, 1)
            sv.finish_date = datetime(2023, 1, 1) if sv.settings.prep_data != 'B' else datetime(2023, 11, 1)
            await av.avarag_saldo(coin_list)
    elif sv.settings.main_variant == 2:
        prep_data_vars = [
            # 'A1',
            # 'A2',
            # 'AA',
            'B'
            ] if sv.prep_data == 'A' else ['B']
        for vars in prep_data_vars:
            print(f'variant - {vars}')

            sv.settings.prep_data = vars
            serv.train_models()
            if sv.settings.prep_data == 'B':
                # coin_list = coins.test
                coin_list = coins.new_collection + coins.coins_to_add


            sv.start_date = datetime(2022, 1, 1) if sv.settings.prep_data != 'B' else datetime(2017, 4, 1)
            sv.finish_date = datetime(2023, 1, 1) if sv.settings.prep_data != 'B' else datetime(2023, 11, 1)

            for d_t, p_o, s_l, t_p in product(range(len(sv.deep_open_try)), range(len(sv.period_open_try)), range(len(sv.stop_loss)), range(len(sv.take_profit))):
                sv.settings.deep_open_try = sv.deep_open_try[d_t]
                sv.settings.period_open_try = sv.period_open_try[p_o]
                sv.settings.close_strategy.init_stop_loss = sv.stop_loss[s_l]
                sv.settings.close_strategy.take_profit = sv.take_profit[t_p]
                await av.avarag_saldo(coin_list)
if __name__ == "__main__":
    setup.setup()
    asyncio.run(main(sys.argv[1:]))