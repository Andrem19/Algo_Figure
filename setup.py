from models.settings import CloseStrategy, Settings
from datetime import datetime
from itertools import product
import shared_vars as sv

def setup():
    close_s: CloseStrategy = CloseStrategy()
    close_s.target_len = 320
    close_s.init_stop_loss = 0.012
    close_s.take_profit = 0.007
    close_s.distance = 0.002

    settings: Settings = Settings('BTCUSDT', close_s)
    settings.prep_data = 'A1'
    settings.main_variant: int = 1
    settings.printer: bool = False
    settings.drawing: bool = False
    settings.send_pic: bool = False
    settings.iter_count: int = 1
    settings.coin: str = 'BTCUSDT'
    settings.amount: int = 20
    settings.chunk_len: int = 30
    settings.time: int = 5
    settings.deep_open_try = 0.002
    settings.period_open_try = 2
    settings.only = 1

    settings.expressiveness_targ: float = 0.008
    settings.anti_ex_targ: float = 0.005
    
    settings.revers: bool = False
    settings.random: bool = False
    settings.taker_fee: float = 0.12
    settings.maker_fee: float = 0.12

    settings.rsi_max_border = 85
    settings.rsi_min_border = 15
    settings.timeperiod = 16

    sv.settings = settings
    sv.cand_month = 2419200 // (settings.time * 60)
    sv.s = [1] if settings.only == 1 else [2] if settings.only == 2 else (1,2)
def setup_args(args):
    pass