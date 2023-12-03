class Indicators:
    def __init__(self) -> None:
        self.rsi: int = 2
        self.stochastic: int = 2
        self.natr: int = 2
        self.bol: int = 2
        self.supres: int = 2
        self.fib: int = 2
        self. adx: int = 1
        self.mfi: int = 1
        self.obv: int = 1
        self.smaema: int = 1
        self.macd: int = 1
    
    def to_dict(self):
        properties = [attr for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))]
        
        return {prop: getattr(self, prop) for prop in properties if getattr(self, prop) == 1}

    def __str__(self) -> str:
        properties = [attr for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))]

        output = ''
        for prop in properties:
            value = getattr(self, prop)
            if value == 1:
                output += f'{prop}: {value}\n'

        return output

class CloseStrategy:
    def __init__(self):
        self.target_len: int = 5
        self.init_stop_loss: float = 0.008
        self.take_profit: float = 0.007
        self.distance: float = 0.0008

    def to_dict(self):
        return {
            'target_len': self.target_len,
            'init_stop_loss': self.init_stop_loss,
            'take_profit': self.take_profit,
            'distance': self.distance,
        }

    def __str__(self) -> str:
        properties = [attr for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))]

        output = ''
        for prop in properties:
            value = getattr(self, prop)
            output += f'{prop}: {value}\n'

        return output

    
class Settings:
    def __init__(self, coin: str, close_s: CloseStrategy):
        self.prep_data: str = 'A1'
        self.main_variant: int = 1
        self.printer: bool = False
        self.drawing: bool = False
        self.send_pic: bool = False
        self.iter_count: int = 100
        self.coin: str = coin
        self.amount: int = 20
        self.chunk_len: int = 252
        self.step_len: int = 9
        self.time: int = 1
        self.deep_open_try: float = 0.003
        self.period_open_try: int = 3
        self.only: int = 0

        self.expressiveness_targ: float = 0.005
        self.anti_ex_targ: float = 0.004
        
        self.revers: bool = True
        self.random: bool = False
        self.taker_fee: float = 0.12
        self.maker_fee: float = 0.12
        self.close_strategy: CloseStrategy = close_s

        self.rsi_max_border: int = 85
        self.rsi_min_border: int = 15
        self.timeperiod: int = 14
        