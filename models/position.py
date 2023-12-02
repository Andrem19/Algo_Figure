import shared_vars as sv
class Position:
    def __init__(self, s_t, amount):
        self.start_time = s_t
        self.amount = amount
        self.price_open = None
        self.price_close = None
        self.type_close = None
        self.duration = None
        self.profit = None
        self.saldo = None
        self.finish_time = None
        self.signal = sv.signal.signal
    def __str__(self):
        return f"StartTime: {self.start_time}, Amount: {self.amount}, Duration: {self.duration}, Price_open: {self.price_open}, Price_close: {self.price_close}, Profit: {self.profit}, Saldo: {self.saldo}, FinishTime: {self.finish_time}, TypeClose: {self.type_close}, Signal: {self.signal}"