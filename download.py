from binance_historical_data import BinanceDataDumper
import multiprocessing
import coins as coins

def main():
    symbols = coins.newCoins + coins.oldCoins
    
    times = [1, 5, 15]
    for t in times:
        for symbol in symbols:
            data_dumper = BinanceDataDumper(
                path_dir_where_to_dump="newdata",
                asset_class="um",  # spot, um, cm
                data_type="klines",  # aggTrades, klines, trades
                data_frequency=f"{t}m",
            )
            data_dumper.dump_data(
                tickers=symbol,
                date_start=None,
                date_end=None,
                is_to_update_existing=True,
                tickers_to_exclude=["UST"],
            )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
