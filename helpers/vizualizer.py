import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
import matplotlib.dates as mdates
from datetime import datetime
import os
from PIL import Image
import numpy as np
from keras.preprocessing import image

binance_dark = {
    "base_mpl_style": "dark_background",
    "marketcolors": {
        "candle": {"up": "#3dc985", "down": "#ef4f60"},  
        "edge": {"up": "#3dc985", "down": "#ef4f60"},  
        "wick": {"up": "#3dc985", "down": "#ef4f60"},  
        "ohlc": {"up": "green", "down": "red"},
        "volume": {"up": "#247252", "down": "#82333f"},  
        "vcedge": {"up": "green", "down": "red"},  
        "vcdopcod": False,
        "alpha": 1,
    },
    "mavcolors": ("#ad7739", "#a63ab2", "#62b8ba"),
    "facecolor": "#1b1f24",
    "gridcolor": "#2c2e31",
    "gridstyle": "--",
    "y_on_right": True,
    "rc": {
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.edgecolor": "#474d56",
        "axes.titlecolor": "red",
        "figure.facecolor": "#161a1e",
        "figure.titlesize": "x-large",
        "figure.titleweight": "semibold",
    },
    "base_mpf_style": "binance-dark",
}

# def save_candlesticks(candles: list, path: str):
#     # Convert the candlesticks data into a pandas DataFrame
#     df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
#     df.set_index('timestamp', inplace=True)
#     figsize = (6, 6)
#     # Plot the candlestick chart using mpf.plot()
#     fig, axes = mpf.plot(df, type='candle', style='binance', returnfig=True, figsize=figsize)

#     # Save the figure to the specified path
#     fig.savefig(path)
#     plt.close(fig)
def save_candlesticks(candles: list, path: str):
    # Convert the candlesticks data into a pandas DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    df.set_index('timestamp', inplace=True)

    # Define the style dictionary
    my_style = mpf.make_mpf_style(base_mpf_style='binance', gridstyle='')

    # Plot the candlestick chart using mpf.plot()
    mpf.plot(df, type='candle', style=my_style, axisoff=True, figratio=(4,4), savefig=path)

def compile_pic_candel(candles: list):
    # Convert the candlesticks data into a pandas DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    df.set_index('timestamp', inplace=True)
    figsize = (6, 6)
    # Plot the candlestick chart using mpf.plot()
    fig, axlist = mpf.plot(df, type='candle', style='binance', returnfig=True, figsize=figsize)
    # Convert the figure to a PIL Image
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    img = img.resize((180, 180), Image.ANTIALIAS)
    # Convert the PIL Image to a numpy array
    x = image.img_to_array(img)
    # Expand the dimensions to match the input shape of your model
    # x = np.expand_dims(x, axis=0)
    return x

def draw_candlesticks(candles: list, type_labels: str, mark_index: int):
    # Convert the candlesticks data into a pandas DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    df.set_index('timestamp', inplace=True)
    figsize = (10, 6)
    # Plot the candlestick chart using mpf.plot()
    fig, axlist = mpf.plot(df, type='candle', style=binance_dark, title=type_labels, returnfig=True, figsize=figsize)

    # Add percentage labels to the candlestick chart
    # _add_candlestick_labels(axlist[0], df)

    if type_labels == 'up':
        axlist[0].annotate('MARK', (mark_index, df.iloc[mark_index]['open']), xytext=(mark_index, df.iloc[mark_index]['open']-10),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))
    elif type_labels == 'down':
        axlist[0].annotate('MARK', (mark_index, df.iloc[mark_index]['open']), xytext=(mark_index, df.iloc[mark_index]['open']+10),
                        arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Display the chart
    mpf.show()

def draw_graph(values):
    periods = range(1, len(values) + 1)
    plt.plot(periods, values)
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.title('Graph for Values')
    plt.show()



def plot_time_series(data_list: list, save_pic: bool, points: int, dont_show: bool):
    path = f'_pic/{datetime.now().date().strftime("%Y-%m-%d")}'
    timestamps = [item[0] for item in data_list]
    values = [item[1] for item in data_list]
    
    # Преобразование timestamp в формат даты
    dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
    
    # Создание графика
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45) # Поворот дат для лучшей читаемости
    periods = range(1, len(values) + 1)
    # Построение графика
    ax.plot(dates, values)
    
    # Добавление подписей и заголовка
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.title('Graph for Values')

     # Add periods close to the dates
    for i, (date, value) in enumerate(zip(dates, values)):
        if i % points == 0:
            ax.text(date, value, f"{i}", verticalalignment='top', horizontalalignment='center', fontsize=9, color='red')
    
    # Отображение графика
    if not dont_show:
        plt.tight_layout()

    if save_pic:
        if not os.path.exists(path):
            os.makedirs(path)
        end_path = f'{path}/{datetime.now().timestamp()}.png'
        plt.savefig(end_path)
        return end_path
    if not dont_show:
        plt.show()
    return None

import matplotlib.pyplot as plt

def plot_two_lists(list1, list2):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(list1)
    plt.title('Prediction')
    
    plt.subplot(1, 2, 2)
    plt.plot(list2)
    plt.title('Reality')
    
    plt.show()
