import shared_vars as sv
from sklearn.decomposition import PCA
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import Indicators.talibr as ta
import tensorflow as tf
from joblib import load
from keras.models import load_model
import helpers.vizualizer as viz
import helpers.relative_1 as rel_1
import helpers.relative_2 as rel_2
import helpers.services as serv
from keras.preprocessing import image
import helpers.vizualizer as viz


# tf.get_logger().setLevel('ERROR')

def get_signal(data: np.ndarray,):
    serv.delete_folder_contents('_sample')
    viz.save_candlesticks(data, '_sample/sample.png')


    # Загрузка изображения
    img_path = '_sample/sample.png'
    img = image.load_img(img_path, target_size=(240, 240))
    # img = viz.compile_pic_candel(data)
    # Преобразование изображения в массив numpy
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Создание пакета из одного изображения

    # Нормализация значений пикселей от 0 до 1
    img_array /= 255.

    # Предсказание класса изображения
    predictions_1 = sv.model_1.predict(img_array, verbose=0)
    predicted_class_1 = np.argmax(predictions_1[0]) # Выбор класса с наибольшей вероятностью
    if predicted_class_1 == 0:
        return 0
    elif predicted_class_1 == 1:
        return 1
    elif predicted_class_1 == 2:
        return 2
    else:
        print(f'predicted_class_1 something wrong: {predicted_class_1}')
        return 0


def get_signal_2(data: np.ndarray,):
    
    cols_1 = [2,4]
    cols_2 = [2,3,4]
    futures = rel_2.prep_rel_data(data[-sv.settings.chunk_len:], cols_2)
    lenth = len(futures) //2
    futures = futures[lenth:]
    signals = []
    closes = data[:,4]
    highs = data[:,2]
    lows = data[:,3]
    volumes = data[:,5]
    signals.append(ta.rsi(closes))
    signals.append(ta.bollinger(closes))
    signals.append(ta.trend(closes))
    signals.append(ta.natr(highs, lows, closes))
    signals.append(ta.sma_ema(closes))
    signals.append(ta.mfi(highs, lows, closes, volumes))
    signals.append(ta.adx(highs, lows, closes))
    signals.append(ta.support_resistance(highs, lows, closes))
    signals.append(ta.macd(closes))
    signals.append(ta.obv(closes, volumes))
    signals.append(ta.stochastic(highs, lows, closes))
    signals.append(ta.average_true_range(highs, lows, closes))
    signals.append(ta.commodity_channel_index(highs, lows, closes))
    signals.append(ta.rate_of_change(closes))

    for key, value in sv.indicators.items():
        futures.append(value)
    # data = np.expand_dims([futures], axis=2)
    # Предсказание класса изображения
    predictions_1 = sv.model_1.predict(futures, verbose=0)
    # print(f'prediction: {predictions_1}')
    predicted_class_1 = predictions_1[0] # np.argmax(predictions_1[0]) # Выбор класса с наибольшей вероятностью
    if predicted_class_1 == 0:
        return 0
    elif predicted_class_1 == 1:
        return 1
    elif predicted_class_1 == 2:
        return 2
    else:
        print(f'predicted_class_1 something wrong: {predicted_class_1}')
        return 0


def get_signal_GAF(data: np.ndarray):
    cols_2 = [2,3,4]
    features = rel_2.prep_rel_data(data[-sv.settings.chunk_len:], cols_2)
    features = np.array(features).reshape(1, -1)
    
    # features = sv.scaler.transform(features)

    # features = sv.pca.transform(features)

    # GAF Transformation

    data_gaf = sv.gaf.transform(features)

    # Reshape the data for CNN
    data_gaf = data_gaf.reshape(-1, 20, 20, 1)

    # Make prediction
    predictions = sv.model_1.predict(data_gaf, verbose=0)

    predicted_class_1 = np.argmax(predictions[0]) # Выбор класса с наибольшей вероятностью
    if predicted_class_1 == 0:
        return 0
    elif predicted_class_1 == 1:
        return 1
    elif predicted_class_1 == 2:
        return 2
    else:
        print(f'predicted_class_1 something wrong: {predicted_class_1}')
        return 0

def get_signal_LSTM(data: np.ndarray, target: np.ndarray):
    last_candel = data[-1][4]
    futures = data[:, 2:5].flatten()
    # futures = np.delete(futures, slice(0, 60), axis=1)

    data = np.array(futures).reshape(1, -1)
    data_scaled = sv.scaler.transform(data)
    
    # Изменение формы данных для LSTM
    data_scaled = data_scaled.reshape((data_scaled.shape[0], 1, data_scaled.shape[1]))

    predictions_1 = sv.model_1.predict(data_scaled, verbose=0)
    list1 = arr = np.insert(predictions_1[0], 0, last_candel)
    list2 = arr = np.insert(target, 0, last_candel)
    viz.plot_two_lists(list1, list2)
    # print(predictions_1)
    return
    predicted_class_1 = np.argmax(predictions_1[0]) # Выбор класса с наибольшей вероятностью
    if predicted_class_1 == 0:
        return 0
    elif predicted_class_1 == 1:
        return 1
    elif predicted_class_1 == 2:
        return 2
    else:
        print(f'predicted_class_1 something wrong: {predicted_class_1}')
        return 0
