import shared_vars as sv
import numpy as np

def convert_to_relative(data: np.ndarray, first: np.ndarray, col: list):
    eps = 1e-10
    data = np.where(data == 0, eps, data)
    first = np.where(first == 0, eps, first)
    
    relative_data = np.zeros_like(data)
    relative_data[1:] = (data[1:] - data[:-1]) / data[:-1] * 100
    relative_data[0] = (data[0] - first) / first * 100
    relative_data = np.round(relative_data, 3)
    relative_data[:, 5] = np.round(relative_data[:, 5], 3)
    return relative_data[:, col].flatten()

def prep_rel_data(data: np.ndarray, first: np.ndarray, cols: list)->list:
    result_arr = convert_to_relative(data, first, cols)
    return result_arr.tolist()
# def unpack_arrays(arr):
#     result = []
    
#     for sub_arr in arr:
#         result.extend(sub_arr)
    
#     return result

# def convert_to_relative(data: np.ndarray, first: np.ndarray, col: list):
#     newdata = []
#     try:
#         for i in range(len(data)):
#             try:
#                 if i == 0:
#                     previous_candle = first
#                 else:
#                     previous_candle = data[i - 1]
#                 current_candle = data[i]

#                 if previous_candle[1] != 0 and previous_candle[1] != current_candle[1]:
#                     if previous_candle[1] < current_candle[1]:
#                         relative_open = round(((current_candle[1] - previous_candle[1]) / previous_candle[1]) * 100, 3)
#                     else:
#                         relative_open = -round(((previous_candle[1] - current_candle[1]) / previous_candle[1]) * 100, 3)
#                 else:
#                     relative_open = 0

#                 if previous_candle[2] != 0 and previous_candle[2] != current_candle[2]:
#                     if previous_candle[2] < current_candle[2]:
#                         relative_high = round(((current_candle[2] - previous_candle[2]) / previous_candle[2]) * 100, 3)
#                     else:
#                         relative_high = -round(((previous_candle[2] - current_candle[2]) / previous_candle[2]) * 100, 3)
#                 else:
#                     relative_high = 0

#                 if previous_candle[3] != 0 and previous_candle[3] != current_candle[3]:
#                     if previous_candle[3] < current_candle[3]:
#                         relative_low = round(((current_candle[3] - previous_candle[3]) / previous_candle[3]) * 100, 3)
#                     else:
#                         relative_low = -round(((previous_candle[3] - current_candle[3]) / previous_candle[3]) * 100, 3)
#                 else:
#                     relative_low = 0

#                 if previous_candle[4] != 0 and previous_candle[4] != current_candle[4]:
#                     if previous_candle[4] < current_candle[4]:
#                         relative_close = round(((current_candle[4] - previous_candle[4]) / previous_candle[4]) * 100, 3)
#                     else:
#                         relative_close = -round(((previous_candle[4] - current_candle[4]) / previous_candle[4]) * 100, 3)
#                 else:
#                     relative_close = 0

#                 if previous_candle[5] != 0 and previous_candle[5] != current_candle[5]:
#                     if previous_candle[5] < current_candle[5]:
#                         relative_volume = round(((current_candle[5] - previous_candle[5]) / previous_candle[5]), 3)
#                     else:
#                         relative_volume = -round(((previous_candle[5] - current_candle[5]) / previous_candle[5]), 3)
#                 else:
#                     relative_volume = 0

#                 cols = [relative_open, relative_high, relative_low, relative_close, relative_volume]
#                 cols_we_add = []
#                 for n in col:
#                     cols_we_add.append(cols[n-1])

#                 newdata.append(cols_we_add)
#             except Exception as e:
#                 print(f"Error in loop iteration {i}: {str(e)}")
#     except Exception as e:
#         print(f"Error in main loop: {str(e)}")

#     one_array = unpack_arrays(newdata)
#     return one_array


# def prep_rel_data(data: np.ndarray, first: np.ndarray, cols: list, aditional_cols: list):
#     result_arr = convert_to_relative(data, first, cols)
#     if len(aditional_cols)>0:
#         for col in aditional_cols:
#             result_arr.append(col)
#     return result_arr