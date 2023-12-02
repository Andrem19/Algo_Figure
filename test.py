import helpers.services as serv
import shared_vars as sv
import numpy as np


arr = np.array([1,2,3,4,5,6,7,8,9,10,11])
arr_2 = np.array([1,2,3,4,5,5,6])

arr_3 = arr + arr_2

print(arr_3)