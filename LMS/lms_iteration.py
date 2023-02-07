import math;
import numpy as np;
import random;
import matplotlib.pyplot as plt;
import utils 
import os


YITA = 0.0001 # learning rate parameter


def randomNoise(base = 10):
  # -base  ~  +base 
  return  random.random() * base * 2 - base

# suppose (x1, x2, x3, x4) -> y
# W = (w1, w2, w3, w4)T
# Y = WX + e
def generateData(num_count = 4, max_num = 30):
  xList = [(random.random() * max_num) for i in range(num_count)]
  # W = [5, 3, 7, 2]T
  W = [5, 3, 7, 2]
  y_matrix = np.matrix(xList) * (np.transpose(np.matrix(W))) 
  y = np.asarray(y_matrix).tolist()[0][0] + randomNoise(3)
  # print('y', y_matrix)
  return {'xList': xList, 'y': y}

# plt.plot(x,y)
# plt.show()5



def lms_iteration(iterration_times = 10):
  w_iter = [0, 0, 0, 0]
  for i in range(iterration_times):
    data = generateData()
    # print('xList', utils.dotdict(data).xList)
    x = data['xList']
    y = data['y']
    # print('x', x, 'y', y)
    # e(n) = y(n) - w_iter(n)T * x(n)

    e_iter = y - np.matrix(w_iter) * np.transpose(np.matrix(x))
    q = YITA * np.array(x)
    # a = YITA * np.array(x) * e_iter
    # b = np.multiply(YITA * np.array(x), (e_iter[0][0]))
    w_iter = np.matrix(w_iter) + np.multiply(YITA * np.array(x), (e_iter[0][0]))

  return w_iter





if __name__ == "__main__":
  # allList = generateData();
  # print(allList)
  os.system( 'cls' ) #
  os.system( 'clear' )
  times = math.ceil(1 / YITA) * 2
  w_estimate = lms_iteration(times)
  print('w_estimate diff', [round(i, 3) for i in (w_estimate - [5, 3, 7, 2]).tolist()[0]])


