import math;
import numpy as np;
import random;
import matplotlib.pyplot as plt;
from double_moon import *
from utils import *


# YITA = 0.0001 # learning rate parameter

# when the double moon data is linear separable
def rosenblatt_perceptron(sample_data):
  times = len(sample_data) # math.ceil(1 / YITA) * 2
  YITA = 1 / times

  print('sample_data length', len(sample_data))

  # use iteration
  global w_iter
  w_iter = np.matrix([0, 0, 0])
  # X = [1, x, y]
  for i in range(times):
    item = sample_data[i]
    # nonlocal w_iter
    res = w_iter * np.transpose(np.matrix([1, item[0], item[1]]))
    y_iter = res.tolist()[0][0]
    
    sign_iter = np.sign(y_iter)
    class_sign = np.sign(item[2])
    # print('y_iter ', sign_iter)
    if (sign_iter == class_sign):
      # keep w(n + 1) = w(n)
      w_iter = w_iter
    elif (class_sign == -1):
      w_iter = w_iter - np.multiply(item, YITA)
      # print('w_iter deduce to', w_iter)
    elif (class_sign == 1):
      w_iter = w_iter + np.multiply(item, YITA)
      # print('w_iter add to', w_iter)
  return np.asarray(w_iter)[0]

def logistic_regression():
  return 1 

def support_vector_machine():
  return 1

# when the double moon data is not linear separable
def lms():
  # suppose devide line y = ax + b
  # class A value +1, ax + b - y > 0
  # class B value -1, ax + b - y < 0
  # error e = ( sign(ax + b - y) - class_vlaue )^2
  # W = [
  #  w00, w01,
  #  w10, w11
  # ]
  # X = [
  #   1, x
  #   1, y
  # ]
  return 1





if __name__ == "__main__":
  clear()
  # print('get_distance', math.dist([0, 0], [1, 1]))
  data = double_moon(4)

  w = rosenblatt_perceptron(data['all'])

  w_a = -1 * (w[1] / w[2])
  w_b = -1 * (w[0] / w[2])

  a = data['class_a']
  b = data['class_b']
  
  dot1 = plt.scatter(
    [item[0] for item in a],
    [item[1] for item in a],
    label='class 1',
    s=1
  )
  dot2 = plt.scatter(
    [item[0] for item in b],
    [item[1] for item in b],
    label='class 2',
    s=1
  )
  l1 = plt.plot(
    [i * 0.3 for i in range(100)],
    [w_a* i * 0.3 + w_b for i in range(100)],
    label='line'
  )
  # l2 = plt.plot(xList,ySimList, label='eee')
  plt.legend()
  plt.show()
  print('rosenblatt_perceptron w is ', w)
