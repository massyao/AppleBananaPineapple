import math;
import numpy as np;
import random;
import matplotlib.pyplot as plt;
import matplotlib.pylab as pylab;
# from perceptron.perceptron import Perceptron
from double_moon import *
from utils import *

# p = Perceptron() # use a short

# YITA = 0.0001 # learning rate parameter


def activation_function(x):
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# when the double moon data is linear separable
def rosenblatt_perceptron(sample_data):
  times = 100 # math.ceil(1 / YITA) * 2
  YITA = 0.01 # 1 / len(sample_data)

  # error tip: w_iter should use [0, 0],but I write [0, 0, 0], which includes a bias weight
  # np.matrix([random.random() * 2 - 1 for _ in range(3)])
  w_iter = np.zeros((1, 3))
  # X = [1, x, y]
  for j in range(times):
    for i in range(len(sample_data)):
      item = sample_data[i]
      x = [1, item[0], item[1]]
      # res = w_iter * np.transpose(np.matrix([item[0], item[1]]))
      res = w_iter * np.transpose(np.matrix(x))
      iterError = item[2] - (1 if res.tolist()[0][0] >= 0 else -1)
      # new_w = [i + YITA * iterError * j for i, j in zip(w, [item[0], item[1]])]
      # error tip: item[0:2], I write item[:1] which make the error
      # w_iter = w_iter + np.multiply(item[0:2], YITA * iterError)
      w_iter = w_iter + np.multiply(x, YITA * iterError)
      # print()
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


def test_rosenblatt_perceptron():
  # print('get_distance', math.dist([0, 0], [1, 1]))
  data = double_moon(3, 1000)
  # data = double_moon(0, 1000)
  # data = double_moon(-3, 1000)

  w = rosenblatt_perceptron(data['all'])

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

  # w = [0] + list(ww)
  x1 = [i * 0.3 for i in range(100)] # [min(X[:,0]), max(X[:,0])]
  # w0 + w1 * x + w2 * y  = 0 
  m = -w[1]/w[2]
  c = -w[0]/w[2]
  x2 = [m*x + c for x in x1]
  plt.plot(x1, x2, 'y-')

  plt.legend()
  plt.show()
  print('rosenblatt_perceptron w is ', w)
  return


# print(1.0 == 1)

if __name__ == "__main__":
  clear()
  test_rosenblatt_perceptron()

