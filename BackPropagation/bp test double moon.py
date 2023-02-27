
import math;
import numpy as np;
import random;
import matplotlib.pyplot as plt;
import matplotlib.pylab as pylab;
# from matplotlib.pylab import *
# from perceptron.perceptron import Perceptron
from double_moon import *
from utils import *
import os

# this back propagation code costs me three weeks
# from 2023/2/7 to 2023/2/28
# no reference, just translate Haykin's book formula to python code
#  I use mattmazur.com's example to revise my code

absolute_path = os.path.dirname(__file__)
# file_path = os.path.join(absolute_path, 'log.txt')

# use double moon as data 
# inputs number 2
# out puts number 1
# hidden layer nodes set to 5


train_epochs = 10

ETA = 0.5

def activation_function(x):
  return 1 / (1 + np.exp(-x))

def local_gradient(x):
  f = activation_function(x)
  if type(x) == type(0):
    return f * (1 - f)
  elif isinstance(x, np.ndarray):
    return np.multiply(f, 1 - f)



def train(
    data_list,
    layer_node_num = [2, 2, 2],
  ):

  H_W=[0.15, 0.2, 0.25, 0.3],
  hidden_layer_bias=0.35,
  O_W=[0.4, 0.45, 0.5, 0.55],
  output_layer_bias=0.6

  input_node_num = layer_node_num[0]
  output_node_num = layer_node_num[-1]
  hidden_node_num = 2
  # allen config
  input_bias = 0.35 # 1
  hidden_bias = 0.6 # 1
  # add bias weight

  # first layer is first hidden layer, the last layer is output layer
  N = len(layer_node_num) - 1 # hidden_and_output_layer_num
  # layer_node_weight = [np.random.rand(layer_node_num[i + 1], layer_node_num[i] + 1) for i in range(N)]
  layer_node_weight = [np.insert(np.matrix(H_W if i == 0 else O_W ).reshape((2, 2)), 0, 1, axis=1) for i in range(N)]
  
  layer_node_v = [np.zeros((layer_node_num[i + 1], 1)) for i in range(N)]
  layer_node_value = [np.zeros((layer_node_num[i + 1], 1)) for i in range(N)]

  layer_node_v_new = [np.zeros((layer_node_num[i + 1], 1)) for i in range(N)]
  layer_node_value_new = [np.zeros((layer_node_num[i + 1], 1)) for i in range(N)]


  print()

  err = np.zeros((output_node_num, 1))

  # f = open(file_path, "a")
  err_for_plot = []


  if len(data_list[0]) != input_node_num + output_node_num:
    raise Exception('input_value and desired_value length not match layer_node_num')


  for train_epoch_index in range(train_epochs):
    print('------------------  ', train_epoch_index, '  ------------------')
    # print('train_epoch_index', train_epoch_index)
    for  data_index, data_item in enumerate(data_list):
      # one train batch
      # print('input_hidden_weight', input_hidden_weight)

      input_value = data_item[0: -output_node_num]
      input_extend = np.matrix([input_bias] + input_value).T
      # desired_value = [ e if e > 0 else 0 for i, e in enumerate(data_item[-output_node_num: ])]
      desired_value = data_item[-output_node_num: ]
      print()

      # forward phase, input -> output
      for i in range(N):
        # print('layer_node_weight[i]',i , ' weight shape', layer_node_weight[i].shape, ' layer_node_value[i - 1] shape ', layer_node_value[i - 1].shape)
        layer_node_v[i] = layer_node_weight[i] * ((np.insert(layer_node_value[i - 1], 0, hidden_bias, axis=0)) if i != 0 else input_extend)
        layer_node_value[i] = activation_function(layer_node_v[i])
        print()
      
      output_node_value = layer_node_value[-1]
      err = np.matrix(desired_value).T - output_node_value

      # if data_index % 100 == 99:
      #   err_for_plot.append((0.5 * np.multiply(err, err)).sum(axis=0)[0,0])
      # print('err', err)


      # backward phase, output -> input
      # local_gradient(layer_node_v[i]) also equals to layer_node_value[i] * (1 - layer_node_value[i])
      node_delta = [None for i in range(N)]
      reverse_index = [i for i in range(N)]
      reverse_index.reverse()
      for i in reverse_index:
        node_local_gradient = local_gradient(layer_node_v[i])
        if i == N - 1:
          node_delta[i] = np.multiply(node_local_gradient,  err)
        else:
          node_delta[i] = np.multiply(
            node_local_gradient,
            (layer_node_weight[i - 1].T * node_delta[i - 1])[1:]
          )
      # node_delta.reverse()

      node_weight_adjustment = [None for i in range(N)]
      for i in range(N):
        if i == 0:
          node_weight_adjustment[i] = ETA * (1) * node_delta[i] * (input_extend.T)
        else:
          node_weight_adjustment[i] = ETA * (1) * node_delta[i] * np.insert(layer_node_value[i - 1], 0, 1, axis=0).T
      print()
      # layer_node_weight is a python list,'+' will concat the two lists
      # layer_node_weight = layer_node_weight + node_weight_adjustment
      for i in range(N):
        layer_node_weight[i] = layer_node_weight[i] + node_weight_adjustment[i]
      # NeuralNetwork did not update bias weight
      # if update bias weight, the result will be better
      for i in range(N):
        layer_node_weight[i][:, 0] = 1

      print()
      # to compare with NeuralNetwork's error 
      for i in range(N):
        # print('layer_node_weight[i]',i , ' weight shape', layer_node_weight[i].shape, ' layer_node_value[i - 1] shape ', layer_node_value[i - 1].shape)
        layer_node_v_new[i] = layer_node_weight[i] * ((np.insert(layer_node_value[i - 1], 0, hidden_bias, axis=0)) if i != 0 else input_extend)
        layer_node_value_new[i] = activation_function(layer_node_v_new[i])

      
      output_node_value_new = layer_node_value_new[-1]
      err = np.matrix(desired_value).T - output_node_value_new
      err_for_plot.append((0.5 * np.multiply(err, err)).sum(axis=0)[0,0])

  return err_for_plot
  # return input_hidden_weight, hidden_output_weight

def test_train():
  clear()

  # test_data  = double_moon(2, 500, '01')
  # new_data = [
  #   [e[0] / 10, e[1] / 10, e[2], e[3]] for i, e in enumerate(test_data['all'])
  # ]
  # print('test_data', test_data)
  data_item = [0.05, 0.1, 0.01, 0.99]
  err_list = train([data_item])
  # x = [i for i in range(len(err_list))]
  # plt.plot(x, err_list)
  # plt.show()
  print(err_list)

  
  # dot1 = plt.scatter(
  #   [item[0] for item in a],
  #   [item[1] for item in a],
  #   label='class 1',
  #   s=1
  # )
  # dot2 = plt.scatter(
  #   [item[0] for item in b],
  #   [item[1] for item in b],
  #   label='class 2',
  #   s=1
  # )
  
  return

if __name__ == '__main__':

  clear()
  # test_data  = double_moon(-2, 5000)
  # for i, ele in enumerate(test_data['all']):
  #   ele.append(0)
  # print(test_data)
  
  #   train(test_data)
  test_train()
