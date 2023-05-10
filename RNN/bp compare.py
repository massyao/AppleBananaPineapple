import math;
import numpy as np;
import random;
import matplotlib.pyplot as plt;
import matplotlib.pylab as pylab;
# from matplotlib.pylab import *
# from perceptron.perceptron import Perceptron
from double_moon import *
# from utils import *
import os



absolute_path = os.path.dirname(__file__)
# file_path = os.path.join(absolute_path, 'log.txt')

# use double moon as data 
# inputs number 2
# out puts number 1
# hidden layer nodes set to 5


train_times = 5

ETA = 0.5

def activation_function(x):
  # return x if x > 0 else 0

  # try:
  #   return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
  # except Exception:
  #   return 0
  # x = xx / 100
  # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
  return 1 / (1 + np.exp(-x))

# (cosh x) ^(-2)
def local_gradient(x):
  # x = xx / 100
  # return 4 * np.exp(2 * x) / (np.exp(4 * x) + 2 * np.exp(2 * x) + 1)
  f = activation_function(x)
  if type(x) == type(0):
    return f * (1 - f)
  elif isinstance(x, np.ndarray):
    return np.multiply(f, 1 - f)



def train(
    data_list,
    input_node_num = 2,
    output_node_num = 2,
    hidden_node_num = 2
  ):

  # NeuralNetwork config
  input_hidden_weight_const = [0.15, 0.2, 0.25, 0.3]
  hidden_output_weight_const = [0.4, 0.45, 0.5, 0.55]
  input_bias = 0.35 # 1
  hidden_bias = 0.6 # 1

  input_hidden_weight = np.insert(np.matrix(input_hidden_weight_const).reshape((2, 2)), 0, 1, axis=1) # np.random.rand(hidden_node_num, input_node_num + 1)
  hidden_output_weight = np.insert(np.matrix(hidden_output_weight_const).reshape((2, 2)), 0, 1, axis=1)# np.random.rand(output_node_num, hidden_node_num + 1)


  # allen config
  # input_bias = 0 # 1
  # hidden_bias = 0 # 1
  # # add bias weight
  # input_hidden_weight = np.random.rand(hidden_node_num, input_node_num + 1)
  # hidden_output_weight = np.random.rand(output_node_num, hidden_node_num + 1)

  hidden_node_v = np.zeros((hidden_node_num, 1))
  hidden_node_value = np.zeros((hidden_node_num, 1))
  
  output_node_v = np.zeros((output_node_num, 1))
  output_node_value = np.zeros((output_node_num, 1))

  err = np.zeros((output_node_num, 1))

  w_delta_hidden = np.random.rand(output_node_num, 1)
  w_delta_input = np.random.rand(hidden_node_num, 1)

  # f = open(file_path, "a")

  for train_times_index in range(train_times):
    print('----------------------------')
    # print('train_times_index', train_times_index)
    for  data_index, data_item in enumerate(data_list):

      # print('input_hidden_weight', input_hidden_weight)

      input_value = data_item[0: -output_node_num]
      # class_value = [ e if e > 0 else 0 for i, e in enumerate(data_item[-output_node_num: ])]
      class_value = data_item[-output_node_num: ]

      
      # forward phase, input -> hidden
      input_extend = np.matrix([input_bias] + input_value).T
      hidden_node_v = input_hidden_weight * input_extend
      hidden_node_value = activation_function(hidden_node_v)

      # forward phase, hidden -> output
      output_node_v = (hidden_output_weight) * np.insert(hidden_node_value, 0, hidden_bias).T
      output_node_value = activation_function(output_node_v)

      # calculate error
      # notice nn use sum(error) as error
      # notice: err should be 0.5 * np.multiply(err_temp, err_temp)
      err = np.matrix(class_value).T - output_node_value
      # print('class_value ', class_value, ' output_node_value ', output_node_value.T.tolist()[0])
      # use new weight to calculate error



      # backward phase, output -> hidden
      local_gradient_output_node_v = local_gradient(output_node_v)
      delta_output = np.multiply(local_gradient_output_node_v, err)
      delta_w = ETA * (1) * delta_output * np.insert(hidden_node_value, 0, 1)
      # hidden_output_weight = hidden_output_weight + delta_w
      hidden_output_weight_new = hidden_output_weight + delta_w
      # print()

      # backward phase, hidden -> input
      delta_hidden_sum = (hidden_output_weight.T * delta_output)[1:]
      delta_input = np.multiply(local_gradient(hidden_node_v), delta_hidden_sum)
      # notice delta_w_input should be delta_input * input_value
      # delta_w_input = ETA *  np.multiply(delta_input, hidden_node_value)
      delta_w_input = ETA * (1) * delta_input * np.matrix(input_value)

      # update weight
      hidden_output_weight = hidden_output_weight_new
      # input_hidden_weight = input_hidden_weight + delta_w_input * (np.matrix([input_bias] + input_value))
      input_hidden_weight = input_hidden_weight + np.insert(delta_w_input, 0, 0, axis=1)

      # NeuralNetwork did not update bias weight
      # if update bias weight, the result will be better
      # input_hidden_weight[:, 0] = 1
      # hidden_output_weight[:, 0] = 1

      # print('allen input_hidden_weight ', input_hidden_weight)
      # print('allen hidden_output_weight ', hidden_output_weight)

      hidden_node_v_new = input_hidden_weight * input_extend
      hidden_node_value_new = activation_function(hidden_node_v_new)
      output_node_v_new = (hidden_output_weight) * np.insert(hidden_node_value_new, 0, hidden_bias).T
      output_node_value_new = activation_function(output_node_v_new)
      err_new = np.matrix(class_value).T - output_node_value_new
      # print('mathmatical err ',  (0.5 * np.multiply(err_new, err_new)).sum(axis=0)[0,0], ' output_node_value_new ', output_node_value_new.T.tolist()[0], ' class_value ', class_value)
      print('mathmatical err ',  (0.5 * np.multiply(err_new, err_new)).sum(axis=0)[0,0])
      # print()

  return input_hidden_weight, hidden_output_weight

def test_train():
  # clear()



  nn = NeuralNetwork(
  2, 2, 2,
  hidden_layer_weights=[0.15, 0.2, 0.25, 0.3],
  hidden_layer_bias=0.35,
  output_layer_weights=[0.4, 0.45, 0.5, 0.55],
  output_layer_bias=0.6
  )
  for i in range(train_times):
    nn.train([0.05, 0.1], [0.01, 0.99])
    # this may be an error, calculate_total_error changed the output value, (it use the new weight to calculate error)
    print('NeuralNetwork err ', nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]))

  data_item = [0.05, 0.1, 0.01, 0.99]
  train([data_item])

  print()

  
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

  # clear()
  # test_data  = double_moon(-2, 5000)
  # for i, ele in enumerate(test_data['all']):
  #   ele.append(0)
  # print(test_data)
  
  #   train(test_data)
  test_train()
