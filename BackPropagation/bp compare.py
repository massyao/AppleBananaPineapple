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



# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
class NeuralNetwork:
  LEARNING_RATE = 0.5

  def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
    self.num_inputs = num_inputs

    self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
    self.output_layer = NeuronLayer(num_outputs, output_layer_bias)
    


    self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
    self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

  def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
    weight_num = 0
    for h in range(len(self.hidden_layer.neurons)):
      for i in range(self.num_inputs):
        if not hidden_layer_weights:
          self.hidden_layer.neurons[h].weights.append(random.random())
        else:
          self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
        weight_num += 1

  def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
    weight_num = 0
    for o in range(len(self.output_layer.neurons)):
      for h in range(len(self.hidden_layer.neurons)):
        if not output_layer_weights:
          self.output_layer.neurons[o].weights.append(random.random())
        else:
          self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
        weight_num += 1

  def inspect(self):
    print('------')
    print('* Inputs: {}'.format(self.num_inputs))
    print('------')
    print('Hidden Layer')
    self.hidden_layer.inspect()
    print('------')
    print('* Output Layer')
    self.output_layer.inspect()
    print('------')

  def feed_forward(self, inputs):
    # print('NeuralNetwork feed_forward with input ', inputs)
    hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
    # print('NeuralNetwork feed_forward hidden_layer_outputs ', hidden_layer_outputs)
    # print('NeuralNetwork feed_forward outputs ', self.output_layer.feed_forward(hidden_layer_outputs))
    return self.output_layer.feed_forward(hidden_layer_outputs)

  # Uses online learning, ie updating the weights after each training case
  def train(self, training_inputs, training_outputs):
    self.feed_forward(training_inputs)

    # 1. Output neuron deltas
    pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
    for o in range(len(self.output_layer.neurons)):

      # ∂E/∂zⱼ
      pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])
    # print('NeuralNetwork feed_forward pd_errors_wrt_output_neuron_total_net_input ', pd_errors_wrt_output_neuron_total_net_input)

    # 2. Hidden neuron deltas
    pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
    for h in range(len(self.hidden_layer.neurons)):

      # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
      # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
      d_error_wrt_hidden_neuron_output = 0
      for o in range(len(self.output_layer.neurons)):
        d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]


      # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
      pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()
      # print('NeuralNetwork hidden_layer deltas ', pd_errors_wrt_hidden_neuron_total_net_input[h],' =  output layer deltas ', pd_errors_wrt_output_neuron_total_net_input , ' * hidden layer weights ', self.output_layer.neurons[o].weights[h], ' * hidden layer gradient  ', self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input())

    # 3. Update output neuron weights
    for o in range(len(self.output_layer.neurons)):
      for w_ho in range(len(self.output_layer.neurons[o].weights)):

        # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
        pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

        
        # Δw = α * ∂Eⱼ/∂wᵢ
        self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight
        # print('NeuralNetwork output_layer  delta_w_input ', self.LEARNING_RATE * pd_error_wrt_weight , ' = LEARNING_RATE * delta ',  pd_errors_wrt_output_neuron_total_net_input[o] , ' * calculate_pd_total_net_input_wrt_weight', self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho))
        # print('NeuralNetwork delta_w   ', self.LEARNING_RATE * pd_error_wrt_weight, ' = output_layer deltas ', pd_errors_wrt_output_neuron_total_net_input[o], ' * output_layer gradient ', self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho))
        # print('NeuralNetwork hidden_output_weight_new ', self.output_layer.neurons[o].weights[w_ho])
        

    # 4. Update hidden neuron weights
    for h in range(len(self.hidden_layer.neurons)):
      # print('NeuralNetwork hidden_layer ', o)
      for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
        
        # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
        pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)
        # print('NeuralNetwork hidden_layer pd_error_wrt_weight   ', self.LEARNING_RATE * pd_error_wrt_weight, ' = hidden_layer deltas ', pd_errors_wrt_hidden_neuron_total_net_input[h], ' * input_hidden_layer inputs ', self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih))
        # Δw = α * ∂Eⱼ/∂wᵢ
        self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight
        # print('NeuralNetwork input_hidden_weight_new  ', self.hidden_layer.neurons[h].weights[w_ih])

    # print('NeuralNetwork feed_forward end')


  def calculate_total_error(self, training_sets):
    total_error = 0
    for t in range(len(training_sets)):
      training_inputs, training_outputs = training_sets[t]
      # when calculating the error, the code feed forward the inputs, is it correct?
      self.feed_forward(training_inputs)
      for o in range(len(training_outputs)):
        total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
    return total_error

class NeuronLayer:
  def __init__(self, num_neurons, bias):

    # Every neuron in a layer shares the same bias
    self.bias = bias if bias else random.random()

    self.neurons = []
    for i in range(num_neurons):
      self.neurons.append(Neuron(self.bias))

  def inspect(self):
    print('Neurons:', len(self.neurons))
    for n in range(len(self.neurons)):
      print(' Neuron', n)
      for w in range(len(self.neurons[n].weights)):
        print('  Weight:', self.neurons[n].weights[w])
      print('  Bias:', self.bias)

  def feed_forward(self, inputs):
    # print('feed_forward inputs ', inputs)
    outputs = []
    for neuron in self.neurons:
      outputs.append(neuron.calculate_output(inputs))
    # print('feed_forward outputs ', outputs)
    # print('------------------------')
    return outputs

  def get_outputs(self):
    outputs = []
    for neuron in self.neurons:
      outputs.append(neuron.output)
    return outputs

class Neuron:
  def __init__(self, bias):
    self.bias = bias
    self.weights = []

  def calculate_output(self, inputs):
    self.inputs = inputs
    self.output = self.squash(self.calculate_total_net_input())
    # print('Neuron output ', self.output)
    return self.output

  def calculate_total_net_input(self):
    total = 0
    for i in range(len(self.inputs)):
      total += self.inputs[i] * self.weights[i]
    # total_net_input = total + self.bias
    # print('total_net_input', total_net_input, ' squash ', 1 / (1 + math.exp(-total_net_input)))
    return total + self.bias

  # Apply the logistic function to squash the output of the neuron
  # The result is sometimes referred to as 'net' [2] or 'net' [1]
  def squash(self, total_net_input):
    return 1 / (1 + math.exp(-total_net_input))

  # Determine how much the neuron's total input has to change to move closer to the expected output
  #
  # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
  # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
  # the partial derivative of the error with respect to the total net input.
  # This value is also known as the delta (δ) [1]
  # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
  #
  def calculate_pd_error_wrt_total_net_input(self, target_output):
    # print('NeuralNetwork err ', self.calculate_pd_error_wrt_output(target_output), ' *  local_gradient  ', self.calculate_pd_total_net_input_wrt_input())
    # print('NeuralNetwork feed_forward pd_errors_wrt_output_neuron_total_net_input ', pd_errors_wrt_output_neuron_total_net_input)
    return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

  # The error for each neuron is calculated by the Mean Square Error method:
  def calculate_error(self, target_output):
    # print('NeuronLayer calculate_error inputs', self.inputs, 'target_output ', target_output, ' self.output ', self.output, ' error is ', target_output - self.output, ' nn_error is ', 0.5 * (target_output - self.output) ** 2)
    return 0.5 * (target_output - self.output) ** 2

  # The partial derivate of the error with respect to actual output then is calculated by:
  # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
  # = -(target output - actual output)
  #
  # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
  # = actual output - target output
  #
  # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
  #
  # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
  # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
  def calculate_pd_error_wrt_output(self, target_output):
    # print('target_output', target_output, ' self.output ', self.output, '  -(target_output - self.output) ', -(target_output - self.output))
    return -(target_output - self.output)

  # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
  # yⱼ = φ = 1 / (1 + e^(-zⱼ))
  # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
  #
  # The derivative (not partial derivative since there is only one variable) of the output then is:
  # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
  def calculate_pd_total_net_input_wrt_input(self):
    return self.output * (1 - self.output)

  # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
  # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
  #
  # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
  # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
  def calculate_pd_total_net_input_wrt_weight(self, index):
    # print('NeuralNetwork  self.inputs[index] ', self.inputs[index])
    return self.inputs[index]


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------



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
  clear()



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

  clear()
  # test_data  = double_moon(-2, 5000)
  # for i, ele in enumerate(test_data['all']):
  #   ele.append(0)
  # print(test_data)
  
  #   train(test_data)
  test_train()
