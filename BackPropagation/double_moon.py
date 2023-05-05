import math;
import numpy as np;
import random;
import matplotlib.pyplot as plt;


def double_moon(class_distance = 0, times = 10, output_range = '01'):
  # class 1
  # (10, 0) range 7 ~ 10
  class_a = []
  for i in range(times):
    x = random.random() * 20
    y = random.random() * 10
    dist = math.dist([x, y], [10, 0])
    if (dist <= 10 and dist >= 7):
      if (output_range == '01'):
        class_a.append([x, y, 1, 0])
      else:
        class_a.append([x, y, 1])
      
  # class 2
  # (20, 4) range 7 ~ 10
  class_b = []
  for j in range(times):
    x = random.random() * 20 + 10
    y = 0 - class_distance - random.random() * 10
    dist = math.dist([x, y], [20, 0 - class_distance])
    if (dist <= 10 and dist >= 7):
      if (output_range == '01'):
        class_a.append([x, y, 0, 1])
      else:
        class_a.append([x, y, 0])

  # a = [1]
  # b = [2]
  # print(a + b)
  all = class_a + class_b
  random.shuffle(all)
  return {
    'class_a': class_a,
    'class_b': class_b,
    'all': all
  }
if __name__ == "__main__":
  # print('get_distance', math.dist([0, 0], [1, 1]))
  data = double_moon(-5, 1000)
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
  # l2 = plt.plot(xList,ySimList, label='eee')
  plt.legend()
  plt.show()

  

