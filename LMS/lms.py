import math;
import numpy as np;
import random;
import matplotlib.pyplot as plt;


# m = np.matrix('1 3; 4 2');
# print(m.I);

# for index in range(100):
#   # 0 ~ 99
#   temp = math.floor(random.random() * 100)
#   if (temp < 1):
#     print(temp)
#   elif (temp > 98) :
#     print(temp)

order = 3 #阶数 x的最高次幂
num = order + 1 #X的列数

def randomNoise(base = 10):
  # -base  ~  +base 
  return  random.random() * base * 2 - base

def calculation(x):
  # y = (x-2)(x-6)
  # y = x^2 - 8 * x + 16
  return x * x - 8 * x + 16


def generateData(num = 10):
  tempList = [(random.random() * num) for i in range(num * 2)]
  xList = list(set(tempList))[0:num]
  xList.sort()
  yList = [calculation(x) + randomNoise(4) for x in xList]
  # x = np.arange(-1,1,0.02)
  # y = [((a*a-1)**3 + (a-0.5)**2 + 3*np.sin(2*a)) for a in x]
  # xList = [b1*(random.randint(90,120))/100 for b1 in x]
  # yList = [b2*(random.randint(90,120))/100 for b2 in y]
  # print(len(newList))
  return {'xList': xList, 'yList': yList}

# plt.plot(x,y)
# plt.show()


def lms(X, Y, order):

  # array_x =[[0 for i in range(order+1)] for i in range(len(X))]
  # #对二维数组赋值
  # for i in range(0,order+1):
  #     for j in range(0,len(X)):
  #         array_x[j][i] = X[j]**i

  #对二维数组赋值
  xFormat = [[math.pow(xItem, i) for i in range(order + 1)] for xItem in X]
  # xFormat = [[math.pow(xItem, i) for i in range(2)] for xItem in X]
  # print('xFormat', xFormat)
  # print('xFormat === array_x', xFormat[4][6] == array_x[4][6])
  x = np.matrix(np.array(xFormat))
  # print('x shape', x.shape)
  y = np.matrix(np.array(Y)).T
  # print('y shape', y.shape)
  fgergd = x.T.dot(x)
  # print('fgergd shape', fgergd.shape)
  dfsdfs = np.linalg.pinv(fgergd)
  dgfdghdrye = dfsdfs.dot(x.T)
  A = dgfdghdrye.dot(y)
  # A = (x.T * x).I * (x.T) * (y)
  # print('A shape', A)
  # a = reverse(a * transpose(a)) * transpose(a) * y
  # return [ item[0] for item in np.asarray(A).tolist()]
  return A

def ySimulate(X, A):
  xFormat = [[math.pow(xItem, i) for i in range(order + 1)] for xItem in X]
  x = np.matrix(xFormat)
  a = np.matrix(A)
  temp = x.dot(a)
  print('temp',temp)
  return [ item[0] for item in np.asarray(temp).tolist()]

# def test(a):
#   X = [1, 3, 2]
#   b = [[math.pow(xItem, i) for i in range(len(X))] for xItem in X]
#   print(b)
#   # print(a or 1)
#   # X = np.matrix('1 3; 4 2');
#   # Y = [[1, 0], [0, 1]];
#   # Xt = X.T
#   # XtX = Xt.dot(X)
#   # a = np.matrix(XtX).I.dot(Xt).dot(Y)
#   # print(a)
#   return a
# test(None)

allList = generateData();
# print(allList)
xList = allList['xList'];
yList = allList['yList'];

aList  = lms(xList, yList, order)
print('aList shape', aList)

ySimList = ySimulate(xList, aList)
# print('ySimulate shape', ySimList)


diff = [ySimList[i] - yList[i] for i in range(10)]
print('diff shape', diff)


plt.plot(xList,yList, label='www')
plt.plot(xList,ySimList, label='eee')
# plt.plot(xList, diff, label='fs')
plt.show()
# print(randomNoise(10))

