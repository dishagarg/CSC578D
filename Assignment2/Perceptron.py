import matplotlib.pyplot as plt
import csv
import numpy as np

# Read the data from csv
n = 0
y = []
x = []
with open('Q1a.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        n = n + 1
        m = len(row) - 1
        y.append(float(row[-1]))
        x.append(row[:-1])
f.close()

# Take some random weights with the size(number of attributes+1)
# w = np.random.uniform(low=0.5, high=13.3, size=(m+1))
w = [0.45, 0.862, -2.15]
eta = 0.01

x = np.c_[x, np.ones(n)]
x = [[float(j) for j in i] for i in x]
x = np.array(x)
y = np.array(y)
w = np.array(w)

xt = np.transpose(x)
flag = True

# Update the weights and compute till ywx is not < 0
while(flag):
    flag = False
    ywx = np.multiply(y, np.dot(w, xt))
    yx = np.zeros(np.shape(x))

    for i in range(len(yx[0])):
        sums = 0
        for j in range(len(yx)):
            if ywx[j] < 0:
                yx[j][i] = eta * x[j][i] * y[j]
                flag = True
            sums = sums + (yx[j][i])
        w[i] = w[i] + sums


# Compute the points(x1, x2) for the resulting line: "w2x2 + w1x1 + w0x0 = 0"
L = np.arange(np.max(x) + 2)
R = (-w[-1] - (w[0] * L)) / w[1]
plt.plot(R)
for i in range(len(y)):
    if y[i] < 0:
        plt.plot(x[i, 0], x[i, 1], 'ro')
    else:
        plt.plot(x[i, 0], x[i, 1], 'b^')

plt.ylabel('x[1]=x1')
plt.xlabel('x[0]=x2')
plt.show()
