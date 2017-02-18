import matplotlib.pyplot as plt
import csv
import numpy as np

# Step 1: Reading the data from the csv file
n = 0
y = []
x = []
with open('regdata.csv', 'r') as f:
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
kappa = 1

x = np.c_[x, np.ones(n)]
x = [[float(j) for j in i] for i in x]
x = np.array(x)
y = np.array(y)
w = np.array(w)

# Step 2: Scaling the attributes
xmean = x.mean(axis=0)
ymean = y.mean()
xmin = np.amin(x, axis=0)
ymin = np.amin(y)
xmax = np.amax(x, axis=0)
ymax = np.amax(y)

for i in range(len(x[0]) - 1):
    for j in range(len(x)):
        x[j][i] = (x[j][i] - xmean[i]) / (xmax[i] - xmin[i])

for j in range(len(y)):
    y[j] = (y[j] - ymean) / (ymax - ymin)


# Step 3: Compute the error at each iteration and save the error values in vector
xt = np.transpose(x)
error_vector = []
w_old = np.array([])

while True:
    if np.array_equal(np.around(w, decimals=3), np.around(w_old, decimals=3)):
        break
    error = (1.0 / 2.0 * n) * np.sum((y - np.dot(w, xt))**2)
    error_vector.append(error)
    nabla = -(1.0 / n) * np.sum(np.multiply(xt, y - np.dot(w, xt)), axis=1)
    # Update the weights
    w_old = w
    w = w - (kappa * nabla)
    print nabla, error, w

print "Final weights: ", w

# Step 4: Plot the error vector as a curve in the end
plt.plot(error_vector)
plt.ylabel('Error Vector')
plt.show()

# Step 5: Find a good learning rate based on the error curve.
