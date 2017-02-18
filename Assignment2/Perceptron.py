"""
Extending the linear regression algorithm to compute the error curve
and find a good learning rate.
"""
import matplotlib.pyplot as plt
import numpy as np
import main as mn

global ETA
ETA = 0.01


class Perceptron:
    """Question 2."""

    def __init__(self, parent=None):
        """Start."""
        # Step 1: Reading the data from the csv file
        x, y, n, m = mn.read_csv('Q1a.csv')

        # Take some random weights with the size(number of attributes+1)
        # w = np.random.uniform(low=0.5, high=13.3, size=(m+1))
        w = [0.45, 0.862, -2.15]
        w = np.array(w)
        w = self.perceptron_computation(x, y, w)
        self.plot_graphs(x, y, w)

    def perceptron_computation(self, x, y, w):
        """Perceptron computation.

        Compute ywx at each iteration
        and update the weights accordingly.
        """
        xt = np.transpose(x)
        flag = True

        # Update the weights and compute till ywx is not < 0
        while flag:
            flag = False
            ywx = np.multiply(y, np.dot(w, xt))
            yx = np.zeros(np.shape(x))

            for i in range(len(yx[0])):
                sums = 0
                for j in range(len(yx)):
                    if ywx[j] < 0:
                        yx[j][i] = ETA * x[j][i] * y[j]
                        flag = True
                    sums = sums + (yx[j][i])
                w[i] = w[i] + sums
        return w

    def plot_graphs(self, x, y, w):
        """Plot the graphs.

        Compute the points(x1, x2) for the resulting line:
        "w2x2 + w1x1 + w0x0 = 0"
        """
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


if __name__ == "__main__":
    Perceptron()
