"""
Implementing the perceptron algorithm.
"""
import matplotlib.pyplot as plt
import numpy as np
import main as mn

global KAPPA
KAPPA = 1


class Regression:
    """Question 2."""

    def __init__(self, parent=None):
        """Start."""
        # Step 1: Reading the data from the csv file
        x, y, self.n, m = mn.read_csv('regdata.csv')

        # Take some random weights with the size(number of attributes+1)
        # w = np.random.uniform(low=0.5, high=13.3, size=(m+1))
        w = [0.45, 0.862, -2.15]
        w = np.array(w)

        # Step 2: Scaling the attributes.
        x, y = self.scale_attributes(x, y)

        # Step 3: Compute the error at each iteration and save the error values in vector.
        error_vector, w = self.regression_computation(x, y, w)
        print "Final weights: ", w

        # Step 4: Plot the error vector as a curve in the end.
        self.plot_graphs(error_vector)

        # Step 5: Find a good learning rate based on the error curve.
        # kappa = 1 is a good learning rate based on this error curve.
        # As, it is neither too big nor too small.

    def scale_attributes(self, x, y):
        """Scaling the attributes."""
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
        return x, y

    def regression_computation(self, x, y, w):
        """Linear regression computation.

        Compute the error at each iteration
        and update the weights accordingly.
        """
        xt = np.transpose(x)
        error_vector = []
        w_old = np.array([])

        # Loop until the weights(w) do not change too much
        while True:
            if np.array_equal(np.around(w, decimals=3), np.around(w_old, decimals=3)):
                break
            error = (1.0 / 2.0 * self.n) * np.sum((y - np.dot(w, xt))**2)
            error_vector.append(error)
            nabla = -(1.0 / self.n) * np.sum(np.multiply(xt, y - np.dot(w, xt)), axis=1)
            # Update the weights
            w_old = w
            w = w - (KAPPA * nabla)
            print nabla, error, w
        return error_vector, w

    def plot_graphs(self, vector):
        """Plot the graphs."""
        plt.plot(vector)
        plt.ylabel('Error Vector')
        plt.show()


if __name__ == "__main__":
    Regression()
