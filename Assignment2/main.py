"""Helper functions to read and write data to txt or csv files."""
import csv
import numpy as np


def read_csv(file):
    """Reading data from csv."""
    n = 0
    y = []
    x = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            n = n + 1
            m = len(row) - 1
            y.append(float(row[-1]))
            x.append(row[:-1])
    x = np.c_[x, np.ones(n)]
    x = [[float(j) for j in i] for i in x]
    x = np.array(x)
    y = np.array(y)
    return x, y, n, m


def read_txt(file, text):
    """Reading data from text files."""
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                text.append(item)
    return text
