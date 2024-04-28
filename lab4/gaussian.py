import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn


def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    (sign, det) = np.linalg.slogdet(C)
    inv = np.linalg.inv(C)
    res = -M / 2 * np.log(2 * np.pi) - 0.5 * det - 0.5 * (x - mu).T.dot(inv).dot((x - mu))


#x = np.zeros((5, 1))
x = np.array([[1], [2], [3], [4], [5]])
mu = x.mean(1).reshape(x.shape[0], 1)
centeredData = x - mu
covarianceMatrix = (centeredData @ centeredData.T) / float(x.shape[1])

logpdf_GAU_ND(x, mu, covarianceMatrix)
