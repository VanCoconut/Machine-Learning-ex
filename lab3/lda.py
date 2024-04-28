import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn


def load(name):
    lista = []
    category = []
    f = open(name, "r")
    for line in f:
        line = line.strip().split(",")
        lista.append(np.array([[float(line[0])], [float(line[1])], [float(line[2])], [float(line[3])]]))
        if line[4] == "Iris-setosa":
            category.append(0)
        if line[4] == "Iris-versicolor":
            category.append(1)
        if line[4] == "Iris-virginica":
            category.append(2)
    f.close()
    return np.hstack(lista), np.hstack(category)


[a, b] = load("iris.csv")

# compute the mean and center the data
mu = a.mean(1).reshape(a.shape[0], 1)
# centeredData = a - mu

setosa = a[:, b == 0]
versicolor = a[:, b == 1]
virginica = a[:, b == 2]

########
# WITHIN CLASS COVARIANCE
########
# compure the covariance for each class
# mSetosa is the mean of the class, you can center the data by subtracting the mean from the dataset
mSetosa = setosa.mean(1).reshape(a.shape[0], 1)
centeredDataSetosa = setosa - mSetosa
covarianceMatrixSetosa = (centeredDataSetosa @ centeredDataSetosa.T) / float(setosa.shape[1])

mVersicolor = versicolor.mean(1).reshape(a.shape[0], 1)
centeredDataVesicolor = versicolor - mVersicolor
covarianceMatrixVesicolor = (centeredDataVesicolor @ centeredDataVesicolor.T) / float(versicolor.shape[1])

mVirginica = virginica.mean(1).reshape(a.shape[0], 1)
centeredDataVirginica = virginica - mVirginica
covarianceMatrixVirginica = (centeredDataVirginica @ centeredDataVirginica.T) / float(virginica.shape[1])

# sum of all within matix

withinCovariance = covarianceMatrixSetosa + covarianceMatrixVesicolor + covarianceMatrixVirginica
# numnber of classes
nClasses = 3
withinCovariance = withinCovariance / nClasses
# print(withinCovariance)

########
# BETWEEN CLASS COVARIANCE
########

# formula taken from the slides, without nc
betweenCovarianceSetosa = ((mSetosa - mu) @ (mSetosa - mu).T)
betweenCovarianceVersicolor = ((mVersicolor - mu) @ (mVersicolor - mu).T)
betweenCovarianceVirginica = ((mVirginica - mu) @ (mVirginica - mu).T)

# sum between matrices of each class of the dataset
betweenCovariance = betweenCovarianceSetosa + betweenCovarianceVersicolor + betweenCovarianceVirginica
betweenCovariance = betweenCovariance / nClasses
# print(betweenCovariance)

###########################################################################
#################     Generalizde eigenvalue problem   ##############

# s, U = scipy.linalg.eigh(betweenCovariance, withinCovariance)
# W = U[:, ::-1][:, 0:m]
# UW, _, _ = np.linalg.svd(W)
# U = UW[:, 0:m]

######## solving the eigenvalue problem by joint diagonalization of Sb and Sw #############
U, s, _ = np.linalg.svd(withinCovariance)
P1 = np.dot(np.dot(U, np.diag(1.0 / (s ** 0.5))), U.T)
Sbt = P1 @ betweenCovariance @ P1.T
s, U = np.linalg.eigh(Sbt)
# the last number stands for the discrimiant directions
P2 = U[:, ::-1][:, :2]
# should be the LDA
y = P2.T @ P1 @ a

# Plot histograms 1DD
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
labels = ['Setosa', 'Versicolor', 'Virginica']
for i in range(3):
    plt.hist(y[0, b == i], color=colors[i], alpha=0.5, label=labels[i])

plt.title('Histogram of LDA-transformed data (Component 1)')
plt.xlabel('LDA Component 1')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
# plt.show()

# Plot histograms for Component 2DD
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.hist(y[1, b == i], color=colors[i], alpha=0.5, label=labels[i])

plt.title('Histogram of LDA-transformed data (Component 2)')
plt.xlabel('LDA Component 2')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
# plt.show()

# Plot scatter plots
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.scatter(y[1, b == i], y[1, b == i], color=colors[i], label=labels[i])

plt.title('Scatter plot of LDA-transformed data')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.legend()
plt.grid(True)
# plt.show()

# 2DD
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
labels = ['Setosa', 'Versicolor', 'Virginica']
for i in range(3):
    plt.scatter(y[0, b == i], y[1, b == i], color=colors[i], label=labels[i])

plt.title('Scatter plot of LDA-transformed data')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.legend()
plt.grid(True)
# plt.show()

###########################################################################
#################     PCA + LDA   ##############

# non ho capito cosa fa
# def load_iris():
#     return sklearn.datasets.load_iris()["data"].T, sklearn.datasets.load_iris()["target"]
#
#
# DIris, LIris = load_iris()
# D = DIris[:, LIris != 0]
# L = LIris[LIris != 0]


def split_db_2to1(d, l, seed=0):
    nTrain = int(d.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(d.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = d[:, idxTrain]
    DVAL = d[:, idxTest]
    LTR = l[idxTrain]
    LVAL = l[idxTest]
    return (DTR, LTR), (DVAL, LVAL)


dataM = a[:, a != 0]
labelM = b[b != 0]
# DTR and LTR are model training data and labels
# DVAL and LVAL are validation data and labels
(DTR, LTR), (DVAL, LVAL) = split_db_2to1(dataM, labelM)

########
# WITHIN CLASS COVARIANCE
########
# compure the covariance for each class
# mSetosa is the mean of the class, you can center the data by subtracting the mean from the dataset
mSetosa = setosa.mean(1).reshape(a.shape[0], 1)
centeredDataSetosa = setosa - mSetosa
covarianceMatrixSetosa = (centeredDataSetosa @ centeredDataSetosa.T) / float(setosa.shape[1])

mVersicolor = versicolor.mean(1).reshape(a.shape[0], 1)
centeredDataVesicolor = versicolor - mVersicolor
covarianceMatrixVesicolor = (centeredDataVesicolor @ centeredDataVesicolor.T) / float(versicolor.shape[1])

mVirginica = virginica.mean(1).reshape(a.shape[0], 1)
centeredDataVirginica = virginica - mVirginica
covarianceMatrixVirginica = (centeredDataVirginica @ centeredDataVirginica.T) / float(virginica.shape[1])

# sum of all within matix

withinCovariance = covarianceMatrixSetosa + covarianceMatrixVesicolor + covarianceMatrixVirginica
# numnber of classes
nClasses = 3
withinCovariance = withinCovariance / nClasses
# print(withinCovariance)

########
# BETWEEN CLASS COVARIANCE
########

# formula taken from the slides, without nc
betweenCovarianceSetosa = ((mSetosa - mu) @ (mSetosa - mu).T)
betweenCovarianceVersicolor = ((mVersicolor - mu) @ (mVersicolor - mu).T)
betweenCovarianceVirginica = ((mVirginica - mu) @ (mVirginica - mu).T)

# sum between matrices of each class of the dataset
betweenCovariance = betweenCovarianceSetosa + betweenCovarianceVersicolor + betweenCovarianceVirginica
betweenCovariance = betweenCovariance / nClasses
# print(betweenCovariance)

######## solving the eigenvalue problem by joint diagonalization of Sb and Sw #############
U, s, _ = np.linalg.svd(withinCovariance)
P1 = np.dot(np.dot(U, np.diag(1.0 / (s ** 0.5))), U.T)
Sbt = P1 @ betweenCovariance @ P1.T
s, U = np.linalg.eigh(Sbt)
# the last number stands for the discrimiant directions
P2 = U[:, ::-1][:, :2]
# should be the LDA
DTR_lda = P2.T @ P1 @ a

threshold = (DTR_lda[0, LTR == 1].mean() + DTR_lda[0, LTR == 2].mean()) / 2.0
# Projected samples have only 1 dimension

PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
PVAL[DVAL_lda[0] >= threshold] = 2
PVAL[DVAL_lda[0] < threshold] = 1
