import numpy as np
import matplotlib.pyplot as plt


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
centeredData = a - mu

setosa = a[:, b == 0]
versicolor = a[:, b == 1]
virginica = a[:, b == 2]

###########################################################################
#######################     STATISTICS   #################################

covarianceMatrix = (centeredData @ centeredData.T) / float(a.shape[1])

var = a.var(1)
std = a.std(1)

varianceSetosa = setosa.var(1)
varianceVersicolor = versicolor.var(1)
varianceVirginica = virginica.var(1)

setosaMean = setosa.mean(1)
versicolorMean = versicolor.mean(1)
virginicaMean = virginica.mean(1)

# print("Variance of the whole matrix: " + str(var))
# print("Standard deviation of the whole matrix: " + str(std))
# print("Variance of the setosa class: " + str(varianceSetosa))
# print("Variance of the versicolor class: " + str(varianceVersicolor))
# print("Variance of the virginica class: " + str(varianceVirginica))
# print("\n Mean of the whole matrix: " + str(mu))
# print("Mean of the setosa class: " + str(setosaMean))
# print("Mean of the versicolor class: " + str(versicolorMean))
# print("Mean of the veriginica class: " + str(virginicaMean))

#print("Covariance Matrix" +str(covarianceMatrix))
#print("Mean of the matrix"+ str(mu))

s, U = np.linalg.eigh(covarianceMatrix)
P = U[:, ::-1][:, 0:3]
U2, s2, Vh = np.linalg.svd(covarianceMatrix)
P2 = U2[:, 0:3]
DP = np.dot(P.T, a)
#print("U2: \n"+str(U2))
#print("P: \n"+str(P))

# plt.hist(DP[0], bins=10, density=True, ec="white", color="red", alpha=0.6)
# plt.show()
# plt.hist(DP[1], bins=10, density=True, ec="white", color="red", alpha=0.6)
# plt.show()
# plt.hist(DP[2], bins=10, density=True, ec="white", color="red", alpha=0.6)
# plt.show()
# print("DP: \n"+str(DP))
# tr = np.load("IRIS_PCA_matrix_m4.npy")
# print("matrice del prof: \n"+str(tr))

def pca(m, dataset):
    [data, bi] = load(dataset)
    mean = data.mean(1).reshape(data.shape[0], 1)
    cData = a - mean
    covarMatrix = (cData @ cData.T) / float(data.shape[1])
    s1, U1 = np.linalg.eigh(covarMatrix)
    P1 = U[:, ::-1][:, 0:m]
    #you can find P with SVD too
    #U2, s2, Vh = np.linalg.svd(covarianceMatrix)
    #P2 = U2[:, 0:3]
    y = np.dot(P1.T, data)
    plt.scatter(y[0], y[1])
    plt.show()


pca(3, "iris.csv")






