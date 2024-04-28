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
print(a)

#compute the mean and center the data
mu = a.mean(1).reshape(a.shape[0], 1)
centeredData = a - mu

setosa = a[:, b == 0]
versicolor = a[:, b == 1]
virginica = a[:, b == 2]

####################################################################

# HISTOGRAM sepal length
plt.hist(setosa[0], bins=10, density=True, ec="white", color="black", alpha=0.6)
plt.hist(versicolor[0], bins=10, density=True, ec="black", color="yellow", alpha=0.6)
plt.hist(virginica[0], bins=10, density=True, ec="black", color="red", alpha=0.6)
plt.show()

#SCATTER PLOT sepal lenght, sepal width 
plt.scatter(setosa[0],setosa[1])
plt.scatter(versicolor[0],versicolor[1],color="red")
plt.scatter(virginica[0],virginica[1],color="black")
plt.xlabel("Sepal lenght")
plt.ylabel("Sepal width")
plt.show()

#SCATTER PLOT sepal lenght, petal lenght
plt.scatter(setosa[0],setosa[2])
plt.scatter(versicolor[0],versicolor[2],color="red")
plt.scatter(virginica[0],virginica[2],color="black")
plt.xlabel("Sepal lenght")
plt.ylabel("Petal lenght")
plt.show()

#SCATTER PLOT sepal lenght, petal width
plt.scatter(setosa[0],setosa[3])
plt.scatter(versicolor[0],versicolor[3],color="red")
plt.scatter(virginica[0],virginica[3],color="black")
plt.xlabel("Sepal lenght")
plt.ylabel("Petal width")
plt.show()

###########################################################################


# HISTOGRAM sepal width
plt.hist(setosa[1], bins=10, density=True, ec="black", color="green", alpha=0.6)
plt.hist(versicolor[1], bins=10, density=True, ec="black", color="yellow", alpha=0.6)
plt.hist(virginica[1], bins=10, density=True, ec="black", color="red", alpha=0.6)
plt.show()

#SCATTER PLOT sepal width, sepal lenght 
plt.scatter(setosa[1],setosa[0])
plt.scatter(versicolor[1],versicolor[0],color="red")
plt.scatter(virginica[1],virginica[0],color="black")
plt.xlabel("Sepal width")
plt.ylabel("Sepal lenght")
plt.show()

#SCATTER PLOT sepal width, petal lenght 
plt.scatter(setosa[1],setosa[2])
plt.scatter(versicolor[1],versicolor[2],color="red")
plt.scatter(virginica[1],virginica[2],color="black")
plt.xlabel("Sepal width")
plt.ylabel("Petal lenght")
plt.show()

#SCATTER PLOT sepal width, petal width 
plt.scatter(setosa[1],setosa[3])
plt.scatter(versicolor[1],versicolor[3],color="red")
plt.scatter(virginica[1],virginica[3],color="black")
plt.xlabel("Sepal width")
plt.ylabel("Petal width")
plt.show()

#########################################################################


#HISTOGRAM  petal length
plt.hist(setosa[2], bins=10, density=True, ec="black", color="green", alpha=0.6)
plt.hist(versicolor[2], bins=10, density=True, ec="black", color="yellow", alpha=0.6)
plt.hist(virginica[2], bins=10, density=True, ec="black", color="red", alpha=0.6)
plt.show()

#SCATTER PLOT petal lenght, sepal lenght 
plt.scatter(setosa[2],setosa[0])
plt.scatter(versicolor[2],versicolor[0],color="red")
plt.scatter(virginica[2],virginica[0],color="black")
plt.xlabel("Petal lenght")
plt.ylabel("Sepal lenght")
plt.show()

#SCATTER PLOT petal lenght, sepal width 
plt.scatter(setosa[2],setosa[1])
plt.scatter(versicolor[2],versicolor[1],color="red")
plt.scatter(virginica[2],virginica[1],color="black")
plt.xlabel("Petal lenght")
plt.ylabel("Sepal width")
plt.show()

#SCATTER PLOT petal lenght, petal width 
plt.scatter(setosa[2],setosa[3])
plt.scatter(versicolor[2],versicolor[3],color="red")
plt.scatter(virginica[2],virginica[3],color="black")
plt.xlabel("Petal lenght")
plt.ylabel("Petal width")
plt.show()

#####################################################################


# HISTOGRAM petal width
plt.hist(setosa[3], bins=10, density=True, ec="black", color="green", alpha=0.6)
plt.hist(versicolor[3], bins=10, density=True, ec="black", color="yellow", alpha=0.6)
plt.hist(virginica[3], bins=10, density=True, ec="black", color="red", alpha=0.6)
plt.show()

#SCATTER PLOT petal width, sepal lenght 
plt.scatter(setosa[3],setosa[0])
plt.scatter(versicolor[3],versicolor[0],color="red")
plt.scatter(virginica[3],virginica[0],color="black")
plt.xlabel("Petal width")
plt.ylabel("Sepal lenght")
plt.show()

#SCATTER PLOT petal width, sepal width 
plt.scatter(setosa[3],setosa[1])
plt.scatter(versicolor[3],versicolor[1],color="red")
plt.scatter(virginica[3],virginica[1],color="black")
plt.xlabel("Petal width")
plt.ylabel("Sepal width")
plt.show()

#SCATTER PLOT petal width, petal lenght 
plt.scatter(setosa[3],setosa[2])
plt.scatter(versicolor[3],versicolor[2],color="red")
plt.scatter(virginica[3],virginica[2],color="black")
plt.xlabel("Petal width")
plt.ylabel("Petal lenght")
plt.show()

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


print("Variance of the whole matrix: "+ str(var))
print("Standard deviation of the whole matrix: "+str(std))  
print("Variance of the setosa class: "+ str(varianceSetosa))
print("Variance of the versicolor class: "+ str(varianceVersicolor))
print("Variance of the virginica class: "+ str(varianceVirginica))
print("\n Mean of the whole matrix: "+str(mu))
print("Mean of the setosa class: "+str(setosaMean))
print("Mean of the versicolor class: "+str(versicolorMean))
print("Mean of the veriginica class: "+str(virginicaMean))



