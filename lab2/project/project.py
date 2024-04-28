import numpy as np
import matplotlib.pyplot as plt


def load(name):
    lista = []
    category = []
    f = open(name, "r")
    for line in f:
        line = line.strip().split(",")
        lista.append(np.array([[float(line[0])], [float(line[1])], [float(line[2])], [float(line[3])], [float(line[4])], [float(line[5])]]))
        category.append(int(line[6]))
    f.close()
    return np.hstack(lista), np.hstack(category)

def histogram(num):
    if (num==0):
        string="First"
    elif (num==1):
        string="Second"
    elif (num==2):
        string="Third"
    elif (num==3):
        string="Fourth"
    elif (num==4):
        string="Fifth"
    elif (num==5):
        string="Sixth"
        
    plt.hist( trueReading[num], bins=100, density=False, ec="black", color="green", alpha=0.5)
    plt.hist(falseReading[num], bins=100, density=False, ec="black", color="red",   alpha=0.5)
    plt.xlabel(""+string+" feature")
    
    plt.show()

def scatterPlot(num):
        
        if (num==0):
            string="First"
        elif (num==1):
            string="Second"
        elif (num==2):
            string="Third"
        elif (num==3):
            string="Fourth"
        elif (num==4):
            string="Fifth"
        elif (num==5):
            string="Sixth"
    
        if(num != 0):
            #SCATTER PLOT num, first 
            plt.scatter(trueReading[num],trueReading[0],color="green", alpha=0.6)
            plt.scatter(falseReading[num],falseReading[0],color="red", alpha=0.6)
            plt.xlabel(""+string+" feature")
            plt.ylabel("First feature")
            plt.show()
        
        if(num != 1):
            #SCATTER PLOT num, second 
            plt.scatter(trueReading[num],trueReading[1],color="green", alpha=0.6)
            plt.scatter(falseReading[num],falseReading[1],color="red", alpha=0.6)
            plt.xlabel(""+string+" feature")
            plt.ylabel("Second feature")
            plt.show()

        if(num != 2):
            #SCATTER PLOT num, third 
            plt.scatter(trueReading[num],trueReading[2],color="green", alpha=0.6)
            plt.scatter(falseReading[num],falseReading[2],color="red", alpha=0.6)
            plt.xlabel(""+string+" feature")
            plt.ylabel("Third feature")
            plt.show()
            
        if(num != 3):
            #SCATTER PLOT num, fourth 
            plt.scatter(trueReading[num],trueReading[3],color="green", alpha=0.6)
            plt.scatter(falseReading[num],falseReading[3],color="red", alpha=0.6)
            plt.xlabel(""+string+" feature")
            plt.ylabel("Fourth feature")
            plt.show()
            
        if(num!= 4):
            #SCATTER PLOT num, fifth 
            plt.scatter(trueReading[num],trueReading[4],color="green", alpha=0.6)
            plt.scatter(falseReading[num],falseReading[4],color="red", alpha=0.6)
            plt.xlabel(""+string+" feature")
            plt.ylabel("Fifth feature")
            plt.show()
            
        if(num!= 5):
            #SCATTER PLOT num, sixth 
            plt.scatter(trueReading[num],trueReading[5],color="green", alpha=0.6)
            plt.scatter(falseReading[num],falseReading[5],color="red", alpha=0.6)
            plt.xlabel(""+string+" feature")
            plt.ylabel("Sixth feature")
            plt.show()
        
        
        

[a, b] = load("trainData.txt")


trueReading = a[:, b == 1]
falseReading = a[:, b == 0]

####################################################################

# HISTOGRAM first feature
histogram(0)
#SCATTER first feature
scatterPlot(0)

# HISTOGRAM second feature
histogram(1)
#SCATTER second feature
scatterPlot(1)

# HISTOGRAM third feature
histogram(2)
#SCATTER third feature
scatterPlot(2)

# HISTOGRAM fourth feature
histogram(3)
#SCATTER fourth feature
scatterPlot(3)

# HISTOGRAM fifth feature
histogram(4)
#SCATTER fifth feature
scatterPlot(4)

# HISTOGRAM sixth feature
histogram(5)
#SCATTER sixth feature
scatterPlot(5)

###########################################################################
########################     STATISTICS   #################################
 
mu = a.mean(1).reshape(a.shape[0], 1)
centeredData = a - mu

covarianceMatrix = (centeredData @ centeredData.T) / float(a.shape[1])

variance = a.var(1)
varianceTrue = trueReading.var(1)
varianceFalse = falseReading.var(1)

meanTrue = trueReading.mean(1)
meanFalse = falseReading.mean(1)

print("Mean of the true readings: "+str(meanTrue))
print("Mean of the false readings: "+str(meanFalse))

print("Variance of the true readings: "+str(varianceTrue))
print("Variance of the false readings: "+str(varianceFalse))






