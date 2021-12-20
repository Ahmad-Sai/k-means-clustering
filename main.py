import math
import numpy as np
import matplotlib.pyplot as plt


class clustering:
    
    def __init__(self, k, threshold):
        self.k = k
        self.threshold = threshold
        self.weights = np.array([1,1,1])
        
        
    # creates 700 random data points using numpy arrays
    def createData(self):
        self.cluster1 = np.array([[.30,.15,.92]]) + np.random.normal(0, 0.05, (100,3))
        self.cluster2 = np.array([[.59,.73,.22]]) + np.random.normal(0, 0.05, (100,3))
        self.cluster3 = np.array([[.51,.32,.73]]) + np.random.normal(0, 0.05, (100,3))
        self.cluster4 = np.array([[.02,.41,.45]]) + np.random.normal(0, 0.05, (100,3))
        self.cluster5 = np.array([[.21,.72,.64]]) + np.random.normal(0, 0.05, (100,3))
        self.cluster6 = np.array([[.44,.32,.55]]) + np.random.normal(0, 0.05, (100,3))
        self.cluster7 = np.array([[.66,.62,.55]]) + np.random.normal(0, 0.05, (100,3))


        self.result = np.concatenate((self.cluster1, self.cluster2, self.cluster3, self.cluster4, self.cluster5, self.cluster6, self.cluster7))  # puts all of the data points into 1 numpy array
        self.normalized = self.result/np.max(self.result,axis=0)      # normalizes data

        
    # plots the unclustered data onto a 3D graph    
    def plotOriginal(self):
        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(projection='3d')

        ax.scatter(self.cluster1[:,0], self.cluster1[:,1], self.cluster1[:,2])
        ax.scatter(self.cluster2[:,0], self.cluster2[:,1], self.cluster2[:,2])
        ax.scatter(self.cluster3[:,0], self.cluster3[:,1], self.cluster3[:,2])
        ax.scatter(self.cluster4[:,0], self.cluster4[:,1], self.cluster4[:,2])
        ax.scatter(self.cluster5[:,0], self.cluster5[:,1], self.cluster5[:,2])
        ax.scatter(self.cluster6[:,0], self.cluster6[:,1], self.cluster6[:,2])
        ax.scatter(self.cluster7[:,0], self.cluster7[:,1], self.cluster7[:,2])

        plt.show()
        
        
        
    # chooses 7 unqiue and random intial points to begin comparisons
    def randoom(self):
        index = np.random.choice(len(self.normalized), self.k, replace=False)
        self.representatives = self.normalized[index]        
    
    # finds the distance betweeen 2 points and calculates the dot product using weights 
    def distance(self, pointA, pointB, weights):
        diff = np.abs(pointA-pointB)
        return diff@weights

    # assigns the data points to 7 classes based on calculated distances
    def assignClass(self):
        assignment = [-1]*len(self.normalized)
        for j in range(len(self.normalized)):
            closestDist = 9999999
            closestClassIndex = 0
            for i in range(len(self.representatives)):
                dist = self.distance(self.representatives[i], self.normalized[j], self.weights)
                if dist < closestDist:
                    closestDist = dist
                    closestClassIndex = i
            assignment[j] = closestClassIndex

        return assignment
    
    # makes the classes array
    def makeClasses(self):
        self.classes = self.assignClass()
    
    # calculates new represnatives of which new distances will be calculated from
    def newReprenstatives(self):
        newReps = self.representatives[:]
        for i in range(self.k):
            countClass = 0
            classTotal = np.zeros(len(self.weights))

            for j in range(len(self.normalized)):
                if self.classes[j] == i:
                    countClass +=1
                    classTotal += self.normalized[j]
            if countClass !=0:
                newReps[i] = classTotal/countClass
        return newReps

    # optimized the clustering process until the thershold is met
    def optimize(self):
        while True:
            self.representatives = self.newReprenstatives()
            new_classes = self.assignClass()

            counter = np.count_nonzero(self.classes == new_classes)
            if counter>0:
                if counter/len(self.classes) < self.threshold:  # stops looping when threshold is met
                    break

            self.classes = new_classes
            
    # puts clustered data into 7 different classes (arrays)
    def sortClasses(self):
        self.clusters = [[] for i in range(self.k)]

        for j in range(self.k):
            for i in range(len(self.normalized)):
                if self.classes[i] == j:
                    self.clusters[j].append(self.normalized[i])

        
    # plots the clustered data onto a 3D graph
    def plotNew(self):
        class1 = np.array(self.clusters[0])
        class2 = np.array(self.clusters[1])
        class3 = np.array(self.clusters[2])
        class4 = np.array(self.clusters[3])
        class5 = np.array(self.clusters[4])
        class6 = np.array(self.clusters[5])
        class7 = np.array(self.clusters[6])
        
        fig = plt.figure(figsize=(6, 6), dpi=100)
        bx = fig.add_subplot(projection='3d')
        
        bx.scatter(class1[:,0], class1[:,1], class1[:,2])
        bx.scatter(class2[:,0], class2[:,1], class2[:,2])
        bx.scatter(class3[:,0], class3[:,1], class3[:,2])
        bx.scatter(class4[:,0], class4[:,1], class4[:,2])
        bx.scatter(class5[:,0], class5[:,1], class5[:,2])
        bx.scatter(class6[:,0], class6[:,1], class6[:,2])
        bx.scatter(class7[:,0], class7[:,1], class7[:,2])
        plt.show()

    # calculates the standard deviation of the clustered and unclustered data points
    def getStandDev(self):
        sumClasses = 0

        for i in range(len(self.result)):
            sumClasses += self.distance(self.result[i],self.representatives[self.classes[i]],self.weights)


        return (sumClasses/len(self.result), math.sqrt(0.05))
    
    
# runs the program
if __name__ == "__main__":
    cluster1 = clustering(7, 0.1)
    cluster1.createData()      
    cluster1.plotOriginal() 
    cluster1.randoom()
    cluster1.assignClass()
    cluster1.makeClasses()
    cluster1.optimize()
    cluster1.sortClasses()
    cluster1.plotNew()
    print(cluster1.getStandDev())
