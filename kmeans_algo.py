from numpy import *
from sklearn import preprocessing
import matplotlib.pyplot as plt

def randomStart(x, mx):
    a = random.uniform(x,mx)
    
    return a

def distance(x,y, starts):
    dist = []
    
    for i in range(len(starts)):
        xnew = (x - starts[i][0])**2
        ynew = (y - starts[i][1])**2
        dist.append(math.sqrt(xnew + ynew))

    return dist.index(min(dist))

def distance2(x,y,starts,j):
    dist = 0
    return (x - starts[j][0])**2 + (y - starts[j][1])**2


def distortion(dataSet, k, belong, starts):
    result = 0
    
    for i in range(len(dataSet[0])):
        for j in range(len(starts)):
            dist   = distance2(dataSet[0][i], dataSet[1][i], starts,j)
            result = result + belong[i][j]*dist

    return result
    
def kmeans(dataSet, k, numOfRandomStarts, plot):

    distortionValue = 10000000
    best = []
    centroids = zeros((k,2))
    kvsd = []

    for n in range(numOfRandomStarts):
        
        starts = zeros((k,2))    
      
        
        avgX    = 0
        avgY    = 0
        count   = 0
        newPoints = []
        colors  = ['r','g','b','k','c']
        results = []
        result  = 0;
        for i in range(k):
            starts[i][0] = randomStart(min(dataSet[0]), max(dataSet[0]))
            starts[i][1] = randomStart(min(dataSet[1]), max(dataSet[1]))
            
        for z in range(100):
            belong = zeros((len(dataSet[0]), len(starts)))

            for i in range(len(dataSet[0])):
                j = distance(dataSet[0][i],dataSet[1][i], starts)
                
                belong[i][j] = 1

            new = []

            for j in range(len(starts)):
                avgX = 0.0
                avgY = 0.0
                
                for i in range(len(belong)):
                    if belong[i][j] == 1:
                        count = count + 1
                        avgX = avgX + dataSet[0][i]
                        avgY = avgY + dataSet[1][i]
                if not count == 0:
                    new.append([avgX/count,avgY/count])
                    count = 0
            
            if array_equal(array(new),starts):
                break
            starts = array(new)
            
            
        a = distortion(dataSet, k, belong, starts)
        
        if a <= distortionValue:
            distortionValue = a
            best = array(belong)
            centroids = array(starts)

    if plot == 1:
        for m in range(len(dataSet[0])):
            c = nonzero(best[m])[0]
            plt.scatter(dataSet[0][m], dataSet[1][m], color = colors[c])
            plt.xlabel('x of data')
            plt.ylabel('y of data')
            
        for p in range(len(centroids)):
            plt.annotate('centroid', xy=(centroids[p]),  xycoords='data',
                xytext=(-100, 60), textcoords='offset points',
                size=20,
                arrowprops=dict(arrowstyle="fancy",
                                fc="0.6", ec="none",
                                
                                connectionstyle="angle3,angleA=0,angleB=-90"),
                )

        plt.show()

    return distortionValue
def main():
    x = genfromtxt('k_means_dataset1.csv', delimiter=',')
    y = genfromtxt('k_means_dataset2.csv', delimiter=',')
    z = genfromtxt('k_means_dataset3.csv', delimiter=',')

    a = ((x[:,0],x[:,1])) # dataset1
    b = ((y[:,0],y[:,1])) # dataset2
    c = array((z[:,0],z[:,1])) # dataset3
    
    numOfRandomStarts = 200
    kfive  = 5;
    kthree = 3;
    plot   = 0;
    kvalues = []
    dists = []

    print "Program started with random start: ", 500, " k: ",2
    z_scaled = preprocessing.scale(z)
   
    z_scaled.mean(axis=0)
    z_scaled.mean(axis=1)
    z_scaled.std(axis=0)
    z_scaled.std(axis=1)
if __name__== "__main__":
    main()
