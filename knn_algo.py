from numpy import *
from sklearn.cross_validation import *
import math
import matplotlib.pyplot as plt

def assign(k, le, y):
    le = sorted(le, key=lambda item: item[1])
    a = [i for i,j in le[:k]]

    return sum([y[i] for i in a])        


def test(k, a, skf, y):
    t = 0.0
    f = 0.0

    for train, test in skf:
        for i in test:
            le = []
            for j in train:
                le.append((j, a[i][j]))
            if (assign(k, le, y) >= 0 and y[i] == 1) or (assign(k, le, y) < 0 and y[i] == -1):
                t = t+1
            else:
                f = f+1
    
    return f/(t+f)

def testError(k, a, train, test, y):
    t = 0.0
    f = 0.0

    for i in range(len(test)):
        le = []
        for j in range(len(train)):
            le.append((j, a[i][j]))

        if (assign(k, le, y) >= 0 and y[i] == 1) or (assign(k, le, y) < 0 and y[i] == -1):
            t = t+1
        else:
            f = f+1

    return f/(t+f)


def yourTest(k, a, train, test, label):

    result = [];
    for i in range(len(test)):
        le = []
        for j in range(len(train)):
            le.append((j, a[i][j]))
        if assign(k,le,label) >= 0:
            result.append(1)
        else:
            result.append(-1)

    return result

def exTest(k, test):
    x = genfromtxt('cancer_features.csv', delimiter=',')
    y = genfromtxt(test, delimiter=',')
    z = genfromtxt('cancer_labels.csv')
    
    dE = zeros((len(y), len(x)))
    dM = zeros((len(y), len(x)))

    print x
    print y
    print len(y)
    
    for i in range(len(y)):
        for j in range(len(x)):
            xnew = (y[i][0] - x[j][0])**2
            ynew = (y[i][1] - x[j][1])**2
            dE[i][j] = math.sqrt(xnew + ynew)

    
    print "Your test result using Euclidean distence for k: ", k
    print yourTest(k, dE, x, y, z)

    for i in range(len(y)):
        for j in range(len(x)):
            xnew = abs(y[i][0] - x[j][0])
            ynew = abs(y[i][1] - x[j][1])
            dM[i][j] = xnew + ynew

    print "Your test resulr using Manhattan distance for k: ", k
    print yourTest(k, dM, x,y,z)
    
    
def main():
    x = genfromtxt('cancer_features.csv', delimiter=',')
    y = genfromtxt('cancer_labels.csv')

    skf = StratifiedKFold(y, 10)

    kvalues = []
    errors  = []
    dE = zeros((len(x), len(x)))
    dM = zeros((len(x), len(x)))

    for i in range(len(x)):
        for j in range(len(x)):
            xnew = (x[i][0] - x[j][0])**2
            ynew = (x[i][1] - x[j][1])**2
            dE[i][j]   = math.sqrt(xnew+ynew)

    for i in range(len(x)):
        for j in range(len(x)):
            xnew = abs(x[i][0] - x[j][0])
            ynew = abs(x[i][1] - x[j][1])
            dM[i][j] = xnew + ynew

    
    ## Error Rates using Euclidean: Training and Validation
    for k  in range(1,21):
        errors.append(test(k, dE, skf, y))
        kvalues.append(k)

    b = kvalues[errors.index(min(errors))]
    print b
    a = min(errors)
    print a
                
    p1 = plt.plot(kvalues, errors, color = 'r')
    plt.xlabel('k values')
    plt.ylabel('Error rates')
    plt.title('Using Euclidean Distance')

                
    kvalues = []
    errors  = []

    for k in range(1, 21):
        errors.append( testError(k, dE, x, x, y ))
        kvalues.append(k)

    p2 = plt.plot(kvalues, errors, color = 'b')
    plt.legend((p1[0],p2[0]),('Validation Error', 'Training Error'))
    plt.show()
    
    
    kvalues = []
    errors  = []
    
    ## Error Rates using Manhattan: Training and Validation
    for k in range(1,21):
        errors.append( test(k, dM, skf, y))
        kvalues.append(k)


    b = kvalues[errors.index(min(errors))]
    print b
    a = min(errors)
    print a
    
    p1 = plt.plot(kvalues, errors, color = 'r')
    plt.xlabel('k values')
    plt.ylabel('Error rates')
    plt.title('Using Manhattan Distance')

    kvalues = []
    errors  = []

    for k in range(1, 21):
        errors.append( testError(k, dM, x, x, y ))
        kvalues.append(k)

    p2 = plt.plot(kvalues, errors, color = 'b')
    plt.legend((p1[0],p2[0]),('Validation Error', 'Training Error'))
    plt.show()

if __name__ == "__main__":
    main()
    
