import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

#reading the data into lists.
data = np.genfromtxt(sys.argv[1], delimiter=',')
raw_features = data[:,0:data.shape[1]-1]
X_train = raw_features[0:4000,:]
X_test = raw_features[4000:5000,:]
#normalize the data for each column
features = np.zeros((raw_features.shape[0],raw_features.shape[1]))
for i in range(0,raw_features.shape[1]):
	minVal = min(raw_features[:,i])
	maxVal = max(raw_features[:,i])
	features[:,i] = (raw_features[:,i] - minVal) / (maxVal - minVal)
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

labels = data[:,data.shape[1]-1] #vector labels[0]
temp = []
temp_c = []
for i in range(0,labels.shape[0]-1000):
	temp.append(X_train_std[i,:])
	temp_c.append(labels[i])
selected = np.asarray(temp)
selected_c = np.asarray(temp_c)
del temp, temp_c

###Part B
#get the mean vector for each image.
mean_vec = np.zeros(selected.shape[1])
for i in range(0,selected.shape[1]):
	mean_vec[i] = np.mean(selected[:,i])
#calculate the covariance matrix and eigen values and vectors.
covar_mat = (selected - mean_vec).T.dot((selected - mean_vec)) / ( selected.shape[0]-1)
eig_val, eig_vec = np.linalg.eig(covar_mat)

#display mean vector as an image.
I = np.reshape(mean_vec,(20,20))
imgplot = plt.imshow(I.T,cmap='gray')
plt.axis('on')
title_text = "Sample mean as an image"
plt.title(title_text)
plt.show()

eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
# get the array of indices that would sort `eig_val` in ascending order of magnitude
asc_order = np.argsort(np.abs(eig_val))
# reverse the order of the indices using slice indexing
desc_order = asc_order[::-1]
# sort eigenvalues and eigenvectors using this index array
sorted_eig_val = eig_val[desc_order]
sorted_eig_vec = eig_vec[:, desc_order]
#to run faster
#ploting the resulting eigenvalues in descending order.

x_to_show = 18
x = [i for i in range(1,len(sorted_eig_val[0:x_to_show])+1)]
y = [sorted_eig_val[i] for i in range(len(sorted_eig_val[0:x_to_show]))]
plt.plot(x,y, 'g.')
plt.title("the eigenvalues in descending order")
plt.xlabel("ith eigenvalue")
plt.ylabel("eigen value")
plt.xticks([i for i in range(x_to_show+1)], [i for i in range(len(sorted_eig_val))])
#plt.yticks([i for i in range(40)], [eig_val[i] for i in range(len(sorted_eig_val))])
plt.yticks([np.around(y[i], decimals=2) for i in range(x_to_show)])
#for i in range(x_to_show):
#	plt.annotate(np.around(y[i], decimals=3),(x[i],y[i]))
#plt.annotate(y[1],(x[1],y[1]))
#plt.autoscale()
plt.ylim(min(y),np.around(max(y),decimals=0))
#plt.ylim(np.around(sorted_eig_val[len(sorted_eig_val)-1], decimals=0), sorted_eig_val[0])
plt.xlim(1,x_to_show)
plt.show()

#ploting the resulting eigenvalues in descending order.
x_to_show = 40
tot = sum(eig_val)
x = [i for i in range(1,len(sorted_eig_val)+1)]
y = [(sum(sorted_eig_val[0:i+1])/tot)*100 for i in range(len(sorted_eig_val))]
plt.plot(x,y)
plt.title("percentage of variance retained")
plt.xlabel("ith principal component chosen")
plt.ylabel("variance retained")
plt.xticks([i for i in range(x_to_show+1)], [i for i in range(len(sorted_eig_val))])
#plt.yticks([i for i in range(40)], [eig_val[i] for i in range(len(sorted_eig_val))])
#plt.yticks(np.arange(min(y), max(y), 10))
plt.yticks([np.around(y[i], decimals=2) for i in range(len(sorted_eig_val))])
#for i in range(x_to_show):
#	plt.annotate(np.around(y[i], decimals=3),(x[i],y[i]))
#plt.annotate(y[1],(x[1],y[1]))
plt.autoscale()
plt.ylim((sum(sorted_eig_val[0:1])/tot)*100,(sum(sorted_eig_val[0:x_to_show])/tot)*100)
plt.xlim(1,x_to_show)
plt.show()

###Part C
num_top_images = 5
m = 1
top_eig_vec = [np.reshape(sorted_eig_vec[:,i],(20,20)) for i in range(0,num_top_images)]
for I in top_eig_vec:
	imgplot = plt.imshow(I.T,cmap='gray')
	plt.axis('on')
	title_text = "as image, eigen vector #"+str(m)
	plt.title(title_text)
	plt.show()
	m += 1
m = eig_vec.shape[0]-4
bot_eig_vec = [np.reshape(sorted_eig_vec[:,i],(20,20)) for i in range(eig_vec.shape[0]-5,eig_vec.shape[0])]
for I in bot_eig_vec:
	imgplot = plt.imshow(I.T,cmap='gray')
	plt.axis('on')
	title_text = "as image, eigen vector #"+str(m)
	plt.title(title_text)
	plt.show()
	m += 1

###Part D
first_pc = sorted_eig_vec[:,0]
#projected = first_pc.T.dot(selected[:,0:10])
projected = selected.dot(first_pc)
x_to_show = 400
tot = sum(eig_val)
x = [i for i in range(1,len(projected)+1)]
y = [projected]
#color = [labels[i] for i in range(1,len(projected))]
colorlist=['black','teal','yellow','sienna','blue','red','purple','orange','magenta','snow']
#color = [labels[i]/10 for i in range(1,len(projected)+1)]
color = [colorlist[int(labels[i])] for i in range(1,len(projected)+1)]
plt.scatter(x,y,s=100,c=color,cmap='gray',alpha=1)
plt.title("projecting the training data on the first pc")
plt.xlabel("")
plt.ylabel("")
plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.yticks(np.arange(min(projected), max(projected), 1))
#plt.xticks([i for i in range(x_to_show+1)], [i for i in range(len(sorted_eig_val))])
#plt.yticks([i for i in range(40)], [eig_val[i] for i in range(len(sorted_eig_val))])
#plt.yticks(np.arange(min(y), max(y), 10))
#plt.yticks([np.around(projected[i], decimals=2) for i in range(len(projected))])
#for i in range(x_to_show):
#	plt.annotate(np.around(y[i], decimals=3),(x[i],y[i]))
#plt.annotate(y[1],(x[1],y[1]))
plt.autoscale()
plt.ylim(min(projected),max(projected))
plt.xlim(1,x_to_show)
plt.show()

projected = selected.dot(sorted_eig_vec[:,0:2])
x_to_show = 400
tot = sum(eig_val)
x = [projected[:,0:1]]
y = [projected[:,1:2]]
#color = [labels[i] for i in range(1,len(projected))]
colorlist=['black','teal','yellow','sienna','blue','red','purple','orange','magenta','snow']
#color = [labels[i]/10 for i in range(1,len(projected)+1)]
color = [colorlist[int(labels[i])] for i in range(1,len(projected)+1)]
plt.scatter(x,y,s=25,c=color,cmap='gray',alpha=1)
plt.title("projecting the training data on the second pc")
plt.xlabel("")
plt.ylabel("")
plt.grid(axis='y',alpha=0.5, linestyle='--')
plt.yticks(np.arange(min(projected[:,1:2]), max(projected[:,1:2]), 1))
plt.autoscale()
plt.ylim(min(projected[:,1:2]),max(projected[:,1:2]))
plt.xlim(min(projected[:,0:1]),max(projected[:,0:1]))
plt.show()


accuracies_knn = np.zeros(100)
for i in range(1,101):
	projected = selected.dot(sorted_eig_vec[:,0:i])
	projected_test = X_test_std.dot(sorted_eig_vec[:,0:i])
	neigh = KNeighborsClassifier(n_neighbors=5)
	neigh.fit(projected, labels[0:4000])
	result = neigh.predict(projected_test)
	count = 0.0
	for j in range(len(result)):
		if(result[j] == labels[4000+j]):
			count = count + 1;
	accuracies_knn[i-1] = count/10
	#print(count/1000)
x = [i for i in range(1,len(accuracies_knn)+1)]
y = accuracies_knn
plt.plot(x,y)
plt.title("accuracy for ith component")
plt.xlabel("#of selected component")
plt.ylabel("accuracy in percentage")
plt.grid()
plt.xticks([i for i in range(1,len(accuracies_knn)+1)])
#plt.yticks([i for i in range(40)], [eig_val[i] for i in range(len(sorted_eig_val))])
#plt.yticks(np.arange(min(y), max(y), 10))
#plt.yticks([np.around(y[i], decimals=2) for i in range(len(accuracies_knn))])
plt.yticks(np.arange(math.ceil(min(y)), math.floor(max(y))+2, 1))
#for i in range(x_to_show):
#	plt.annotate(np.around(y[i], decimals=3),(x[i],y[i]))
#plt.annotate(y[1],(x[1],y[1]))
plt.autoscale()
plt.ylim(math.ceil(min(y)), math.floor(max(y))+1)
plt.xlim(1,101)
plt.show()