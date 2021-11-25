import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
#Part A
###section 1
data = np.genfromtxt(sys.argv[1], delimiter=',')
features = data[:,0:data.shape[1]-1]
labels = data[:,data.shape[1]-1] #vector labels[0]
temp = []
temp_1 = []
temp_2 = []
temp_c = []
num_1 = 3
num_2 = 8
for i in range(0,labels.shape[0]):
	if labels[i] == num_1:
		temp.append(features[i,:])
		temp_c.append(labels[i])
		temp_1.append(features[i,:])
	elif labels[i] == num_2:
		temp.append(features[i,:])
		temp_c.append(labels[i])
		temp_2.append(features[i,:])
selected = np.asarray(temp)
selected_1 = np.asarray(temp_1)
selected_2 = np.asarray(temp_2)
selected_c = np.asarray(temp_c)
del temp, temp_1, temp_2, temp_c

x_train, x_test, y_train, y_test = model_selection.train_test_split(selected,selected_c,test_size=0.2)
std_scale = preprocessing.StandardScaler().fit(x_train)
x_train_std = std_scale.transform(x_train)
x_test_std = std_scale.transform(x_test)

size = x_train.shape[0]
cValues = [0.0001,0.001,0.01,0.1,1,10,100]
pValues = [2,3,4,5,6]
length = x_train.shape[0]-1

for c in cValues:
	estimator = svm.SVC(kernel='linear', C=c)
	scores = model_selection.cross_val_score(estimator,selected,selected_c,cv=2,scoring="accuracy")
	print("Accuracy: %0.4f (+/- %0.4f) for c:%0.4f" % (scores.mean(), scores.std() * 2,c))

accuracy_list = np.zeros(len(cValues))
i = 0
for c in cValues:
	total = 0.0
	for k in (1,2):
		first = (length/2) * (k-1)
		last = (length/2) * (k)
		estimator = svm.SVC(kernel='linear',C=c).fit(x_train_std[first:last,:], y_train[first:last])
		y_pred_svm = estimator.predict(x_train_std[length-last:length-first,:])
		acc_svm = metrics.accuracy_score(y_train[length-last:length-first], y_pred_svm)
	total += (acc_svm)
	accuracy_list[i] = total
	i = i+1
	#print "Linear SVM accuracy: ",total, "for c val:",c
print accuracy_list



estimator = svm.SVC(kernel='linear',C=0.01).fit(x_train_std, y_train)
y_pred_svm = estimator.predict(x_test_std)
acc_svm = metrics.accuracy_score(y_test, y_pred_svm)
print "Linear SVM accuracy: ",acc_svm, "for c val:", (0.01)

###########part b
for c in cValues:
	for p in pValues:
		estimator = svm.SVC(kernel='poly',C=c,degree=p).fit(x_train_std, y_train)
		scores = model_selection.cross_val_score(estimator,x_train_std,y_train,cv=2,scoring="accuracy")
		print("Accuracy: %0.4f (+/- %0.4f) for c:%0.4f and p:%1d" % (scores.mean(), scores.std() * 2,c,int(p)))
		del scores
estim = svm.SVC(kernel='poly',C=100,degree=2).fit(x_train_std, y_train)
y_pred_svm = estim.predict(x_test_std)
acc_svm = metrics.accuracy_score(y_test, y_pred_svm)
print acc_svm
