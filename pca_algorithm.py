from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import sys
#Part A
###section 1
data = genfromtxt(sys.argv[1], delimiter=',')
features = data[:,0:data.shape[1]-1]
labels = data[:,data.shape[1]-1] #vector labels[0]
temp = []
temp_1 = []
temp_2 = []
temp_c = []
num_1 = 1
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

###section 2
mean_vec = np.zeros(selected.shape[1])
mean_vec_1 = np.zeros(selected_1.shape[1])
mean_vec_2 = np.zeros(selected_2.shape[1])
for i in range(0,selected.shape[1]):
	mean_vec[i] = np.mean(selected[:,i])
	if i < selected_1.shape[1]:
		mean_vec_1[i] = np.mean(selected_1[:,i])
	if i < selected_2.shape[1]:
		mean_vec_2[i] = np.mean(selected_2[:,i])

covar_mat = (selected - mean_vec).T.dot((selected - mean_vec)) / ( selected.shape[0]-1)
eigen_val, eigen_vec = np.linalg.eig(covar_mat)

###section 3
I = np.reshape(mean_vec,(20,20))
imgplot = plt.imshow(I.T,cmap='gray')
plt.axis('on')
title_text = "Sample mean as an image"
plt.title(title_text)
plt.show()
I = np.reshape(mean_vec_1,(20,20))
imgplot = plt.imshow(I.T,cmap='gray')
plt.axis('on')
label_i = "mean"
title_text = "Sample mean of\t".expandtabs()
title_text += "image: "+str(num_1)
plt.title(title_text)
plt.show()
I = np.reshape(mean_vec_2,(20,20))
imgplot = plt.imshow(I.T,cmap='gray')
plt.axis('on')
label_i = "mean"
title_text = "Sample mean of\t".expandtabs()
title_text += "image: "+str(num_2)
plt.title(title_text)
plt.show()

covar_mat = (selected - mean_vec).T.dot((selected - mean_vec)) / ( selected.shape[0]-1)
eig_val, eig_vec = np.linalg.eig(covar_mat)

# get the array of indices that would sort `eig_val` in ascending order of 
# magnitude
asc_order = np.argsort(np.abs(eig_val))
# reverse the order of the indices using slice indexing
desc_order = asc_order[::-1]
# sort eigenvalues and eigenvectors using this index array
sorted_eig_val = eig_val[desc_order]
sorted_eig_vec = eig_vec[:, desc_order]

x_to_show = 27
x = [i for i in range(1,len(sorted_eig_val)+1)]
y = [sorted_eig_val[i] for i in range(len(sorted_eig_val))]
plt.plot(x,y)
plt.title("the eigenvalues in descending order")
plt.xlabel("")
plt.ylabel("eigen value")
plt.xticks([i for i in range(x_to_show+1)], [i for i in range(len(sorted_eig_val))])
#plt.yticks([i for i in range(40)], [eig_val[i] for i in range(len(sorted_eig_val))])
for i in range(x_to_show):
	plt.annotate(np.around(y[i], decimals=3),(x[i],y[i]))
#plt.annotate(y[1],(x[1],y[1]))
plt.autoscale()
plt.ylim(sorted_eig_val[len(sorted_eig_val)-1], sorted_eig_val[0])
plt.xlim(1,x_to_show)
plt.show()

###section 4
num_top_images = 5
m = 1
top_eig_vec = [np.reshape(sorted_eig_vec[:,i],(20,20)) for i in range(0,num_top_images)]
for I in top_eig_vec:
	imgplot = plt.imshow(I.T,cmap='gray')
	plt.axis('on')
	title_text = "as image, eigen vector "+str(m)
	plt.title(title_text)
	plt.show()
	m += 1

###section 5
tot = sum(eig_val)
fpc = 0
spc = 0
count = 0
for i in sorted_eig_val:
    if count == 0:
        fpc += (i / tot)
    elif count > 1:
        spc += (i / tot)
    count += 1
fpc += spc
x = [1,2]
y = [fpc*100,spc*100]
# colors = np.random.rand(2)
# area = np.pi * (15 * np.random.rand(2))**2  # 0 to 15 point radiuses
plt.scatter(x,y)
plt.title("First and Second principal component projection")
plt.xlabel("ith pca")
plt.ylabel("percentage of variance retained")
plt.xticks([i for i in range(1,3)],
  ['PC %s' %i for i in range(1,3)])
plt.annotate(y[0],(x[0],y[0]))
plt.annotate(y[1],(x[1],y[1]))
plt.grid()
plt.autoscale()
plt.ylim(50, 100)
plt.show()