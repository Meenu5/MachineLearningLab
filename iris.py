
# coding: utf-8

# In[145]:

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import IPython


# In[146]:

from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[147]:

print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))


# In[148]:

print(iris_dataset['DESCR'][:193] + "\n ...")


# In[149]:

print("Target names: {}".format(iris_dataset['target_names']))


# In[150]:

print("Feature names: {}".format(iris_dataset['feature_names']))


# In[151]:

print("Type of data: {}".format(type(iris_dataset['data'])))


# In[152]:

print("Shape of data: {}".format(iris_dataset['data'].shape))


# In[153]:

print("First five columns of data: \n{}".format(iris_dataset['data'][:5]))


# In[154]:

print("Type of target: {}".format(type(iris_dataset['target'])))


# In[155]:

print("Target: \n{}".format(iris_dataset['target']))


# In[170]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)


# In[157]:

print("X_train shape :{}".format(X_train.shape))
print("y_train shape :{}".format(y_train.shape))


# In[158]:

print("X_test shape :{}".format(X_test.shape))
print("y_test shape :{}".format(y_test.shape))


# In[159]:

print("X_Train Values:{}".format(X_train[:5]))


# In[160]:

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=30,alpha=0.8)
# alpha is for transparency of dots


# In[ ]:




# In[162]:

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[163]:

knn.fit(X_train,y_train)


# In[164]:

X_new = np.array([5,2.9,1,0.2])


# In[176]:

prediction = knn.predict(X_new.reshape(1,-1))
print("Prediction : {}".format(prediction))
print("Prediction target name : {}".format(iris_dataset['target_names'][prediction]))


# In[166]:

y_pred = knn.predict(X_test)
print("Test set prediciton: \n{}".format(y_pred))


# In[167]:

np.mean(y_pred == y_test)


# In[169]:

knn.score(X_test,y_test)


# In[177]:

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
print("Prediction score: {}".format(knn.score(X_test,y_test)))

