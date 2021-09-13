#!/usr/bin/env python
# coding: utf-8

# # PREDICTING TELECOM CHURN 
# 
# # Predicting whether customer will likely to Churn or not
# 
# # based on their account informaton & services  they are using

# In[33]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[34]:


# Importing the dataset
dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
#X = dataset.iloc[:, [2,3]].values
#y = dataset.iloc[:, 20].values
dataset.head()


# In[35]:


dataset.shape


# In[36]:


dataset.isnull().sum()


# In[37]:


dataset["MultipleLines"].value_counts()


# In[38]:


dataset["InternetService"].value_counts()


# In[39]:


# Import label encoder 
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()
dataset['MultipleLines']= label_encoder.fit_transform(dataset['MultipleLines'])   
dataset['MultipleLines'].unique() 

dataset['InternetService']= label_encoder.fit_transform(dataset['InternetService'])   
dataset['InternetService'].unique() 

dataset['gender']= label_encoder.fit_transform(dataset['gender'])   
dataset['gender'].unique() 

dataset['Partner']= label_encoder.fit_transform(dataset['Partner'])   
dataset['Partner'].unique() 

dataset['Dependents']= label_encoder.fit_transform(dataset['Dependents'])   
dataset['Dependents'].unique() 

dataset['StreamingMovies']= label_encoder.fit_transform(dataset['StreamingMovies'])   
dataset['StreamingMovies'].unique() 

dataset['Churn']= label_encoder.fit_transform(dataset['Churn'])   
dataset['Churn'].unique() 


# In[40]:


X = dataset.iloc[:, [1,2,3,4,5,7,8]].values
y = dataset.iloc[:, 20].values
dataset.head()


# In[41]:


# Splitting the dataset into the Training set and Test set
# from sklearn.cross_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[42]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[43]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
KNN=KNeighborsClassifier()


# # KNN

# In[44]:


param_grid=[{'n_neighbors':[3,5,10,15]}]
grid_search_KNN=GridSearchCV(KNN,param_grid,cv=5)
grid_search_KNN.fit(X_train, y_train)


# In[45]:


grid_search_KNN.best_params_


# In[46]:


cvres_KNN=grid_search_KNN.cv_results_
for mean_score,params in zip(cvres_KNN["mean_test_score"],cvres_KNN["params"]):
    print(mean_score,params)


# In[47]:


# Prediction with KNN classifier

from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors = 15, metric='minkowski', p=2)
classifier1.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier1.predict(X_test)
df=pd.DataFrame(y_pred)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print('Accuracy Score: ',accuracy_score(y_test,y_pred))
print('--------------')
print(classification_report(y_test,y_pred))


# # Random Forest

# In[48]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

RF=RandomForestClassifier(random_state=123)


# In[49]:


from sklearn.model_selection import GridSearchCV
param_grid=[{'n_estimators':[4,5,10,20,50]}]
grid_search_RF=GridSearchCV(RF,param_grid,cv=5)
grid_search_RF.fit(X_train, y_train)


# In[50]:


grid_search_RF.best_params_


# In[51]:


cvres_RF=grid_search_RF.cv_results_
for mean_score,params in zip(cvres_RF["mean_test_score"],cvres_RF["params"]):
    print(mean_score,params)


# In[52]:


# Prediction with Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 4, criterion='entropy', random_state = 0)
classifier2.fit(X_train, y_train) 
# Predicting the Test set results
y_pred = classifier2.predict(X_test)
df=pd.DataFrame(y_pred)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print('Accuracy Score: ',accuracy_score(y_test,y_pred))
print('--------------')
print(classification_report(y_test,y_pred))


# # SVM

# In[53]:


# Prediction with SVM classifier
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'linear', random_state = 0)
classifier3.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier3.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print('Accuracy Score: ',accuracy_score(y_test,y_pred))
print('--------------')
print(classification_report(y_test,y_pred))


# # Decision Tree

# In[54]:


# Prediction with Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier4 = DecisionTreeClassifier(criterion='entropy', random_state = 0)
classifier4.fit(X_train, y_train) 
# Predicting the Test set results
y_pred = classifier4.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print('Accuracy Score: ',accuracy_score(y_test,y_pred))
print('--------------')
print(classification_report(y_test,y_pred))


# # Naive Bayes

# In[55]:


# Prediction with naive_bayes classifier
from sklearn.naive_bayes import GaussianNB
classifier5 = GaussianNB()
classifier5.fit(X_train, y_train) 
# Predicting the Test set results
y_pred = classifier5.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print('Accuracy Score: ',accuracy_score(y_test,y_pred))
print('--------------')
print(classification_report(y_test,y_pred))


# # Plotting a bar Graph between the accuracy of all 3 algorithms :

# In[56]:


acc=[]


# In[57]:


acc.append(classifier1.score(X_test, y_test))
acc.append(classifier2.score(X_test, y_test))
acc.append(classifier3.score(X_test, y_test))
acc.append(classifier4.score(X_test, y_test))
acc.append(classifier5.score(X_test, y_test))


# In[58]:


acc_name=['KNN','Random Forest','SVM','Decision Tree','Naive Bayes']


# In[59]:


colours=['b','r','g','c','m']
plt.xlabel('machine learning algorithms',fontsize=15)
plt.ylabel('Accuracy',fontsize=15)
plt.title('Accuracy Comparisions',fontsize=15)
plt.bar(acc_name,acc,color=colours,width=0.5)
plt.show()


# In[ ]:




