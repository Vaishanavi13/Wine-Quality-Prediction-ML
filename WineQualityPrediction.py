#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


red_wine = pd.read_csv(r'C:\Users\HP\Downloads\winequality-red.csv', sep = ';')
white_wine = pd.read_csv(r'C:\Users\HP\Downloads\winequality-white.csv', sep = ';')

red_wine.head()
red_wine.info()

white_wine.head()
white_wine.info()


# In[21]:


#Visualizing data of red-wine
sns.barplot(x = 'quality', y = 'fixed acidity', data = red_wine)
plt.show()
sns.barplot(x = 'quality', y = 'volatile acidity', data = red_wine)
plt.show()
sns.barplot(x = 'quality', y = 'citric acid', data = red_wine)
plt.show()
sns.barplot(x = 'quality', y = 'residual sugar', data = red_wine)
plt.show()
sns.barplot(x = 'quality', y = 'chlorides', data = red_wine)
plt.show()
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = red_wine)
plt.show()
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = red_wine)
plt.show()
sns.barplot(x = 'quality', y = 'sulphates', data = red_wine)
plt.show()
sns.barplot(x = 'quality', y = 'alcohol', data = red_wine)
plt.show()


# In[22]:


#Visualizing data of white-wine
sns.barplot(x = 'quality', y = 'fixed acidity', data = white_wine)  
plt.show()
sns.barplot(x = 'quality', y = 'volatile acidity', data = white_wine)  
plt.show()
sns.barplot(x = 'quality', y = 'citric acid', data = white_wine)  
plt.show()
sns.barplot(x = 'quality', y = 'residual sugar', data = white_wine)  
plt.show()
sns.barplot(x = 'quality', y = 'chlorides', data = white_wine)  
plt.show()
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = white_wine)  
plt.show()
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = white_wine)  
plt.show()
sns.barplot(x = 'quality', y = 'sulphates', data = white_wine)  
plt.show()
sns.barplot(x = 'quality', y = 'alcohol', data = white_wine)  
plt.show()


# In[23]:


#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
#for red-wine
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
red_wine['quality'] = pd.cut(red_wine['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
#Bad becomes 0 and good becomes 1 
red_wine['quality'] = label_quality.fit_transform(red_wine['quality'])
red_wine['quality'].value_counts()
sns.countplot(red_wine['quality'])


# In[24]:


#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
#for white-wine
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
white_wine['quality'] = pd.cut(white_wine['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
#Bad becomes 0 and good becomes 1 
white_wine['quality'] = label_quality.fit_transform(white_wine['quality'])
white_wine['quality'].value_counts()
sns.countplot(white_wine['quality'])


# In[30]:


#Now seperate the dataset into response variable and feature variabes
X_r = red_wine.drop('quality', axis = 1)
y_r= red_wine['quality']
#Splitting data into Train and Test  
X_Rtrain, X_Rtest, y_Rtrain, y_Rtest = train_test_split(X_r, y_r, test_size = 0.2, random_state = 42)
#Standard scaling to get optimized result
sc = StandardScaler()
X_Rtrain = sc.fit_transform(X_Rtrain)
X_Rtest = sc.fit_transform(X_Rtest)


# In[31]:


#Same for white-wine
X_w = white_wine.drop('quality', axis = 1)
y_w= white_wine['quality']
#Splitting data into Train and Test  
X_Wtrain, X_Wtest, y_Wtrain, y_Wtest = train_test_split(X_w, y_w, test_size = 0.2, random_state = 42)
#Standard scaling to get optimized result
sc = StandardScaler()
X_Wtrain = sc.fit_transform(X_Wtrain)
X_Wtest = sc.fit_transform(X_Wtest)


# In[32]:


#training and testing model for red wine
svc_r = SVC()
svc_r.fit(X_Rtrain, y_Rtrain)
pred_svc_r = svc_r.predict(X_Rtest)

print(pred_svc_r)
print(classification_report(y_Rtest, pred_svc_r))


# In[33]:


#training and testing model for white-wine
svc_w = SVC()
svc_w.fit(X_Wtrain, y_Wtrain)
pred_svc_w = svc.predict(X_Wtest)

print(pred_svc_w)
print(classification_report(y_Wtest, pred_svc_w))


# In[34]:


from sklearn import metrics
print("Accuracy: ", metrics.accuracy_score(y_Rtest,pred_svc_r))
print("Precision:", metrics.precision_score(y_Rtest,pred_svc_r))
print("Recall:", metrics.recall_score(y_Rtest,pred_svc_r))


# In[35]:


from sklearn import metrics
print("Accuracy: ", metrics.accuracy_score(y_Wtest,pred_svc_w))
print("Precision:", metrics.precision_score(y_Wtest,pred_svc_w))
print("Recall:", metrics.recall_score(y_Wtest,pred_svc_w))


# In[ ]:




