#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Importing the dataset
dataset = pd.read_csv('task1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# In[3]:


dataset.head()


# In[4]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[5]:


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[6]:


# Predicting the score respective to a value:
y_pred = regressor.predict(X_test)


# In[9]:


print(y_pred)


# In[10]:


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hours vs Scores (Training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[11]:


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hours vs Scores (Test set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[13]:


#Predicting the new value
y_prd=regressor.predict([[9.25]])
print(y_prd)


# In[ ]:




