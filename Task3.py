#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics


# In[12]:


df = pd.read_csv('C:\\Users\\SKD\\Downloads\\advertising.csv')


# In[13]:


df


# In[14]:


df.isnull().sum()


# In[16]:


df.shape


# In[17]:


df.describe()


# In[19]:


X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


# In[20]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# In[22]:


# Evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[23]:


# Visualize the predictions
plt.scatter(X_test['TV'], y_test, color='black', label='Actual Sales')
plt.scatter(X_test['TV'], y_pred, color='blue', label='Predicted Sales')
plt.xlabel('TV Advertising Budget')
plt.ylabel('Sales')
plt.legend()
plt.show()


# In[ ]:




