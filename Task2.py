#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[3]:


# Load the dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('C:\\Users\\SKD\\Downloads\\creditcard.csv')


# In[4]:


df.isnull().sum()


# In[6]:


# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']


# In[7]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Standardize the features (normalize)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[9]:


# Handle class imbalance using oversampling (RandomOverSampler)
oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)


# In[10]:


# Train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test)


# In[12]:


# Evaluate the model
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
confusion_matrix(y_test, y_pred)


# In[13]:


# Create a hypothetical transaction data (replace these values with realistic ones)
hypothetical_data = pd.DataFrame({
    'Time': [100],  # Replace with a realistic value
    'V1': [0.0],    # Replace with a realistic value
    'V2': [-1.5],   # Replace with a realistic value
    'V3': [2.5],    # Replace with a realistic value
    'V4': [-0.5],   # Replace with a realistic value
    'V5': [1.0],    # Replace with a realistic value
    'V6': [-0.5],   # Replace with a realistic value
    'V7': [0.5],    # Replace with a realistic value
    'V8': [-0.2],   # Replace with a realistic value
    'V9': [0.3],    # Replace with a realistic value
    'V10': [0.1],   # Replace with a realistic value
    'V11': [-0.5],  # Replace with a realistic value
    'V12': [-0.6],  # Replace with a realistic value
    'V13': [-1.0],  # Replace with a realistic value
    'V14': [-0.3],  # Replace with a realistic value
    'V15': [1.5],   # Replace with a realistic value
    'V16': [-0.5],  # Replace with a realistic value
    'V17': [0.2],   # Replace with a realistic value
    'V18': [0.03],  # Replace with a realistic value
    'V19': [0.4],   # Replace with a realistic value
    'V20': [0.25],  # Replace with a realistic value
    'V21': [-0.02], # Replace with a realistic value
    'V22': [0.3],   # Replace with a realistic value
    'V23': [-0.1],  # Replace with a realistic value
    'V24': [0.07],  # Replace with a realistic value
    'V25': [0.1],   # Replace with a realistic value
    'V26': [-0.2],  # Replace with a realistic value
    'V27': [0.1],   # Replace with a realistic value
    'V28': [-0.02], # Replace with a realistic value
    'Amount': [200.0]  # Replace with a realistic value
})

# Standardize the hypothetical data
hypothetical_data_scaled = scaler.transform(hypothetical_data)

# Predict the class (fraudulent or not) using the trained model
prediction = model.predict(hypothetical_data_scaled)

# Print the prediction
print(f'Predicted Class: {"Fraudulent" if prediction[0] == 1 else "Not Fraudulent"}')


# In[14]:


df[df['Class'] == 1]


# In[32]:


hypothetical_data = pd.DataFrame({
    'Time': [21839],
     'V1': [1.017054969],
    'V2': [-0.185048783],
    'V3': [1.181666067],
    'V4': [1.073490204],
    'V5': [-0.550586426],
    'V6': [0.971826768],
    'V7': [-0.806451931],
    'V8': [0.484792244],
    'V9': [0.517268097],
    'V10': [-0.060123955],
    'V11': [1.446990443],
    'V12': [1.19801346],
    'V13': [-0.070890784],
    'V14': [0.006577366],
    'V15': [0.432215271],
    'V16': [-0.149872874],
    'V17': [-0.090841394],
    'V18': [-0.195277245],
    'V19': [-0.870040664],
    'V20': [-0.164292863],
    'V21': [0.239286567],
    'V22': [0.875847991],
    'V23': [-0.053457579],
    'V24': [-0.269918491],
    'V25': [0.328906924],
    'V26': [-0.217332072],
    'V27': [0.096207619],
    'V28': [0.020796223],
    'Amount': [90]
})

# Standardize the hypothetical data
hypothetical_data_scaled = scaler.transform(hypothetical_data)

# Predict the class (fraudulent or not) using the trained model
prediction = model.predict(hypothetical_data_scaled)

# Print the prediction
print(f'Predicted Class: {"Fraudulent" if prediction[0] == 1 else "Not Fraudulent"}')


# In[ ]:





# In[ ]:




