#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\SKD\\Downloads\\tested.csv')


# In[2]:


df.isnull().sum()


# In[3]:


df['Fare'].fillna(method='ffill', inplace=True)


# In[4]:


df['Age'].hist()


# In[8]:


# Separate the DataFrame into male and female subsets
male_df = df[df['Sex'] == 'male'].copy()
female_df = df[df['Sex'] == 'female'].copy()

# Forward fill missing values in the 'Age' column for males
male_df['Age'].fillna(method='ffill', inplace=True)

# Forward fill missing values in the 'Age' column for females
female_df['Age'].fillna(method='ffill', inplace=True)

# Combine the updated subsets back into the original DataFrame
df.update(male_df)
df.update(female_df)

# Display the DataFrame after filling null values in the 'Age' column based on gender
df


# In[9]:


df


# In[10]:


df['Age'].hist()


# In[11]:


df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# In[12]:


df


# In[14]:


# Select features and target variable
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# In[15]:


# Make predictions on the testing set
y_pred = model.predict(X_test)


# In[19]:


# Evaluate the model on the testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

def predict_survival(new_data):
    # Preprocess the new data (similar to the preprocessing done for training data)
    new_data['Age'].fillna(new_data['Age'].mean(), inplace=True)

    # Make predictions using the trained model
    new_data_features = new_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
    predicted_survival = model.predict(new_data_features)

    return predicted_survival


# In[17]:


df


# In[20]:


# Example usage of the predict_survival function
new_passenger_data = pd.DataFrame({
    'Pclass': [3],
    'Sex': [1],  # Use 'Sex' instead of 'Sex_male'
    'Age': [25],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked_Q': [0],
    'Embarked_S': [1]
})

predicted_survival = predict_survival(new_passenger_data)
print(f'Predicted Survival: {"Survived" if predicted_survival[0] == 1 else "Did not survive"}')


# In[32]:


# Example usage of the predict_survival function
new_passenger_data = pd.DataFrame({
    'Pclass': [],
    'Sex': [0],  # Use 'Sex_male' instead of 'Sex'
    'Age': [25],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [9.687],
    'Embarked_Q': [1],  # Add the 'Embarked_Q' value for your passenger
    'Embarked_S': [0]   # Add the 'Embarked_S' value for your passenger
})

predicted_survival = predict_survival(new_passenger_data)
print(f'Predicted Survival: {"Survived" if predicted_survival[0] == 1 else "Did not survive"}')

# Print the predicted probabilities for both classes
predicted_probabilities = model.predict_proba(new_passenger_data)
print(f'Predicted Probabilities: Not Survived={predicted_probabilities[0, 0]:.2f}, Survived={predicted_probabilities[0, 1]:.2f}')


# 
# 

# In[ ]:





# In[ ]:




