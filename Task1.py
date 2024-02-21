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


# In[6]:


df = pd.read_csv('C:\\Users\\SKD\\Downloads\\tested.csv')


# In[8]:


df.head()


# In[9]:


pd.set_option('display.max_rows', None)


# In[13]:


df


# In[14]:


duplicate_count = df.duplicated().sum()


# In[15]:


print("number of duplicated rows ", duplicate_count)


# In[16]:


df.isnull().sum()


# In[17]:


df = df.drop('Cabin', axis=1)


# In[20]:


df.isnull().sum()


# In[23]:


df['Fare'].fillna(method='ffill', inplace=True)


# In[24]:


df.isnull().sum()


# In[26]:


df['Age'].hist()


# In[27]:


# Separate the DataFrame into male and female subsets
male_df = df[df['Sex'] == 'male']
female_df = df[df['Sex'] == 'female']

# Forward fill missing values in the 'Age' column for males
male_df['Age'].fillna(method='ffill', inplace=True)

# Forward fill missing values in the 'Age' column for females
female_df['Age'].fillna(method='ffill', inplace=True)

# Combine the updated subsets back into the original DataFrame
df.update(male_df)
df.update(female_df)

# Display the DataFrame after filling null values in the 'Age' column based on gender
print(df)


# In[28]:


df['Age'].hist()


# In[31]:


df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})


# In[45]:


df


# In[50]:


X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']




# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)


# In[58]:


# Evaluate the model on the testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

def predict_survival(new_data):
    # Preprocess the new data (similar to the preprocessing done for training data)
    new_data['Age'].fillna(new_data['Age'].mean(), inplace=True)

    # Make predictions using the trained model
    new_data_features = new_data[['Pclass', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
    predicted_survival = model.predict(new_data_features)

    return predicted_survival


# In[63]:


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


# In[ ]:




