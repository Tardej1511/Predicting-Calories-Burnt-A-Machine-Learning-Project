#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import pandas as pd

# Correct file paths
calories = pd.read_csv('Downloads/calories.csv')
exercise = pd.read_csv('Downloads/exercise.csv')


# In[3]:


calories.head(2)


# In[4]:


exercise.head(2)


# In[5]:


df = exercise.merge(calories, on='User_ID')


# In[6]:


df.head(1)


# In[7]:


sns.barplot(x=df['Gender'],y=df['Age'])


# In[8]:


sns.barplot(x=df['Gender'],y=df['Duration'])


# In[9]:


df.head(3)


# In[10]:


sns.barplot(x=df['Gender'],y=df['Calories'])


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt  # Recommended for showing plots

# Plotting the Gender count
sns.countplot(x='Gender', data=df)
plt.show()


# In[12]:


df['Gender'].value_counts()


# In[13]:


plt.hist(df['Age'],bins=20)


# In[14]:


df.head(1)


# In[15]:


sns.scatterplot(x=df['Calories'],y=df['Height'])


# In[16]:


sns.scatterplot(x=df['Age'],y=df['Calories'])


# In[17]:


sns.scatterplot(x=df['Age'],y=df['Duration'])


# In[18]:


sns.scatterplot(x=df['Calories'],y=df['Duration'])


# In[19]:


sns.scatterplot(x=df['Duration'],y=df['Calories'])


# In[20]:


sns.barplot(x=df['Age'],y=df['Calories'])


# In[21]:


sns.boxplot(x=df['Gender'],y=df['Age'])


# In[22]:


df.describe()


# In[23]:


sns.distplot(df['Age'])


# In[24]:


sns.lineplot(x=df['Age'],y=df['Calories'])


# # Encoding

# In[25]:


df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})


# In[26]:


df.head(1)


# In[27]:


df.head(3)


# In[28]:


x=df.drop(['User_ID','Calories'],axis=1)
y=df['Calories']


# In[29]:


x.shape


# In[30]:


y.shape


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


# In[33]:


x_train.shape


# In[34]:


x_test.shape


# # Training Model 

# In[35]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Dictionary of regression models
models = {
    'lr': LinearRegression(),         # Linear Regression
    'rd': Ridge(),                    # Ridge Regression
    'ls': Lasso(),                    # Lasso Regression
    'dtr': DecisionTreeRegressor(),   # Decision Tree Regressor
    'rfr': RandomForestRegressor()    # Random Forest Regressor
}


# In[36]:


from sklearn.metrics import mean_squared_error, r2_score

# Loop to train and evaluate each model
for name, mod in models.items():
    mod.fit(x_train, y_train)            # Train the model
    y_pred = mod.predict(x_test)         # Predict on test data

    # Evaluate and print performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name}  MSE: {mse:.4f}, R² Score: {r2:.4f}")


# # Selecting Model

# In[37]:


rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)
y_pred=rfr.predict(x_test)


# In[38]:


import pickle
# pickle.dump(rfr,open('rfr.pkl','wb'))
# Save the retrained model
with open('rfr.pkl', 'wb') as file:
    pickle.dump(rfr, file)


# In[39]:


x_train.to_csv('x_train.csv')


# In[40]:


x_test



# In[41]:


y_test


# In[42]:


df


# In[ ]:


df.describe()


# In[ ]:





# In[ ]:





# In[ ]:




