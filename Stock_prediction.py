#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("C:/Users/harsh/Downloads/NFLX.csv")
df.head()


# In[3]:


df['Date'] = pd.to_datetime(df['Date']) #since normaly date time is object so we need to convert it into date time useing pyton


# In[4]:


df.info()


# In[5]:


df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day


# In[6]:


df.drop('Date', axis = 1, inplace= True)


# In[7]:


df.head()


# In[8]:


#DATA CLEANING


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# In[11]:


df.duplicated().sum()


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


#EDA


# In[15]:


import seaborn as sns


# In[16]:


corr = df.corr()
corr


# In[17]:


plt.figure(figsize=(10,5))
sns.heatmap(corr,annot=True , cbar = True , cmap = 'coolwarm')
plt.show()


# In[18]:


sns.pairplot(df)


# In[19]:


for i in df.columns:
    plt.figure(figsize=(5,10))
    sns.distplot(df[i])
    plt.title(i)
    plt.show()


# In[20]:


X = df.drop(columns=['Close'])
y = df['Close']


# In[21]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(x_train.shape, y_train.shape)  # Should have the same number of rows


# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


scale = StandardScaler()


# In[24]:


x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)


# In[25]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)


# In[26]:


pred


# In[27]:


#MODEL EVALUTAION
from sklearn.metrics import r2_score , mean_squared_error


# In[28]:


score = r2_score(y_test,pred)
print(f"accuracy: {score*100}%")


# In[29]:


score = mean_squared_error(y_test,pred)
score


# In[30]:


def get_stock_pred(Open, High, Low, Adj_Close, Volume, Year, Month,day):
    features = np.array([[Open, High, Low, Adj_Close, Volume, Year, Month,day]])
    features = scale.fit_transform(features)
    pred = lr.predict(features)
    return pred[0]


# In[31]:


Open = 230.0
High = 450.0
Low = 190.8
Adj_Close = 100.34
Volume = 100.34
Year = 2024
Month = 9
day = 15


# In[32]:


res = get_stock_pred(Open, High, Low, Adj_Close, Volume, Year, Month,day)
res


# In[ ]:




