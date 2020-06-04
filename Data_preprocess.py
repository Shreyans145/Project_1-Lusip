#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[11]:


df = pd.read_csv('RELIANCE.NS.csv')


# In[12]:


print(df.head())


# In[18]:


fig, ((ax_1, ax_2),(ax_3,ax_4),(ax_5,ax_6)) = plt.subplots(3,2)
ax_1.scatter(df['Open'], df['Volume'],color ='red')
ax_1.set_title('Volume vs Open')
ax_1.set_xlabel('Open')
ax_1.set_ylabel('Volume')

ax_2.scatter(df['High'], df['Date'],color ='blue')
ax_2.set_title('Date vs High')
ax_2.set_xlabel('High')
ax_2.set_ylabel('Date')

ax_3.scatter(df['Low'], df['Date'],color ='green')
ax_3.set_title('Date vs Low')
ax_3.set_xlabel('Low')
ax_3.set_ylabel('Date')

ax_4.scatter(df['Close'], df['Date'],color ='black')
ax_4.set_title('Date vs Close')
ax_4.set_xlabel('Close')
ax_4.set_ylabel('Date')

ax_5.scatter(df['Adj Close'], df['Date'],color ='blue')
ax_5.set_title('Date vs Adj Close')
ax_5.set_xlabel('Adj Close')
ax_5.set_ylabel('Date')

ax_6.scatter(df['Volume'], df['Date'],color ='red')
ax_6.set_title('Date vs Volume')
ax_6.set_xlabel('Volume')
ax_6.set_ylabel('Date')

plt.show()


# In[22]:


df.shape


# In[24]:


df.dropna(inplace = True)


# In[25]:


df.shape


# In[26]:


df.isnull().sum()


# In[27]:


df.info()


# In[28]:


import seaborn as sns


# In[ ]:





# In[33]:


sns.distplot(df['Open'])


# In[34]:


new = np.log(df['Open'])


# In[36]:


sns.distplot(new)


# In[41]:


new_1 = df['Open']**(1/3)


# In[42]:


sns.distplot(new_1)


# In[ ]:




