#!/usr/bin/env python
# coding: utf-8

# https://github.com/jha-prateek/Stock-Prediction-RNN

# In[15]:


import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



# In[21]:


yahoo = pd.read_csv('RELIANCE.NS.csv',index_col=['Date'])


# In[17]:


print(yahoo)


# In[19]:


# preparing input features
yahoo = yahoo.drop(['Volume'], axis=1)


# In[23]:


yahoo = yahoo[['Open', 'Low', 'High', 'Close']]


# In[25]:


yahoo_shift = yahoo.shift(-1)
label = yahoo_shift['Close']


# In[26]:


yahoo.drop(yahoo.index[len(yahoo)-1], axis=0, inplace=True)
label.drop(label.index[len(label)-1], axis=0, inplace=True)


# In[27]:


x, y = yahoo.values, label.values


# In[28]:


x_scale = MinMaxScaler()
y_scale = MinMaxScaler()

X = x_scale.fit_transform(x)
Y = y_scale.fit_transform(y.reshape(-1,1))


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
X_train = X_train.reshape((-1,1,4))
X_test = X_test.reshape((-1,1,4))


# In[30]:


model_name = 'stock_price_GRU'


# In[31]:


model = Sequential()
model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(1, 4)))
model.add(Dropout(0.2))
model.add(GRU(units=256))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')


# In[32]:


model.fit(X_train,y_train,batch_size=250, epochs=500, validation_split=0.1, verbose=1)
model.save("{}.h5".format(model_name))
print('MODEL-SAVED')

score = model.evaluate(X_test, y_test)
print('Score: {}'.format(score))
yhat = model.predict(X_test)
yhat = y_scale.inverse_transform(yhat)
y_test = y_scale.inverse_transform(y_test)
plt.plot(yhat[-100:], label='Predicted',color = )
plt.plot(y_test[-100:], label='Ground Truth')
plt.legend()
plt.show()


# In[34]:


score = model.evaluate(X_test, y_test)
print('Score: {}'.format(score))
yhat = model.predict(X_test)
yhat = y_scale.inverse_transform(yhat)
y_test = y_scale.inverse_transform(y_test)
plt.plot(yhat[-100:], label='Predicted',color = 'blue')
plt.plot(y_test[-100:], label='Ground Truth', color = 'green')
plt.legend()
plt.show()


# In[ ]:




