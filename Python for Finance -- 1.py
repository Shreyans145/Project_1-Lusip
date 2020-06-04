#!/usr/bin/env python
# coding: utf-8

# # Part - 1 

# In[37]:


from sklearn.model_selection import cross_validate
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


# In[42]:


from sklearn.model_selection import train_test_split
import datetime as dt
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc #an alternate to use previous version
import matplotlib.dates as mdates


# In[30]:


from matplotlib import style
from collections import Counter


# In[4]:


import pandas as pd
import numpy


# In[5]:


import pandas_datareader.data as web


# In[6]:


style.use('ggplot')


# In[7]:


#start = dt.datetime(2000,1,1)
#end = dt.datetime(2020,5,31)


# In[8]:


#df = web.DataReader('RELIANCE.NS','yahoo',start,end)


# In[9]:


#df.to_csv('Reliance_1.csv')


# # Part - 2

# In[10]:


df = pd.read_csv('Reliance_1.csv', parse_dates = True,index_col =0) 


# In[11]:


#print(df.head())


# In[12]:


df['Adj Close'].plot()
plt.show()


# # Part - 3 

# In[13]:


df['100-Moving Avg.'] = df['Adj Close'].rolling(window=100, min_periods = 0).mean() #Does rolling average for 100 but min_periods ensure that it works for minimum of 0 previous enteries


# In[14]:


print(df.head())


# In[15]:


print(df.tail())


# In[16]:


ax1 = plt.subplot2grid((6,1),(0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0), rowspan=1, colspan=1, sharex = ax1)
#sharex = ax1 so thet both share same x-axis
ax1.plot(df.index,df['Adj Close'])
ax1.plot(df.index,df['Adj Close'])
ax2.bar(df.index,df['Volume'])

plt.show()


# # Part - 4

# In[17]:


df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)


# In[18]:


print(df_ohlc.head())


# In[19]:


df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)


# In[20]:


print(df_ohlc.head())


# In[21]:


ax1 = plt.subplot2grid((6,1),(0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0), rowspan=1, colspan=1, sharex = ax1)
ax1.xaxis_date()# xaxis_date. Axes. xaxis_date (self, tz=None),
#Sets up x-axis ticks and labels that treat the x data as dates
candlestick_ohlc(ax1, df_ohlc.values, width =2, colorup='g')  # preparing candlestick graph 
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)  # for comparison of volume!!
plt.show()


# # Part - 5

# In[22]:


import bs4 as bs 


# In[23]:


import pickle 
import requests


# # Part-5 and part-6 are combined 
# 

# In[24]:


import bs4 as bs
import datetime as dt
import os
from pandas_datareader import data as pdr
import pickle
import requests
import yfinance as yf

yf.pdr_override

def save_sp500_tickers():#function for storing S&P 500 names
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:#starting from 1 row since 0th rows are heading
        ticker = row.findAll('td')[0].text.replace('.', '-')#  Originally removed ---.text.replace('.', '-')   starting from 0th column
        ticker = ticker[:-1]# new addition
        tickers.append(ticker) #adding to ticker list
#with statement in Python is used in exception handling to make the code cleaner and much more readable            
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    #print(tickers)    
    return tickers


# save_sp500_tickers()
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    start = dt.datetime(2015,1,1)
    end = dt.datetime.now()
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):# checking for folder
            df = pdr.get_data_yahoo(ticker, start, end)# getting data
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


save_sp500_tickers()
get_data_from_yahoo()


# # part-7

# In[25]:


import pickle
def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
    
    main_df = pd.DataFrame()
    
    for count,ticker in enumerate(tickers): # so to track the process eg = 1.AAPL, 2.MMM
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date',inplace = True) # setting index of the dataframe 
        
        df.rename(columns = {'Adj Close':ticker}, inplace = True)  # changing Adj Close to Ticker Name
        df.drop(['Open','High','Low','Close','Volume'],1,inplace = True)
        
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how = 'outer')  # Outer Join
            
        if count% 10 == 0:  #just printing the progress
            print(count)
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

#compile_data()
    


# # Part - 8

# In[26]:


def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    #pd.DataFrame(df['XRX']).plot()
    df_corr = df.corr()
    print(df_corr.head())
    
    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    heatmap = ax.pcolor(data, cmap='RdBu') #color grid
    fig.colorbar(heatmap)
    ax.set_xticks(numpy.arange(data.shape[0]) + 0.5, minor = False)# arranging ticks at every 0.5,2.5
    ax.set_yticks(numpy.arange(data.shape[1]) + 0.5, minor = False)
    ax.invert_yaxis()# removing top space
    ax.xaxis.tick_top()#x axis at top 
    
    column_lables = df_corr.columns # labels for column
    row_labels = df_corr.index  #labels for row
    
    ax.set_xticklabels(column_lables)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()# to clear the confusion
    plt.show()
    
#visualize_data()


# # Part - 9 

# In[27]:


def process_data_for_labels(ticker):
    hm_days = 7 
    df = pd.read_csv('sp500_joined_closes.csv',index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace = True) # filling null valiues with 0
    
    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker])/df[ticker] #(new - old )/old 
        
    df.fillna(0, inplace = True)
    return tickers, df

process_data_for_labels('XOM')


# # Part - 10

# In[52]:


def buy_sell_hold(*args):   # we can pass mulitple parameters
    cols = [c for c in args]
    requirement = 0.025
    
    for col in cols:
        if col > requirement:    # up 2 percent
            return 1
        if col < -requirement:  # Stop-Loss down 2 percent
            return -1
    return 0    


# # Part - 11 (mapping function)

# In[33]:


def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    
    #iterating to the ticker for 7 days
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,df['{}_1d'.format(ticker)],df['{}_2d'.format(ticker)],df['{}_3d'.format(ticker)],df['{}_4d'.format(ticker)],df['{}_5d'.format(ticker)],df['{}_6d'.format(ticker)],df['{}_7d'.format(ticker)]))
    vals = df['{}_target'.format(ticker)].values.tolist()# storing ticker values for 7 days
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))
    
    df.fillna(0, inplace = True)
    df = df.replace([numpy.inf, -numpy.inf], numpy.nan) # replacing any +/- infinity change
    df.dropna(inplace = True)
    
    df_vals = df[[ticker for ticker in tickers]].pct_change()#normalize and percent change from yesterday
    df_vals = df_vals.replace([numpy.inf,-numpy.inf],0)
    df_vals.fillna(0, inplace = True)
    
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    
    return X, y, df

extract_featuresets('XOM')


# # Part-12

# In[53]:


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    
    #clf = neighbors.KNeighborsClassifier()
    
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),('knn', neighbors.KNeighborsClassifier()), ('rfor', RandomForestClassifier())])
    
    
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test) # To see how we fit the data
    predictions = clf.predict(X_test)
    print(confidence)
    print('Predicted spread', Counter(predictions))
    
    return confidence

do_ml('BAC')
# warning in outpur since we have to increase the number of iterations for svm


# # Part - 13 ---> Quantopian 1 

# In[ ]:




