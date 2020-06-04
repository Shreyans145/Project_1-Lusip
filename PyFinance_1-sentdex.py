#!/usr/bin/env python
# coding: utf-8

# # Part - 1

# In[1]:


from quantopian.interactive.data.sentdex import sentiment  # library like


# In[2]:


from quantopian.pipeline.filters.morningstar import Q1500US  #importing 1500 stocks from US market


# In[ ]:


type(sentiment)   # blaze is faster than pandas with less features


# In[ ]:


dir(sentiment)  # function available in blaze


# In[5]:


BAC = symbols('BAC').sid
bac_sentiment = sentiment[ (sentiment.sid == BAC)]


# In[6]:


bac_sentiment.head().peek()   # this bac_sentiment.head() is deprecated so add .peek()


# In[7]:


bac_sentiment.peek()  # Blaze version!! 


# In[8]:


import blaze


# In[9]:


bac_sentiment = blaze.compute(bac_sentiment) # return back panda dataframe!


# In[10]:


type(bac_sentiment)  # the type has changed!!


# In[11]:


bac_sentiment.head()


# In[12]:


bac_sentiment.set_index("asof_date", inplace = True)#similar to setting pandas index


# In[13]:


bac_sentiment.head()


# In[14]:


bac_sentiment['sentiment_signal'].plot()


# In[15]:


bac_sentiment = bac_sentiment[ (bac_sentiment.index > '2017-01-01') ]


# In[16]:


bac_sentiment['sentiment_signal'].plot()


# #PART - 2 (PIPELINE) ---> very fast

# In[17]:


from quantopian.pipeline import Pipeline


# In[18]:


def make_pipeline():
    return Pipeline()   # return the pipeline 


# In[19]:


from quantopian.research import run_pipeline 


# In[20]:


result = run_pipeline(make_pipeline(), start_date = '2015-05-05', end_date = '2015-05-05')


# In[21]:


type(result)


# In[22]:


result.head()


# In[23]:


len(result)
from quantopian.pipeline.data.sentdex import sentiment #to get out of interactive data


# In[24]:


def make_pipeline():
    sentiment_factor = sentiment.sentiment_signal.latest
    
    universe = (Q1500US() & sentiment_factor.notnull())  # taking data universe
    
    pipe = Pipeline(columns = {'sentiment': sentiment_factor,
                              'longs': (sentiment_factor >= 4),
                              'shorts': (sentiment_factor <= -2)}) # making a pipeline
    return pipe



# In[55]:


result = run_pipeline(make_pipeline(), start_date = '2015-01-01', end_date = '2016-01-01')


# In[68]:


result.head()
result.fillna(0,inplace = 0)


# # part - 3 video - 18  

# Aplha = measure of returns irrespective to market gains ---> should be high 

# Beta = returns attributed to the market ---> should be neutal 

# In[69]:


assets = result.index.levels[1].unique()
len(assets)  # total companies


# In[70]:


pricing = get_pricing(assets, start_date = '2014-12-01', end_date = '2016-02-01', fields = 'open_price') 


# the get_pricing --- > is directly available in quantopian notebooks!!

# In[75]:


import alphalens
factor_data = alphalens.utils.get_clean_factor_and_forward_returns(factor = result['sentiment'], prices = pricing, quantiles = 2, periods = (1,5,10))

alphalens.tears.create_full_tear_sheet(factor_data)


# In[ ]:




