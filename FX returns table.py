#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
from datetime import date
import yfinance as yf

ccylist = ['BRL','MXN','CLP','ZAR','TRY','PLN','HUF','CZK','CNY','KRW','SGD','MYR','IDR','INR','PHP','THB']

def yname(rawname):
    adj_name = "USD" + rawname + "=x"
    return(adj_name)

def get_raw_data(name,years):
    #pull data from yfinance
    start_date = datetime.datetime.now() - datetime.timedelta(days=365*years)
    end_date = date.today()
    ticker = yfinance.Ticker(name)
    df = ticker.history(interval="1d",start=start_date,end=end_date)

    #clean dataframe
    df = df.drop(["Volume", "Dividends","Stock Splits"], axis=1)
    cols = df.columns[df.dtypes.eq(object)]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=0)
    return(df)

adj_ccylist = [yname(x) for x in ccylist]

years = 1
start_date = datetime.datetime.now() - datetime.timedelta(days=365*years)
end_date = date.today()

data = yf.download(adj_ccylist, start=start_date,end=end_date, group_by='tickers')
data


# In[2]:


new_df = [data[x.upper()]["Close"] for x in adj_ccylist]


# In[3]:


close_df = pd.DataFrame(new_df).T
close_df.columns = ccylist
close_df = close_df.bfill(axis=0)
close_df = close_df.ffill()


# In[4]:


periods = [5,20,60,52*5]
period_labels = ["spot","1w %","1m %","3m %","1y %"]
returns = []
for x in periods:
  z = round(((close_df.iloc[-1] - close_df.iloc[-x]) / close_df.iloc[-1]) * -100,1)
  #final - initial divided by final
  returns.append(z)

returns.append(close_df.iloc[-1].round(2))
returns = returns[-1:] + returns[:-1]
returns_df = pd.DataFrame(returns).T
returns_df.columns = period_labels
returns_df = returns_df.sort_values("1m %",ascending=False)


# In[5]:


def color_df(val):
    color = "red" if val < 0 else "green"
    return(f"color: {color}")

final_df = returns_df.style.applymap(color_df)
final_df.to_html('fx_returns.html')
final_df


# In[24]:


def calc_macd(df):
    #calculate macds
    df["k"] = df.ewm(span=12, adjust=False, min_periods=12).mean()
    df["d"] = df.ewm(span=26, adjust=False, min_periods=12).mean()

    # Get the 9-Day EMA of the MACD for the Trigger line
    df["macd"] = df["k"] - df["d"]  #ema diff
    df["macd_s"] = df["macd"].ewm(span=9, adjust=False, min_periods=9).mean() #ema of ema diff
    df["macd_h"] =  df["macd"] - df["macd_s"]
    return(df)

def ewm(df):
    df.


# In[30]:


close_df.ewm(span=12, adjust=False, min_periods=12).mean()


# In[ ]:




