#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#forecasting central bank moves
import pandas as pd
import bql
import plotly
import seaborn as sns
import numpy as np

bq = bql.Service()
date_range = bq.func.range('2005-06-30',bq.func.today())
price = bq.data.px_last(dates=date_range, frq="M", fill="NA")

#list data
bralist = ["IBREGPMY Index","BZPIIPCY Index","BZCIFYOY Index",".CPIXBRL Index","BZIPYOY% Index","BZRTAMPY Index","BZEAYOY% Index","BRCCTTY Index","USDBRL Curncy",".GBIBRL Index",".CDSBRL Index","IBOV Index",]
bralist_names=["a) wholesale inflation", "a) consumer price inflation", "a) weekly consumer inflation","a) 12m inflation expectations", "b) industrial production","b) retail sales","b) economic acitivity", "b) credit growth","c) USDBRL","c) GBIBRL","c) CDS","c) Bovespa" ]
rublist = [""]

bra_cpi_dice = {"BRFCPC12 Index":"12m inflation expectations",
                "BZPCDIFF Index":"IPCA Diffusion Index",
                "IBREGPMY Index":"Wholesale Inflation",
                "BZPIIPCY Index":"IPCA yoy",
                "BZCIFYOY Index":"FIPE yoy",
                "BRGGBE05 Index":"5y Breakeven Inflation",
                "BZCNCNIS Index":"Capacity Utilization",
                "BRFCGD Index":"GDP expectations",
                "BCOITYOY Index":"Brazil Trimmed Means Core",
                "BRYAFHOM Index":"Brazil Food at Home",
                "BZSTSETA Index":"Policy Rate",
                "BZUETOTN Index":"Unemployment",
                "BZCA%GDP Index":"Current Account (% of GDP)",
                "BZIPYOY% Index":"Industrial Production",
                "BZRTAMPY Index":"Retail Sales",
                "BZPBPR% Index":"Primary Budget"} 
                
clp_cpi_dice = {"CLMRCP11 Index":"12m inflation expectations",
                "CLMRCP23 Index":"24m inflation expectations",
                "CHWNYOY Index":"Wholesale Inflation",
                "CLINNSYO Index":"CPI yoy",
                ".CLPCPI2 U Index":"2yr breakevens",
                "CNPITDRY Index":"Tradable Inflation",
                "CNPISAYO Index":"Core Inflation",
                "CNPINTDY Index":"Non Tradable Inflation",
                "OECLDGBO Index":"Capacity Utilization",
                "CLMRGDPY Index":"Growth Expectations",
                "CLINFOOY Index":"Food Inflation yoy",
                "CHUETOTL Index":"Unemployment Rate",
                "CHTBBALM Index":"Trade Balance",
                "CLRTRYOY Index":"Retail Sales yoy",
                "CHIPTOTY Index":"Industrial Production",
                "CLIMYOYN Index":"IMACEC yoy",
                "CHHNYOY Index":"Nominal Wages yoy",
                "CHOVCHOV Index":"Policy Rate"}

pln_cpi_dice = {"PLIEINFE Index":"12m inflation expectations (diffusion)",
                "POPPIYOY Index":"Wholesale Inflation",
                "POCPIYOY Index":"CPI yoy",
                "POBITYOY Index":"Trimmed Means Inflation",
                "CXEFPLY Index":"Core Inflation",
                "EHCPPDY Index":"Constant Tax Inflation",
                "EUUCPL Index":"Capacity Utilization",
                "EUESPL Index":"Economic Sentiment Indicator",
                "EHUPPL Index":"Unemployment Rate",
                "ECOYBPLN Index":"Trade Balance",
                "PORLYOY Index":"Retail Sales yoy",
                "POISCYOY Index":"Industrial Production",
                "POWGYOY Index":"Nominal Wages yoy",
                "PORERATE Index":"Policy Rate"}

cpi_values = list(pln_cpi_dice.keys()) #change dictionary here
#request data

req = bql.Request(cpi_values, price)
resp = bq.execute(req)
df  = bql.combined_df(resp)

#reshape dataframe
df = df.reset_index()
df.columns = ['id','date','_','price']
df = df.pivot(index='date',columns='id',values='price')

df = df.rename(pln_cpi_dice,axis="columns") #change dictionary here
df = df.sort_index(ascending=True,axis=1)
df = df.fillna(method="bfill")
print(df.columns)

#transformations of variables to then see which one is most relevant
for cols in df:
    colslist1 = []
    new_col = cols + " (inverted)"
    df[new_col] = df[cols] * -1
    colslist1.append(new_col)

    colslist2 = []
    new_col = cols + " relative to last 12m"
    df[new_col] = df[cols] - df[cols].rolling(12).mean()
    colslist2.append(new_col)
    
    colslist3 = []
    new_col = cols + " 3m change"
    df[new_col] = df[cols].diff(periods=3)
    colslist3.append(new_col)
    
df.head()

df["Policy Rate Changes"] = df["Policy Rate"].diff(periods=1)
df["Policy Rate Direction"] = np.sign(df["Policy Rate Changes"])
df["Policy Rate Changes"].plot()

#save file and grouby for averages
df.to_csv("brazil policy rate changes.xls")

#create tables of groupbys 
table = df.groupby(["Policy Rate Changes"]).mean().round(2)
table

tablez = df.groupby(["Policy Rate Direction"]).mean().round(2)
tablez

#abbreviate table to most correlated items to rate changes
corr_series = df.corr()["Policy Rate Changes"]
most_corr = list(corr_series.sort_values().round(2).nlargest(14).keys())
most_corr = most_corr[4:] # i need to remove the obvious top ones like rate changes, rate relative to itself, etc
most_corr
table[most_corr]

#abbreviate table to most correlated items to directional changes
corr_series2 = df.corr()["Policy Rate Direction"]
most_corr2 = list(corr_series2.sort_values().round(2).nlargest(14).keys())
most_corr2 = most_corr2[4:]
most_corr2
tablez[most_corr2]

#pull the most recent data
#most_corr is directional, most_corr2 is binary 
last_data = (df.fillna(method="ffill").tail(1)).squeeze().round(2)
last_data1 = last_data[most_corr]
last_data2 = last_data[most_corr2]
last_data2.sort_values()

#this is for the binary model
tablez_bool = tablez[most_corr2].copy()
for col in tablez_bool:
    tablez_bool[col] = tablez_bool[col] < last_data2[col]
tablez_bool = tablez_bool.astype(int)
mean_score = tablez_bool.sum(axis=0).mean().round(1)

print(f"The score is {mean_score} where Cut = 1, Hold = 2, Hike = 3")

#this creates a probit model timeseries
df_probit = pd.DataFrame()

for item in most_corr:
    x = np.where(df[item] > tablez[item][1], 1, np.where(df[item] < tablez[item][-1], -1, 0))
    df_probit[item] = pd.Series(x)
    
df_probit.index = df.index
df_probit["Policy Rate Changes"] = df["Policy Rate Changes"]
df_probit["mean_score"] = df_probit.mean(axis=1)

import matplotlib.pyplot as plt
forplot = df_probit[["mean_score","Policy Rate Changes"]].iloc[:-1]
plt.plot(forplot)
plt.ylim(-1,1)
plt.ylabel("Mean Score (Blue) vs Policy Rate Changes (orange)")

