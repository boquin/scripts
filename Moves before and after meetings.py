#!/usr/bin/env python
# coding: utf-8

# In[ ]:


reers = ["BR","MX","CL","CO","PE","HU","PL","CZ","RU","TR","ZA","TH","MY","KR","CN","ID","IN","PH"]
reerlist = ['BISBBRR Index', 'BISBMXR Index', 'BISBCLR Index', 'BISBCOP Index', 'BISBPER Index', 'BISBHUR Index',
            'BISBPLR Index', 'BISBCZR Index', 'BISBRUR Index', 'BISBTRR Index', 'BISBZAR Index', 'BISBTHR Index',
            'BISBMYR Index', 'BISBKRR Index', 'BISBCNR Index', 'BISBIDR Index', 'BISBINR Index', 'BISBPHR Index']
etflist = ['.XLISPY Index','.XLESPY Index','.XLFSPY Index','.XLUSPY Index','.XLBSPY Index','.XLKSPY Index','.XLPSPY Index']
#list of emfx
emfxlist = ['USDMXN Curncy', 'USDBRL Curncy', 'CHN+1M Curncy', 'CLN+1M Curncy', 'USDPEN Curncy', 'USDTRY Curncy', 'USDZAR Curncy', 'EURPLN Curncy', 'EURHUF Curncy', 'EURCZK Curncy', 'USDRUB Curncy', 'USDEGP Curncy', 'CCN+1M Curncy', 'IHN+1M Curncy', 'IRN+1M Curncy', 'KWN+1M Curncy', 'MRN+1M Curncy', 'SGD+1M Curncy', 'NTN+1M Curncy', 'PPN+1M Curncy', 'THB+1M Curncy']
gbilist2 = ['.GBIBRL Index', '.GBICLP Index', '.GBICNH Index', '.GBICOP Index', '.GBICZK Index', '.GBIDOP Index', '.GBIEGP Index', '.GBIHUF Index', '.GBIIDR Index', '.GBIMYR Index', '.GBIPEN Index', '.GBIMXN Index', '.GBIPLN Index', '.GBIPHP Index', '.GBIRON Index', '.GBIRUB Index', '.GBIZAR Index', '.GBITHB Index', '.GBITRY Index', '.GBIUYU Index']
gbireturnslist = ['JGENBBUU Index', 'JGCLPUSD Index', 'JGENCNTU Index', 'JGENGCUU Index', 'JPMUCZ Index', 'JGENDXPU Index','JPMUHU Index', 'JGIDUUSD Index', 'JGMYUUSD Index', 'JGENPEUU Index', 'JPMUMX Index', 'JGENPDUU Index', 'JGPHUSD Index', 'JGROUUSD Index', 'JGRUUUSD Index', 'JNDCSA Index', 'JGTHUUSD Index', 'JGENTBUU Index', 'JGENUGUU Index']
etflist = ["S5RLST Index","S5ENRS Index","S5COND Index","S5CONS Index","S5INFT Index","S5INDU Index","S5MATR Index","S5HLTH Index","S5UTIL Index","S5FINL Index",".GBIEMFX Index"]

#dictionary for column labels
dice = {'USDMXN Curncy':'MXN', 'USDBRL Curncy':'BRL', 'CHN+1M Curncy':'CLP', 'CLN+1M Curncy':'COP', 'USDPEN Curncy':'PEN', 'USDTRY Curncy':'TRY', 'USDZAR Curncy':'ZAR', 'EURPLN Curncy':'PLN', 'EURHUF Curncy':'HUF', 'EURCZK Curncy':'CZK', 'USDRUB Curncy':'RUB', 'USDEGP Curncy':'EGP', 'CCN+1M Curncy':'CNY', 'IHN+1M Curncy':'IDR', 'IRN+1M Curncy':'INR', 'KWN+1M Curncy':'KRW', 'MRN+1M Curncy':'MYR', 'SGD+1M Curncy':'SGD', 'NTN+1M Curncy':'TWD', 'PPN+1M Curncy':'PHP', 'THB+1M Curncy':'THB'}
dice_gbi = {'.GBIBRL Index': 'Brazil', '.GBICLP Index': 'Chile', '.GBICNH Index': 'China', '.GBICOP Index': 'Colombia', '.GBICZK Index': 'Czech', '.GBIDOP Index': 'DomRep', '.GBIEGP Index': 'Egypt', '.GBIHUF Index': 'Hungary', '.GBIIDR Index': 'Indonesia', '.GBIMYR Index': 'Malaysia', '.GBIPEN Index': 'Peru', '.GBIMXN Index': 'Mexico', '.GBIPLN Index': 'Poland', '.GBIPHP Index': 'Philippines', '.GBIRON Index': 'Romania', '.GBIRUB Index': 'Russia', '.GBIZAR Index': 'SouthAfrica', '.GBITHB Index': 'Thailand', '.GBITRY Index': 'Turkey', '.GBIUYU Index': 'Uruguay'}
dice_gbireturns = {'JGENBBUU Index': 'Brazil', 'JGCLPUSD Index': 'Chile', 'JGENCNTU Index': 'China', 'JGENGCUU Index': 'Colombia', 'JPMUCZ Index': 'Czech', 'JGENDXPU Index': 'DomRep',  'JPMUHU Index': 'Hungary', 'JGIDUUSD Index': 'Indonesia', 'JGMYUUSD Index': 'Malaysia', 'JGENPEUU Index': 'Peru', 'JPMUMX Index': 'Mexico', 'JGENPDUU Index': 'Poland', 'JGPHUSD Index': 'Philippines', 'JGROUUSD Index': 'Romania', 'JGRUUUSD Index': 'Russia', 'JNDCSA Index': 'SouthAfrica', 'JGTHUUSD Index': 'Thailand', 'JGENTBUU Index': 'Turkey', 'JGENUGUU Index': 'Uruguay'}
#list of emfx
emfxlist = ['USDMXN Curncy', 'USDBRL Curncy', 'CHN+1M Curncy', 'CLN+1M Curncy', 'USDPEN Curncy', 'USDTRY Curncy', 'USDZAR Curncy', 'EURPLN Curncy', 'EURHUF Curncy', 'EURCZK Curncy', 'USDRUB Curncy', 'USDEGP Curncy', 'CCN+1M Curncy', 'IHN+1M Curncy', 'IRN+1M Curncy', 'KWN+1M Curncy', 'MRN+1M Curncy', 'SGD+1M Curncy', 'NTN+1M Curncy', 'PPN+1M Curncy', 'THB+1M Curncy']
gbilist2 = ['.GBIBRL Index', '.GBICLP Index', '.GBICNH Index', '.GBICOP Index', '.GBICZK Index', '.GBIDOP Index', '.GBIEGP Index', '.GBIHUF Index', '.GBIIDR Index', '.GBIMYR Index', '.GBIPEN Index', '.GBIMXN Index', '.GBIPLN Index', '.GBIPHP Index', '.GBIRON Index', '.GBIRUB Index', '.GBIZAR Index', '.GBITHB Index', '.GBITRY Index', '.GBIUYU Index']
gbireturnslist = ['JGENBBUU Index', 'JGCLPUSD Index', 'JGENCNTU Index', 'JGENGCUU Index', 'JPMUCZ Index', 'JGENDXPU Index','JPMUHU Index', 'JGIDUUSD Index', 'JGMYUUSD Index', 'JGENPEUU Index', 'JPMUMX Index', 'JGENPDUU Index', 'JGPHUSD Index', 'JGROUUSD Index', 'JGRUUUSD Index', 'JNDCSA Index', 'JGTHUUSD Index', 'JGENTBUU Index', 'JGENUGUU Index']
etflist = ["S5RLST Index","S5ENRS Index","S5COND Index","S5CONS Index","S5INFT Index","S5INDU Index","S5MATR Index","S5HLTH Index","S5UTIL Index","S5FINL Index",".GBIEMFX Index"]

#dictionary for column labels
dice = {'USDMXN Curncy':'MXN', 'USDBRL Curncy':'BRL', 'CHN+1M Curncy':'CLP', 'CLN+1M Curncy':'COP', 'USDPEN Curncy':'PEN', 'USDTRY Curncy':'TRY', 'USDZAR Curncy':'ZAR', 'EURPLN Curncy':'PLN', 'EURHUF Curncy':'HUF', 'EURCZK Curncy':'CZK', 'USDRUB Curncy':'RUB', 'USDEGP Curncy':'EGP', 'CCN+1M Curncy':'CNY', 'IHN+1M Curncy':'IDR', 'IRN+1M Curncy':'INR', 'KWN+1M Curncy':'KRW', 'MRN+1M Curncy':'MYR', 'SGD+1M Curncy':'SGD', 'NTN+1M Curncy':'TWD', 'PPN+1M Curncy':'PHP', 'THB+1M Curncy':'THB'}
dice_gbi = {'.GBIBRL Index': 'Brazil', '.GBICLP Index': 'Chile', '.GBICNH Index': 'China', '.GBICOP Index': 'Colombia', '.GBICZK Index': 'Czech', '.GBIDOP Index': 'DomRep', '.GBIEGP Index': 'Egypt', '.GBIHUF Index': 'Hungary', '.GBIIDR Index': 'Indonesia', '.GBIMYR Index': 'Malaysia', '.GBIPEN Index': 'Peru', '.GBIMXN Index': 'Mexico', '.GBIPLN Index': 'Poland', '.GBIPHP Index': 'Philippines', '.GBIRON Index': 'Romania', '.GBIRUB Index': 'Russia', '.GBIZAR Index': 'SouthAfrica', '.GBITHB Index': 'Thailand', '.GBITRY Index': 'Turkey', '.GBIUYU Index': 'Uruguay'}
dice_gbireturns = {'JGENBBUU Index': 'Brazil', 'JGCLPUSD Index': 'Chile', 'JGENCNTU Index': 'China', 'JGENGCUU Index': 'Colombia', 'JPMUCZ Index': 'Czech', 'JGENDXPU Index': 'DomRep',  'JPMUHU Index': 'Hungary', 'JGIDUUSD Index': 'Indonesia', 'JGMYUUSD Index': 'Malaysia', 'JGENPEUU Index': 'Peru', 'JPMUMX Index': 'Mexico', 'JGENPDUU Index': 'Poland', 'JGPHUSD Index': 'Philippines', 'JGROUUSD Index': 'Romania', 'JGRUUUSD Index': 'Russia', 'JNDCSA Index': 'SouthAfrica', 'JGTHUUSD Index': 'Thailand', 'JGENTBUU Index': 'Turkey', 'JGENUGUU Index': 'Uruguay'}

tickers = ["BR", "MX", "CL", "CO", "PL", "CZ", "HU", "ZA", "TR", "RU", "IL", "CN", "ID", "IN", "KR", "MY", "PH", "SG", "TH", "TW"]

dice = {'A':5, 'A+':4, 'A-':6, 'AA+':1,'AA':2, 'AA-':3, 'B':14, 'B+':13, 'B-':15, 'BB':11, 'BB+':10, 'BB-':12,
       'BBB':8, 'BBB+':7, 'BBB-':9, 'CC':20, 'CCC+':16, 'CCC-':18, 'D':25}


# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


# key_dates = ["02-25-2005","10-29-2008","10-03-2012","10-08-2014","03-04-2020"]
# key_dates = pd.to_datetime(key_dates).tolist()

df_pln = pd.read_csv("df_pln3.csv")
df_pln.head()
df_pln.index = df_pln["date"]
df_pln.index = pd.to_datetime(df_pln.index)
df_pln

df_pln["change"] = np.sign(df_pln["POREANN Index"].diff())
df_pln["priors"]  = df_pln["change"].rolling(200).sum()
df_pln["1st_cut"] = np.where((df_pln["change"] == -1) & (df_pln["priors"] > -1.5), 1, 0)
cuts_dates = df_pln[df_pln["1st_cut"] == 1].index.tolist()
key_dates = pd.to_datetime(cuts_dates).tolist()

masterlist = []
for datez in df_pln.index:
    xlist = [abs((datez - key_dates[x]).days) for x in range(0,len(key_dates))]
    masterlist.append(xlist)
masterlist = [min(masterlist[x]) for x in range(0,len(masterlist) - 1)]

df_new = pd.DataFrame(zip(masterlist), index=df_pln.index[1:])
df_pln["days_since"] = df_new
df_pln["proper"] = np.where(df_pln["days_since"] < df_pln["days_since"].shift(1),df_pln["days_since"] * -1 , df_pln["days_since"])


swapslist = ["PZSW2 Curncy","PZSW5 Curncy","PZSW10 Curncy"]
swaplist2 = []
n_days = 200
for swap in swapslist:
    df_pln[swap + " transformed"] = df_pln[swap].diff(1).cumsum() * 100#rolling(30).sum()
    swaplist2.append(swap + " transformed")    
df_forplot = df_pln.groupby(df_pln["proper"]).median()[swaplist2]
df_forplot = df_forplot + (df_forplot.loc[-n_days] * -1)
df_forplot[(df_forplot.index < n_days) & (df_forplot.index > -n_days)].plot()

#proper corr notation
corr = df.corr(method='pearson') #sort by the x axis

#plotting
mask = np.triu(np.ones_like(corr, dtype=bool)) # Generate a mask for the upper triangle
f, ax = plt.subplots(figsize=(13, 11)) # Set up the matplotlib figure
cmap = sns.diverging_palette(230, 20, as_cmap=True) # Generate a custom diverging 
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, robust=True,annot=True) 
least = corr.median(axis=0).sort_values()[:5]
most = corr.median(axis=0).sort_values()[-5:]

print('least correlated')
print(least.round(2))
print('most correlated')
print(most.round(2))

