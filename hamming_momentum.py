#!/usr/bin/env python
# coding: utf-8

# In[40]:


#ribbons #momentum

 # Import required libraries
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from scipy.spatial.distance import hamming
from ipynb.fs.full.FXpricelist import full_df

df = full_df(5)

varname = "BRL"
df2 = df[varname]
df2.columns = [varname]
df2 = df2.reset_index()
df2.index = df2["Date"]
df2 = df2.iloc[:,1:]
df2.head()

#df2.plot(alpha=0.8)

df_ma = pd.DataFrame()
rolling_means = []

bottom = 50
top = 210
steps = 10
rangelen = int(top - bottom) / steps + 1

for i in range(bottom,top,steps):
        df_ma[i] = df2[varname].rolling(int(i)).mean()
        rolling_means.append(i)

df_ma.columns = rolling_means

x = stats.rankdata(df_ma.iloc[-500])

dist = hamming(x,np.arange(1,rangelen))
df_rank = pd.DataFrame(index = df_ma.index)
hamming_list = []

for x in np.arange(201,df_ma.shape[0]):
    rankz = stats.rankdata(df_ma.iloc[x])
    dist = hamming(rankz,np.arange(1,rangelen))
    hamming_list.append(dist)
    
df_rank = df_rank.iloc[201:]
    
df_rank["hamming"] = hamming_list
final_df = df_rank.merge(df2,left_index=True,right_index=True)


final_df_trimmed = final_df.iloc[-365*10:]
sns.lineplot(data=final_df_trimmed["hamming"], color="grey")
ax2 = plt.twinx()
sns.lineplot(data=final_df_trimmed[varname], color="b", ax=ax2)


# In[ ]:




