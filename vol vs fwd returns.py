#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#color codeds scatterplots
# Import required libraries
import bql
import bqviz as bqv
from bqplot import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import plotly.express as px

#instatiate object
bq = bql.Service()

#define date range
date_range = bq.func.range('2010-01-01', bq.func.today())

#pass the date range and other parameters
price = bq.data.px_last(dates=date_range, frq="W")

#define universe
univ = ["JPMVXYEM Index", ".GBIEMTR Index"]

#request
req= bql.Request(univ, price)
resp = bq.execute(req)
df1 = bql.combined_df(resp)

df1.reset_index(inplace=True)
df1.columns = ["types","dates","none","Currency"]               
df1.head()

df = df1.pivot(index="dates",columns="types",values="Currency")
df.columns = ["GBIEMTR","JPMVXYEM"]
df["GBIEMTR"] = df["GBIEMTR"].pct_change(52) * 100
df.head()

df["GBIEMTR shifted"] = df["GBIEMTR"].shift(-52)
df
df_trim = df.iloc[104:]
n_buckets = 10
df_trim["percentiles"] = pd.cut(df_trim["JPMVXYEM"],n_buckets,labels=range(0,n_buckets))
df_trim.groupby(df_trim["percentiles"]).mean()["GBIEMTR shifted"].plot(kind="bar")
days = 52 * 10
df_plot = df.iloc[-days:][["JPMVXYEM","GBIEMTR shifted"]]
df_plot.columns = ["EM FX implied vol","GBI-EM total returns"]
fig = px.scatter(df_plot,x="EM FX implied vol",y="GBI-EM total returns",
                #color_discrete_sequence=["red", "blue"],
                width=600,
                height=600,
                color=df_plot.index.year,
                color_continuous_scale=px.colors.sequential.Turbo
                )
fig.add_vline(x=df_plot["EM FX implied vol"].iloc[-1],annotation_text="current level")
fig.show()

