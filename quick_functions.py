def get_timestamp():
    """This prints out a timestamp. Useful for quick way to see runtime."""
    import datetime
    now = datetime.datetime.now()
    hours = str(now.hour).zfill(2)
    minutes = str(now.minute).zfill(2)
    seconds = str(now.second).zfill(2)
    print(f"{hours}:{minutes}:{seconds}")


def create_chart_grid(df, rows, cols, min_y=None, max_y=None, fig_title=None):
    """This creates a chart grid of x rows and y columns across. Useful to see all currency charts at once."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=rows, cols=cols, start_cell="top-left", subplot_titles=(df.columns))
    row_i = 1
    col_i = 1
    for columns in df:
        fig.add_trace(go.Scatter(x=df.index, y=df[columns]), row=row_i, col=col_i)
        if col_i < cols:
            col_i += 1
        else:
            col_i = 1
            row_i += 1
    fig.update_layout(height=1000, width=1000, showlegend=False)
    fig.update_annotations(font_size=10)
    fig.update_layout(title_text=f"{fig_title}")
    fig.show()


# def create_scatter(df,x_col,y_col,labels,title=None):
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     sns.scatterplot(data=df, x=x_col, y=y_col)
#     plt.xlabel(x_col)
#     plt.ylabel(y_col)
#     plt.title(title)
#     for i in range(len(df)):
#         plt.annotate(labels[i], (df.iloc[:,0][i], df.iloc[:,1][i]))
#     plt.show()


def create_scatter(df, x_col, y_col, labels, title=None):
    """quickly create a scatterplot"""
    import plotly.graph_objects as go
    import pandas as pd
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers+text',
        text=labels
    ))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col)
    fig.show()


def delete_outliers(df, z_threshold=3):
    """delete datapoints that exceed z standard deviations"""
    import pandas as pd
    import numpy as np
    from scipy import stats
    cleaned_df = df.copy()
    tempdf = pd.DataFrame(cleaned_df)
    for column in cleaned_df:
        z_scores = np.abs(stats.zscore(cleaned_df[column]))
        z_threshold = z_threshold
        outlier_indices = np.where(z_scores > z_threshold)[0]
        for idx in outlier_indices:
            if idx > 0:
                cleaned_df[column].iloc[idx] = cleaned_df[column].iloc[idx - 1]
    return (cleaned_df)


def get_oecd_data(codelist):
    """quick way to access OECD data via PandaSDMX. Input as list and use OECD acroynms"""
    import pandasdmx as sdmx
    from datetime import date
    get_timestamp()
    oecd = sdmx.Request('OECD', timeout=200)
    for codes in codelist:
        resp = oecd.data(
            codes,
            params={'startPeriod': '2000'})
    print(f"got response for {codes}")
    get_timestamp()
    df = resp.to_pandas().to_frame()
    df.columns.name = None
    df.to_csv(f"{codes}_{date.today()}.csv")
    print(f"done with {codes}")
    get_timestamp()
    return (df)


def get_countries(df):
    """
    this might be too common to the oecd one
    """
    countries = ['AUS', 'BRA', 'CAN', 'CHE', 'CHL', 'CHN', 'COL', 'CZE', 'GBR', 'HUN', 'IDN', 'IND',
                 'KOR', 'MEX', 'NOR', 'NZL', 'POL', 'USA', 'ZAF']
    select_countries = [x for x in countries if x in df.columns]
    return (select_countries)


def decompose_seasonality(timeseries, label):
    """Quickly tease out trend vs seasonality. Currently uses most naive seasonal calculation."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    from plotly.subplots import make_subplots

    decomposition = seasonal_decompose(timeseries)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    ts_log = timeseries

    fig = make_subplots(rows=4, cols=1)

    fig.add_scatter(x=ts_log.index, y=ts_log, mode='lines', row=1, col=1, showlegend=False)
    fig.add_scatter(x=trend.index, y=trend, mode='lines', row=2, col=1, showlegend=False)
    fig.add_scatter(x=seasonal.index, y=seasonal, mode='lines', row=3, col=1, showlegend=False)
    fig.add_scatter(x=residual.index, y=residual, mode='lines', row=4, col=1, showlegend=False)

    fig.update_yaxes(title_text="Timeseries", row=1, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)
    fig.update_yaxes(title_text="Seasonality", row=3, col=1)
    fig.update_yaxes(title_text="Residual", row=4, col=1)

    fig.update_layout(title=f'Decomposition of {label}', width=800, height=600)
    fig.show()

    return (decomposition)


def decompose_seasonality_stl(timeseries, label):
    from statsmodels.tsa.seasonal import STL
    from plotly.subplots import make_subplots

    stl = STL(timeseries, seasonal=13)
    decomposition = stl.fit()
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    ts_log = timeseries
    fig = make_subplots(rows=4, cols=1)

    fig.add_scatter(x=ts_log.index, y=ts_log, mode='lines', row=1, col=1, showlegend=False)
    fig.add_scatter(x=trend.index, y=trend, mode='lines', row=2, col=1, showlegend=False)
    fig.add_scatter(x=seasonal.index, y=seasonal, mode='lines', row=3, col=1, showlegend=False)
    fig.add_scatter(x=residual.index, y=residual, mode='lines', row=4, col=1, showlegend=False)

    fig.update_yaxes(title_text="Timeseries", row=1, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)
    fig.update_yaxes(title_text="Seasonality", row=3, col=1)
    fig.update_yaxes(title_text="Residual", row=4, col=1)

    fig.update_layout(title=f'Decomposition of {label}', width=800, height=600)
    fig.show()

    return (decomposition)


def get_fred(dataseries, years=5):
    '''Get data from Fred database. Return df.'''
    import datetime
    from pandas_datareader import data
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365 * years)
    df = data.DataReader(dataseries, 'fred', start_date, end_date)
    return (df)


def full_df(years, exceptions=0):
    '''Get FX dataframe from Yahoo Finance'''
    import pandas as pd
    import datetime
    from datetime import date
    import yfinance as yf

    def yname(rawname):
        adj_name = "USD" + rawname + "=x"
        adj_name = rawname + "USD" + "=x"  # for returns in local currency keep this ungrayed
        return (adj_name)

    def get_raw_data(name, years):
        start_date = datetime.datetime.now() - datetime.timedelta(days=365 * years)
        end_date = date.today()
        ticker = yf.Ticker(name)
        df = ticker.history(interval="1d", start=start_date, end=end_date)

        # clean dataframe
        df = df.drop(["Volume", "Dividends", "Stock Splits"], axis=1)
        cols = df.columns[df.dtypes.eq(object)]
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=0)
        return (df)

    ccylist = ['BRL', 'MXN', 'CLP', 'ZAR', 'TRY', 'PLN', 'HUF', 'CZK', 'CNY', 'KRW', 'SGD', 'MYR', 'IDR', 'INR', 'PHP',
               'THB',
               'EUR', 'JPY', 'GBP', 'CAD', 'AUD', 'NZD', 'SEK', 'NOK', 'COP']
    adj_ccylist = [yname(x) for x in ccylist]
    start_date = datetime.datetime.now() - datetime.timedelta(days=365 * years)
    end_date = date.today()
    data = yf.download(adj_ccylist, start=start_date, end=end_date, group_by='tickers')
    new_df = [data[x.upper()]["Close"] for x in adj_ccylist]
    close_df = pd.DataFrame(new_df).T
    close_df.columns = ccylist
    close_df = close_df.bfill(axis=0)
    close_df = close_df.ffill()
    if exceptions == 1:
        exceptions_list = ["EUR", "GBP", "NZD", "AUD"]
        close_df[exceptions_list] = 1 / close_df[exceptions_list]
    return (close_df)