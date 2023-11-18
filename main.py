import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from quick_functions import *

mydata = pd.read_csv("/Users/macproajb/Downloads/fx_historical_data.csv",index_col="Date")

def plot_the_chart(df, ccy):
    df_200d = round((df / df.rolling(200).mean() - 1) * 100, 2)
    df_edges = df_200d.describe().loc[["25%", "75%"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_200d.index, y=df_200d[ccy], mode='lines', name='X'))
    fig.add_shape(
        dict(
            type='line',
            x0=df_200d.index[0],
            x1=df_200d.index[-1],
            y0=df_edges[ccy].loc["25%"],
            y1=df_edges[ccy].loc["25%"],
            line=dict(color='red', dash='dash'),
        )
    )
    fig.add_shape(
        dict(
            type='line',
            x0=df_200d.index[0],
            x1=df_200d.index[-1],
            y0=df_edges[ccy].loc["75%"],
            y1=df_edges[ccy].loc["75%"],
            line=dict(color='red', dash='dash'),
        )
    )

    # Update layout with labels and title
    fig.update_layout(
        xaxis_title='%',
        yaxis_title=ccy,
        title=f'{ccy} distance from 200d',
        width=800,
        height=600
    )


    filename = f"200d_charts/{ccy}_plot.html"
    pio.write_html(fig, file=filename)
    return filename


# Generate and save HTML files for each column
html_files = {col: plot_the_chart(mydata, col) for col in mydata.columns}

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Distance from 200d moving average"),
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in mydata.columns],
        value=mydata.columns[0],
        style={'width': '50%'}
    ),
    html.Iframe(id='html-chart', style={'width': '60%', 'height': '500px'})
])


# Callback to update the HTML chart based on the dropdown selection
@app.callback(
    Output('html-chart', 'srcDoc'),
    [Input('column-dropdown', 'value')]
)
def update_html_chart(selected_column):
    with open(html_files[selected_column], 'r') as file:
        return file.read()


# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)

get_timestamp()