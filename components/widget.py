from dash import html, dcc
import plotly.express as px
from preprocess.retrieval import get_data, prediction
import pandas as pd

def combined_df():
    pr = prediction()
    pr["labels"] = "Predicted"

    ac = get_data()
    ac["labels"] = "Actual"

    df = pd.concat([ac, pr])
    return df


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ], style={
        'border': '1px solid black',
        'borderCollapse': 'collapse'
    })

def main_chart():
    data = combined_df()

    fig = px.line(
        data, 
        x="datetime",
        y="value",
        color="labels"
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(l=5, r=5, t=5, b=5),  # ลด margin ให้เต็มพื้นที่
    )

    return dcc.Graph(id='line-chart', figure=fig)