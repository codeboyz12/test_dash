from dash import html
import pandas as pd
from components.widget import main_chart

df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')

def layout():
    return html.Div([
        html.H1(children='Y095 Dashboard'),
        html.Div([
            main_chart()
        ]),
    ], style={
        'fontFamily': 'sans-serif'
    })