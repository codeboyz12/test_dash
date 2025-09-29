from dash import html
from components.widget import *

def layout():
    return html.Div([
        html.H1(
            children='Y095 Dashboard',
            style={
                "margin": "0px",
                "color": "#004f7d",
                "backgroundColor": "#a8c2e0",
                "padding": "10px 20px",
                "marginBottom": "30px",
            }
            ),
        html.Div([
            main_chart(title="CF Total Historical"),

            html.H4(
                children="Recntly data",
                style={
                    "margin": "0px",
                    "backgroundColor": "#a8c2e0",
                    "padding": "20px",
                    "color": "#0075b2"
                }
            ),
            recently_table()

        ], style={
            "height": "100vh",
            "display": "flex",
            "flexDirection": "column",
            "rowGap": "20px"
        }),
    ], style={
        "backgroundColor": "#a8c2e0",  # สีพื้นหลังทั้งหน้า
        "height": "100vh",
        "margin": "0",
        "padding": "0",
        "fontFamily": "sans-serif",
    })