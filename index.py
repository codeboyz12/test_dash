# from dash import Dash
# from components.layout import layout

# from pathlib import Path
# from threading import Lock
# import json, time

# import numpy as np
# import pandas as pd
# import joblib
# from flask import Flask, render_template
# from flask_socketio import SocketIO
# from dash import Dash, html
# from components.widget import main_chart, recently_table, main_table

# # app = Dash()

# # app.layout = layout()

# app = Dash(__name__)
# app.layout = html.Div([
#     html.H1("Y095 Dashboard"),
#     main_chart("CF Total Historical"),
#     html.H3("Recently data"),
#     recently_table(),
#     html.H3("All data"),
#     main_table(),
# ])

# if __name__ == '__main__':
#     app.run(debug=True)

# index.py
from dash import Dash, html
from components.widget import main_chart, recently_table, main_table

app = Dash(__name__)
app.layout = html.Div([
    html.H1("Y095 Dashboard"),
    main_chart("CF Total Historical"),
    html.H3("Recently data"),
    recently_table(),
    html.H3("All data"),
    main_table(),
])

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
