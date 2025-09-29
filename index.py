from dash import Dash
from components.layout import layout

app = Dash()

app.layout = layout()

if __name__ == '__main__':
    app.run(debug=True)
