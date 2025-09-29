from dash import html, dcc, dash_table
import plotly.express as px
from preprocess.retrieval import *
import pandas as pd

def combined_df():
    pr = prediction()
    pr["labels"] = "Predicted"

    ac = get_data()
    ac["labels"] = "Actual"

    df = pd.concat([ac, pr])
    return df

def recently_table():
    df = recently_mock()
    df = df.head(1)  # แสดงแค่ 1 แถวเหมือนเดิม

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in df.columns],

        style_cell={
            "padding": "8px",
            "textAlign": "center",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={
            "backgroundColor": "#4a4a4a",
            "fontWeight": "bold",
            "color": "white",
        },

        style_data_conditional = [
            *[
                {
                    "if": {
                        "filter_query": f"{{{col}}} < 45",
                        "column_id": col
                    },
                    "color": "green",
                    "fontWeight": "bold"
                }
                for col in df.columns
            ],

            # ช่วงที่ 2: ค่ามากกว่า 55 => สีแดง
            *[
                {
                    "if": {
                        "filter_query": f"{{{col}}} > 55",
                        "column_id": col
                    },
                    "color": "red",
                    "fontWeight": "bold"
                }
                for col in df.columns
            ],
        ]
    )

def generate_table(dataframe=recently_mock(), max_rows=10):
    cell_style = {
        'border': '1px solid black',
        'textAlign': 'center',
        'padding': '5px 5px'
    }

    header_cell_style = {
        'border': '1px solid black',
        'textAlign': 'center',
        'padding': '10px 5px',
        'color': 'white',
        'backgroundColor': '#003354'
    }

    return html.Table(
        [html.Thead(html.Tr([
            html.Th(col, style=header_cell_style) for col in dataframe.columns
        ]))] +

        [html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[0][col], style=cell_style) 
                for col in dataframe.columns
            ])
        ])],

        style={
            'border': '1px solid black',
            'borderCollapse': 'collapse',
            'textAlign': 'center',
            'margin': '0px 15px'
        }
    )

def main_chart(title):
    data = combined_df()

    fig = px.line(
        data, 
        x="datetime",
        y="value",
        color="labels",
        title=title,
        color_discrete_sequence=["#3f89c3", "#e0a8a8"]
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        xaxis=dict(
            showgrid=False,
            gridcolor="#0fa3ff"
        ),
        paper_bgcolor="#a8c2e0",
        margin=dict(l=0, r=0, t=25, b=0),  # ลด margin ให้เต็มพื้นที่
        legend=dict(
            orientation="v",        
            y=0.83,                    
            yanchor="bottom",      
            x=0.94,                  
            xanchor="center",
            bgcolor="rgba(255,255,255,0.5)"  # พื้นหลังโปร่งใสครึ่งหนึ่ง
        ),
        font=dict(
            family="Arial, sans-serif",  # ฟอนต์
            size=13,                     # ขนาดตัวอักษร
            color="#004f7d"                # สีข้อความทั้งหมด (legend, title, tick)
        )
    )

    return dcc.Graph(id='line-chart', figure=fig)