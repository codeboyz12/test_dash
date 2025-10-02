from dash import html, dcc, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from preprocess.retrieval import prediction, get_data, recently_mock


def combined_df() -> pd.DataFrame:
    pr = prediction().copy()
    ac = get_data().copy()
    pr["labels"] = "Predicted"
    ac["labels"] = "Actual"

    pr["datetime"] = pd.to_datetime(pr["datetime"])
    ac["datetime"] = pd.to_datetime(ac["datetime"])

    keep_cols = lambda df: [c for c in ["datetime","value","labels","y_lo","y_hi"] if c in df.columns]
    df = pd.concat([ac[keep_cols(ac)], pr[keep_cols(pr)]], ignore_index=True)
    return df.sort_values("datetime").reset_index(drop=True)

def recently_table():
    df = recently_mock().copy().head(1)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_cell={"padding":"8px","textAlign":"center","whiteSpace":"normal","height":"auto"},
        style_header={"backgroundColor":"#4a4a4a","fontWeight":"bold","color":"white"},
        style_data_conditional=[
            *[{"if":{"filter_query":f"{{{col}}} < 45","column_id":col},
                "color":"green","fontWeight":"bold"} for col in df.columns],
            *[{"if":{"filter_query":f"{{{col}}} > 55","column_id":col},
                "color":"red","fontWeight":"bold"} for col in df.columns],
        ],
    )

def main_table():
    df = recently_mock().copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_cell={'padding':'8px','textAlign':'center','whiteSpace':'normal','height':'auto','overflow':'hidden'},
        style_header={'backgroundColor':'#4a4a4a','fontWeight':'bold','color':'white','border':'1px solid #d0d0d0'},
        style_data={'backgroundColor':'white','color':'black','border':'1px solid #d0d0d0'},
        style_data_conditional=[
            *[{"if":{"filter_query":f"{{{col}}} < 45","column_id":col},
                "color":"green","fontWeight":"bold"} for col in df.columns],
            *[{"if":{"filter_query":f"{{{col}}} > 55","column_id":col},
                "color":"red","fontWeight":"bold"} for col in df.columns],
        ],
        page_action='native', page_size=10, sort_action='native', filter_action='native',
    )

def main_chart(title: str):
    data = combined_df()
    have_band = {"y_lo","y_hi"}.issubset(set(data.columns)) and (data["labels"]=="Predicted").any()

    if not have_band:
        fig = px.line(
            data, x="datetime", y="value", color="labels", title=title,
            category_orders={"labels":["Actual","Predicted"]},
            color_discrete_map={"Actual":"#3f89c3","Predicted":"#e07a7a"},
        )
    else:
        dfA = data[data["labels"]=="Actual"]
        dfP = data[data["labels"]=="Predicted"]
        fig = go.Figure()
        if "y_lo" in dfP and "y_hi" in dfP:
            fig.add_trace(go.Scatter(x=dfP["datetime"], y=dfP["y_hi"], mode="lines",
                                     line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=dfP["datetime"], y=dfP["y_lo"], mode="lines",
                                     fill="tonexty", name="Predicted CI", line=dict(width=0),
                                     fillcolor="rgba(224,122,122,0.18)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=dfA["datetime"], y=dfA["value"], mode="lines",
                                 name="Actual", line=dict(color="#3f89c3", width=2)))
        fig.add_trace(go.Scatter(x=dfP["datetime"], y=dfP["value"], mode="lines",
                                 name="Predicted", line=dict(color="#e07a7a", width=1.5)))
        fig.update_layout(title=title)

    fig.update_layout(
        xaxis_title=None, yaxis_title=None, xaxis=dict(showgrid=False),
        paper_bgcolor="#a8c2e0", margin=dict(l=0,r=0,t=25,b=0),
        legend=dict(orientation="v", y=0.83, yanchor="bottom", x=0.94, xanchor="center",
                    bgcolor="rgba(255,255,255,0.5)"),
        font=dict(family="Arial, sans-serif", size=13, color="#004f7d"),
    )
    fig.update_xaxes(tickformat="%Y-%m-%d %H:%M")
    return dcc.Graph(id="line-chart", figure=fig)




# from dash import html, dcc, dash_table
# import plotly.express as px
# from preprocess.retrieval import *
# import pandas as pd

# def combined_df():
#     pr = prediction()
#     pr["labels"] = "Predicted"

#     ac = get_data()
#     ac["labels"] = "Actual"

#     df = pd.concat([ac, pr])
#     return df


# def recently_table():
#     df = recently_mock()
#     df = df.head(1)  # แสดงแค่ 1 แถวเหมือนเดิม

#     return dash_table.DataTable(
#         data=df.to_dict("records"),
#         columns=[{"name": i, "id": i} for i in df.columns],

#         style_cell={
#             "padding": "8px",
#             "textAlign": "center",
#             "whiteSpace": "normal",
#             "height": "auto",
#         },
#         style_header={
#             "backgroundColor": "#4a4a4a",
#             "fontWeight": "bold",
#             "color": "white",
#         },

#         style_data_conditional = [
#             *[
#                 {
#                     "if": {
#                         "filter_query": f"{{{col}}} < 45",
#                         "column_id": col
#                     },
#                     "color": "green",
#                     "fontWeight": "bold"
#                 }
#                 for col in df.columns
#             ],

#             # ช่วงที่ 2: ค่ามากกว่า 55 => สีแดง
#             *[
#                 {
#                     "if": {
#                         "filter_query": f"{{{col}}} > 55",
#                         "column_id": col
#                     },
#                     "color": "red",
#                     "fontWeight": "bold"
#                 }
#                 for col in df.columns
#             ],
#         ]
#     )

# def main_table():
#     df = recently_mock()

#     # สร้าง DataTable พร้อม style
#     return dash_table.DataTable(
#         data=df.to_dict('records'),
#         columns=[
#             {"name": i, "id": i} for i in df.columns
#         ],

#         # ขนาดพื้นฐานของ cell / alignment / overflow
#         style_cell={
#             'padding': '8px',
#             'textAlign': 'center',
#             'whiteSpace': 'normal',
#             'height': 'auto',
#             'overflow': 'hidden',
#         },

#         # style สำหรับ header
#         style_header={
#             'backgroundColor': '#4a4a4a',
#             'fontWeight': 'bold',
#             'color': 'white',
#             'border': '1px solid #d0d0d0'
#         },

#         # style สำหรับ data cells (default)
#         style_data={
#             'backgroundColor': 'white',
#             'color': 'black',
#             'border': '1px solid #d0d0d0'
#         },

#         # conditional styling เช่น highlight ตามเงื่อนไข
#         style_data_conditional = [
#             *[
#                 {
#                     "if": {
#                         "filter_query": f"{{{col}}} < 45",
#                         "column_id": col
#                     },
#                     "color": "green",
#                     "fontWeight": "bold"
#                 }
#                 for col in df.columns
#             ],

#             # ช่วงที่ 2: ค่ามากกว่า 55 => สีแดง
#             *[
#                 {
#                     "if": {
#                         "filter_query": f"{{{col}}} > 55",
#                         "column_id": col
#                     },
#                     "color": "red",
#                     "fontWeight": "bold"
#                 }
#                 for col in df.columns
#             ],
#         ],
#         # ถ้าต้องการให้ table มีลักษณะ list view (ไม่มี vertical grid)
#         # style_as_list_view=True,

#         # สามารถตั้งค่า pagination, sorting, filtering ได้เพิ่มเติม
#         page_action='native',
#         page_size=10,
#         sort_action='native',
#         filter_action='native',
#     )


# def main_chart(title):
#     data = combined_df()

#     fig = px.line(
#         data, 
#         x="datetime",
#         y="value",
#         color="labels",
#         title=title,
#         color_discrete_sequence=["#3f89c3", "#e0a8a8"]
#     )

#     fig.update_layout(
#         xaxis_title=None,
#         yaxis_title=None,
#         xaxis=dict(
#             showgrid=False,
#             gridcolor="#0fa3ff"
#         ),
#         paper_bgcolor="#a8c2e0",
#         margin=dict(l=0, r=0, t=25, b=0),  # ลด margin ให้เต็มพื้นที่
#         legend=dict(
#             orientation="v",        
#             y=0.83,                    
#             yanchor="bottom",      
#             x=0.94,                  
#             xanchor="center",
#             bgcolor="rgba(255,255,255,0.5)"  # พื้นหลังโปร่งใสครึ่งหนึ่ง
#         ),
#         font=dict(
#             family="Arial, sans-serif",  # ฟอนต์
#             size=13,                     # ขนาดตัวอักษร
#             color="#004f7d"                # สีข้อความทั้งหมด (legend, title, tick)
#         )
#     )

#     return dcc.Graph(id='line-chart', figure=fig)