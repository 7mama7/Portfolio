import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Output, Input
import plotly.express as px
import plotly.grapg_objects as go
from dash import no_update

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv')
year = list(range(1980, 2014))
app = dash.Dash(__name__)
app.layout = html.Div(children=[html.H1('Historical Trends of Automobile Sales',
                                        style={'textAligh':'center',
                                               'color': '#503D36',
                                               'fontsize':50}),
                                html.Div(
                                    html.Div([html.H2('Type of Report:', style={'margin-right': '2em'}),
                                    dcc.Dropdown('', 
                                    value='', 
                                    id='report')]),
                                    html.Div([html.H2('Select Year:', style={'margin-right': '2em'}),
                                    dcc.Dropdown(year, 
                                    value=2005, 
                                    id='year')], style={'fontsize':20}
                                    )),
                                html.Div(
                                    html.Div([], id='plot-1'),
                                    html.Div([], id='plot-2'),
                                    style={'display':'flex'})
                                ])

@app.callback([Output(component_id='plot-1', component_property='children'),
               Output(component_id='plot-2', component_property='children')],
              [Input(component_id='year', component_property='value'),
               Input(component_id='report', component_property='value')])

def get_graph(input_year):
    yearly_data = df['Year']
    rec_data = df[df['Recession'] == 1]
    
    fig1 = px.line(yearly_data)
    fig2 = px.line()
    
    return [dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2)]
    