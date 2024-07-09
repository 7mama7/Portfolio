import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Sample data (replace with your actual dataset)
data = pd.read_csv("C:/Users/Abdulrhman Alsir/Desktop/Data Science/Data/CardioGoodFitness.csv")

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1('CardioGood Fitness Dashboard', style={'textAlign': 'center'}),
    
    html.Div([
        dcc.Graph(id='age-dist'),
        dcc.Graph(id='income-dist')
    ], style={'display': 'flex'}),
    
    dcc.Dropdown(
        id='product-dropdown',
        options=[
            {'label': 'TM195', 'value': 'TM195'},
            {'label': 'TM498', 'value': 'TM498'},
            {'label': 'TM798', 'value': 'TM798'}
        ],
        value='TM195',
        style={'width': '50%', 'margin': 'auto'}
    ),
    
    dcc.Graph(id='fitness-boxplot')
])

# Define callbacks to update graphs based on user input
@app.callback(
    Output('age-dist', 'figure'),
    Output('income-dist', 'figure'),
    Output('fitness-boxplot', 'figure'),
    Input('product-dropdown', 'value')
)
def update_graphs(selected_product):
    filtered_data = data[data['Product'] == selected_product]
    
    age_dist = px.histogram(filtered_data, x='Age', title=f'Distribution of Age for {selected_product}')
    income_dist = px.histogram(filtered_data, x='Income', title=f'Distribution of Income for {selected_product}')
    fitness_boxplot = px.box(filtered_data, x='Product', y='Fitness', title=f'Fitness Levels for {selected_product}')
    
    return age_dist, income_dist, fitness_boxplot

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
