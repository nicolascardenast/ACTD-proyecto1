import dash_bootstrap_components as dbc
import os
import re
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import random

#APP_HOME="C:/Users/arcem/OneDrive/Documentos/Universidad!/Analítica Computacional/Proyecto analítica/tablero_de_productividad/tablero_de_productividad"
#APP_HOME

directorio_padre = os.path.dirname(os.getcwd())

ruta = directorio_padre + '/ACTD-proyecto1/datos/datos_limpios.csv'

productivity_data = pd.read_csv(ruta)

productivity_data['team']=[ "team "+ str(_) for _ in productivity_data['team']] 
grouped_summary = productivity_data.groupby('team')['actual_productivity'].median().reset_index()
grouped_summary = grouped_summary.sort_values('actual_productivity')

def summarize_productivity(data, time_level):
    data['date'] = pd.to_datetime(data['date'])

    if time_level == 'daily':
        grouped_data = data.groupby(['date', 'team']).agg({'actual_productivity': 'mean'}).reset_index()

    elif time_level == 'weekly':
        data['year_week'] = data['date'].dt.strftime('%G-W%V')
        grouped_data = data.groupby(['year_week', 'team']).agg({'actual_productivity': 'mean'}).reset_index()
        grouped_data['date'] = grouped_data['year_week'].apply(lambda x: pd.to_datetime(x + '-1', format='%G-W%V-%u'))

    elif time_level == 'monthly':
        data['month'] = data['date'].dt.to_period('M')
        grouped_data = data.groupby(['month', 'team']).agg({'actual_productivity': 'mean'}).reset_index()
        grouped_data['date'] = grouped_data['month'].dt.to_timestamp()

    return grouped_data

productivity_over_time = summarize_productivity(productivity_data, 'monthly')
productivity_over_time

productivity_data['week']  =pd.to_datetime(productivity_data['date']).dt.isocalendar().week
grouped_weekly_data = productivity_data.groupby(['week', 'team']).agg({'actual_productivity': 'mean'}).reset_index()
grouped_weekly_data

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def summarize_data_for_group(index, standardize=False):
    with open('X_data.pkl', 'rb') as file:
        X = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    team_col = f'team_{index}'
    if team_col not in X.columns:
        print(f"Column '{team_col}' not found in the data.")
        return None

    filtered_data = X[X[team_col] == 1]
    summary = filtered_data.mean() 

    if standardize:
        summary = pd.DataFrame([summary])
        summary_scaled = scaler.transform(summary)
        return pd.DataFrame(summary_scaled, columns=summary.columns)

    return pd.DataFrame([summary])

summarize_data_for_group(8)

import pandas as pd
import pickle

def compute_predicted_productivity(index, custom_values=None):

    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)

    summarized_data = summarize_data_for_group(index, standardize=False)
    
    if summarized_data is None or summarized_data.empty:
        print(f"No data available for team {index}.")
        return None

    if custom_values:
        for col, value in custom_values.items():
            if col not in summarized_data.columns:
                raise ValueError(f"Column '{col}' is not a valid column.")
            summarized_data[col] = value

    return predicted_productivity[0]  


custom_replacements = {
    'incentive': 500,  
    'no_of_workers_redondeado': 50  
}

compute_predicted_productivity("12", custom_replacements)

summarize_data_for_group(8,standardize=True)

# Import necessary libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Create a Dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the application
app.layout = html.Div([
    # Navigation bar with tabs
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Exploratory', value='tab-1'),
        dcc.Tab(label='Explanatory', value='tab-2'),
        dcc.Tab(label='Predictive', value='tab-3'),
    ]),
    # Content panel
    html.Div(id='tabs-content')
])

# Additional imports for data processing and visualization
import pandas as pd
import plotly.express as px

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':

        # Line chart for the grouped time series
        line_chart = px.line(productivity_over_time, x='date', y='actual_productivity', color='team',
                             title='Average Actual Productivity Over Time by Team',
                             labels={'date': 'Date', 'actual_productivity': 'Average Actual Productivity'})
        
        line_chart.update_layout(legend_title_text='Team')

        # Boxplot of actual_productivity grouped and sorted by team
        boxplot = px.box(productivity_data, x='team', y='actual_productivity',
                         title='Actual Productivity by Team',
                         category_orders={"team": grouped_summary['team'].tolist()})
        
        # Calculate the 95th percentile for 'no_of_workers_redondeado' and 'incentive'
        workers_95th = productivity_data['no_of_workers_redondeado'].quantile(0.95)
        incentive_95th = productivity_data['incentive'].quantile(0.95)

        # Filter data to 95th percentile
        filtered_data_workers = productivity_data[productivity_data['no_of_workers_redondeado'] <= workers_95th]
        filtered_data_incentive = productivity_data[productivity_data['incentive'] <= incentive_95th]

        # Scatterplot of actual_productivity vs no_of_workers_redondeado with trendline
        scatter_plot_workers = px.scatter(
            filtered_data_workers, 
            x='no_of_workers_redondeado', 
            y='actual_productivity', 
            color='team',
            trendline='ols',  # Ordinary Least Squares regression line
            title='Actual Productivity vs Number of Workers (Up to 95th Percentile)',
            labels={'no_of_workers_redondeado': 'Number of Workers', 'actual_productivity': 'Actual Productivity'}
        )
        scatter_plot_workers.update_layout(legend_title_text='Team')

        # Scatterplot of actual_productivity vs incentive with trendline
        scatter_plot_incentive = px.scatter(
            filtered_data_incentive, 
            x='incentive', 
            y='actual_productivity', 
            color='team',
            trendline='ols',  # Ordinary Least Squares regression line
            title='Actual Productivity vs Incentive (Up to 95th Percentile)',
            labels={'incentive': 'Incentive', 'actual_productivity': 'Actual Productivity'}
        )
        scatter_plot_incentive.update_layout(legend_title_text='Team')

        # Structuring the content as a list of row elements with three rows
        return html.Div([
            html.Div([dcc.Graph(figure=line_chart)], className='row'),  # First row with the line chart
            html.Div([dcc.Graph(figure=boxplot)], className='row'),     # Second row with the boxplot
            html.Div([                                                  # Third row with two scatterplots
                dcc.Graph(figure=scatter_plot_workers, style={'display': 'inline-block', 'width': '50%'}),
                dcc.Graph(figure=scatter_plot_incentive, style={'display': 'inline-block', 'width': '50%'})
            ], className='row')
        ])


    elif tab == 'tab-2':

        tables = []
        filenames = ['univariate_regression_model.csv', 'multivariate_regression_model.csv']

        common_style = {'width': '85%', 'margin': 'auto'}
        
        for filename in filenames:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                table = dbc.Table.from_dataframe(
                    df,
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True
                )
                tables.append(html.Div([table], style=common_style))

        return tables

    elif tab == 'tab-3':

        # Load the data
        #productivity_data = pd.read_csv('datos_limpios.csv')

        # Finding min and max for the numerical input widgets
        min_incentive = productivity_data['incentive'].min()
        max_incentive = productivity_data['incentive'].max()

        #
        min_workers = productivity_data['no_of_workers_redondeado'].min()
        max_workers = productivity_data['no_of_workers_redondeado'].max()

        # Unique values for the 'team' parameter
        team_options = [{'label': f'{team}', 'value':float(re.search("[0-9]+",team)[0])} for team in productivity_data['team'].unique()]

        # Mock data for the tables
        mock_data_team_summary = pd.DataFrame({'Team': ['A', 'B', 'C'], 'Average Productivity': [0.8, 0.75, 0.78]})

        # Read the data from the Excel file
        model_eval_data = pd.read_excel('model_evaluations.xlsx')
        # Round numeric columns to 4 decimal places
        model_eval_data = model_eval_data.round(4)
        #
        model_eval_data = dbc.Table.from_dataframe(
            model_eval_data, striped=True, bordered=True, hover=True
        )
    
        table_team_summary = dbc.Table.from_dataframe(
            mock_data_team_summary, striped=True, bordered=True, hover=True
        )

        return html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Select Team', className='my-2'),
                    dcc.Dropdown(id='input-team', options=team_options, value=team_options[0]['value']),
                    html.Br(),
                    html.Label('Select Incentive', className='mb-2'),
                    dcc.Slider(id='input-incentive', min=min_incentive, max=max_incentive, value=min_incentive,
                            marks={i: str(i) for i in range(min_incentive, max_incentive + 1, int((max_incentive-min_incentive)/10))}),
                    html.Br(),
                    html.Label('Select Number of Workers', className='my-4'),
                    dcc.Slider(id='input-no-of-workers', min=min_workers, max=max_workers, value=min_workers,
                            marks={i: str(i) for i in range(min_workers, max_workers + 1, int((max_workers-min_workers)/10))})
                ], className='d-flex flex-column justify-content-center'), width=4),
                dbc.Col(dcc.Graph(id='gauge-chart'), width=8)
            ]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.H4('Model Evaluation Summary'),
                    model_eval_data
                ]), width=6),
                dbc.Col(html.Div(
                    id='team-data-summary',
                    children=[html.H4('Team Data Summary'),table_team_summary
                ]), width=6)
            ])
        ])

@app.callback(
        Output('gauge-chart', 'figure'),
    [
        Input('input-incentive', 'value'),
        Input('input-no-of-workers', 'value'),
        Input('input-team', 'value')
    ]
)
def update_gauge(incentive, no_of_workers, team):

    # Example usage
    custom_replacements = {
        'incentive': incentive,  # Example custom value for 'incentive'
        'no_of_workers_redondeado': no_of_workers  # Example custom value for 'no_of_workers_redondeado'
    }
    #
    predicted_productivity=compute_predicted_productivity(
        team,                                                   {
        'incentive': incentive, 
        'no_of_workers_redondeado': no_of_workers  
    })[0]

    
    gauge_chart = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_productivity,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Productivity Index"}
    ))
    return gauge_chart


@app.callback(
    Output('team-data-summary', 'children'),
    [Input('input-team', 'value')]
)
def update_team_data_summary(selected_team):
    team_data_summary_title = f"Team Data Summary - Team {selected_team}" if selected_team else "Team Data Summary"

    data_team_summary = summarize_data_for_group(selected_team).round(3)

    data_team_summary_transposed = data_team_summary.T.reset_index()
    data_team_summary_transposed.columns = ['Variable', 'Value']


    data_team_summary_transposed = data_team_summary_transposed[
    [not bool(re.search("team",_)) for _ in data_team_summary_transposed['Variable']]
    ]

    table_team_summary = dbc.Table.from_dataframe(
        data_team_summary_transposed, striped=True, bordered=True, hover=True
    )

    return html.Div([
        html.H4(team_data_summary_title),
        table_team_summary
    ])


port = 5001

if __name__ == '__main__':
    url = f"http://127.0.0.1:{port}"
    print(f"Dash app running on {url}")
    app.run_server(debug=False, port=port)