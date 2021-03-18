# Library Imports
import math
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 400000)
pd.set_option("display.width", 1000)
# Saved variables containing all options for each
from data_creation import funnel_step_options, student_type_options, product_options

def create_grid_cords(plots_needed):
    """When given the plots_needed, returns a list of subplot coordinates for plots_needed amount of plots"""

    if plots_needed % 2 != 0:
        plots_needed += 1

    # Rows
    n1 = ([1, 2] * (plots_needed // 2))
    n1.sort()

    # Columns
    n2 = range(1, (plots_needed // 2) + 1)
    n2 = list(n2) * 2

    cords_list = [[x, y] for (x, y) in zip(n1, n2)]

    return cords_list


# Set your confidence interval
confidence_interval = 0.95

# Named variables for the possible permutations
student_type_options = list(student_type_options)
student_type_options.append('Combined')
forecast_options = ['Facebook Prophet', 'ARIMA']
confidence_interval = 0.95
app = dash.Dash(__name__)

app.layout = html.Div([
            html.Div([html.H1(f"Forecast by course and funnel stage".title(), style={'textAlign': 'center'}),

            html.H4('Forecast Method'),
            dcc.RadioItems(id='forecast-choice',
                           options=[{'label':x, 'value':x} for x in forecast_options],
                           value=forecast_options[0],
                           style={'textAlign': 'center', 'margin-right': '3px', 'margin-right': '8px', 'color': 'black'}),

            html.H4('Course'),
            dcc.Checklist(id='course-choice',
                          options=[{'label':x, 'value':x} for x in sorted(product_options)],
                          style={'textAlign': 'center', 'margin': 'auto', 'width': '90%', 'color': 'black'},
                          value=[x for x in sorted(product_options) if x != 'Unknown'],
                          inputStyle={"margin-right": "3px", 'margin-left': '8px'}),

            html.H4('Funnel Step'),
            dcc.Dropdown(id='funnel-choice',
                         options=[{'label':x, 'value':x} for x in sorted(funnel_step_options)],
                         value='web_traffic',
                         style={"textAlign": 'center', 'margin':'auto', 'color': 'black', 'width':'50%'}),

            html.H4('Student Origin'),
            dcc.Dropdown(id='student-choice',
                                     options=[{'label':x.title(), 'value':x} for x in student_type_options],
                                     value='International',
                                     style={"textAlign": 'center', 'margin':'auto', 'color': 'black', 'width':'50%'}),
            html.Br(),

            dcc.Graph(id='product-graph', style={'height': '75vh'})])],

            style={'backgroundColor': 'white', 'color': 'black', 'fontFamily': 'Arial', 'height':'100vh', 'textAlign': 'center'})

@app.callback(
    # Outputs
    Output(component_id='product-graph', component_property='figure'),
    # Inputs
    Input(component_id='course-choice', component_property='value'),
    Input(component_id='funnel-choice', component_property='value'),
    Input(component_id='student-choice', component_property='value'),
    Input(component_id='forecast-choice', component_property='value'))

def interactive_graphs(course_choice, funnel_choice, student_choice, forecast_choice):
    asteriks = '*' * 20

    number_of_courses_selected = int(len(course_choice))

    grid_cords = create_grid_cords(number_of_courses_selected)

    # print(asteriks)
    # print(grid_cords)
    # print(asteriks)

    nrows = 2
    columns = math.ceil(number_of_courses_selected / 2)

    fig = make_subplots(rows=nrows, cols=columns, shared_yaxes='all', subplot_titles=[x for x in course_choice])

    if forecast_choice == forecast_options[0]:

        # Import Data
        FB_prophet_data = pd.read_csv('assets/fb_prophet_forecast.csv')
        FB_prophet_data = FB_prophet_data.drop(columns='index')
        print (FB_prophet_data)
        course_data = FB_prophet_data
    else:
        ARIMA_data = pd.read_csv('assets/ARIMA_forecast.csv')
        course_data = ARIMA_data
        course_data[['yhat_upper', 'yhat_lower']] = np.nan

    for cord, course in zip(grid_cords, course_choice):
        data = course_data[(course_data['product'] == course) &
                           (course_data['student_type'] == student_choice) &
                           (course_data['funnel_step'] == funnel_choice)]

        if student_choice == 'Combined':
            data = course_data.groupby(['ds', 'funnel_step', 'product'])['y', 'yhat', 'yhat_upper', 'yhat_lower'].sum().reset_index()
            data = data[(data['funnel_step'] == funnel_choice) & (data['product'] == course)]
            data = data.replace(0.0, np.nan)

        row = cord[0]
        column = cord[1]

        if row == 1 and column == 1:
            show_legend_bool = True
        else:
            show_legend_bool = False

        # Plotly figure creation. Adds a line trace for historical data, forecast data and the two confidence bounds
        student_type_color_dict = {'UK': {'historical': 'red', 'conf': 'orange', 'forecast': 'blue'},
                                   'International': {'historical': 'red', 'conf': 'orange', 'forecast': 'blue'},
                                   'Combined': {'historical': 'red', 'conf': 'orange', 'forecast': 'blue'}}
        fig.add_trace(go.Scatter(x=data['ds'],
                                 y=data['yhat'],
                                 name='Forecast',
                                 legendgroup='forecast',
                                 showlegend=show_legend_bool,
                                 line_color=student_type_color_dict[student_choice]['forecast']),
                                 row=row, col=column)
        fig.add_trace(go.Scatter(x=data['ds'],
                                 y=data['yhat_upper'],
                                 name=f'{confidence_interval * 100}% Confidence Interval',
                                 legendgroup='Conf. Int',
                                 showlegend=show_legend_bool,
                                 line=dict(color=student_type_color_dict[student_choice]['conf'],
                                           width=1, dash='dash')),
                                 row=row, col=column)
        fig.add_trace(go.Scatter(x=data['ds'],
                                 y=data['yhat_lower'],
                                 name=f'{round((1 - confidence_interval), 2) * 100}% Confidence Interval',
                                 legendgroup='Conf. Int',
                                 showlegend=show_legend_bool,
                                 line=dict(color=student_type_color_dict[student_choice]['conf'],
                                           width=1, dash='dash')),
                                 row=row, col=column)
        fig.add_trace(go.Scatter(x=data['ds'],
                                 y=data['y'],
                                 name='Historical',
                                 legendgroup='Historical',
                                 showlegend=show_legend_bool,
                                 line_color=student_type_color_dict[student_choice]['historical']),
                                 row=row, col=column)

    title = f'{funnel_choice} Forecast for {student_choice} Students with {forecast_choice}'.title().replace("_", " ")
    fig.update_layout(title=title, template='ggplot2')

    return fig
if __name__=='__main__':
    app.run_server()