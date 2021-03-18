import pandas as pd
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 400000)
pd.set_option("display.width", 1000)

import seaborn as sns
sns.set_style("darkgrid")
sns.set_palette('muted')

import plotly.graph_objs as go

from fbprophet import Prophet

from data_creation import course_data


# Named variables for the possible permutations
list_of_products = course_data['product'].unique()
list_of_student_types = course_data['student_type'].unique()
list_of_funnel_steps = ['web_traffic', 'enquiries', 'applications', 'enrolments']

# Change to True to only run on specified elements
save_time = False
if save_time:
    list_of_products = ['BA Business ']
    list_of_student_types = ['International']

# Set your confidence interval
confidence_interval = 0.95

# Set your target
target = 'enrolments'

historical_and_forecast = pd.DataFrame()

# Loops over all possible variations
for product in list_of_products:
    for step in list_of_funnel_steps:
        for student_type in list_of_student_types:
            # This dataset contains many unkowns which we will ignore
            if product == 'Unknown':
                pass
            else:

                # Groups the neccessary data by year and sums the target variable
                product_data = course_data[(course_data['product'] == product) &
                                           (course_data['student_type'] == student_type)].groupby('year')[step].sum().reset_index()

                # FbProphet needs columns to be named 'ds' for dates and 'y' for the value we're trying to predict
                product_data = product_data.rename(columns={'year':'ds', step:'y'})
                product_data['ds'] = product_data['ds'].apply(lambda x: x[:4])
                # Converts 'ds' to a datetime object
                product_data['ds'] = pd.to_datetime(product_data['ds'])

                # Fbprophet will create a new dataframe containing predicted values for historical points
                # In order to remove those from the plot and have the forecast start at the end of the historical points, we need to know how many rows to ignore.
                number_of_historical_points = len(product_data)

                # Create and fit FbProphet model to our data
                model = Prophet(interval_width=confidence_interval)
                model.fit(product_data, iter=10)
                # FbProphet needs a dataframe to put it's predictions in
                future = model.make_future_dataframe(periods=3, freq='Y')
                # Fill our future dataframe with our predictions
                forecast = model.predict(future)
                # Limit our predictions to be only those after the end of the historical data
                print (forecast)
                forecast = forecast[number_of_historical_points:][['ds', 'yhat_lower', 'yhat_upper', 'yhat']]
                # Fbprophet uses the last day of the year by default, our original data is done on the 1st of the year.
                # We add one day so the dates line up
                forecast['ds'] = forecast['ds'] + pd.Timedelta(1, unit='days')

                # We need to add the last historical data point to our forecast data so we can plot the forecast data starting from the end of the historical

                # Gets last historical date and value
                last_historical_date = product_data['ds'].tail(1).values[0]
                last_historical_y_value = product_data['y'].tail(1).values[0]
                # Adds it to the start of the forecast dataframe
                forecast.loc[-1] = [last_historical_date, last_historical_y_value, last_historical_y_value, last_historical_y_value]
                forecast.index = forecast.index + 1
                forecast.sort_index(inplace=True)

                # Housekeeping to remove the old index column
                forecast = forecast.reset_index().drop(columns=['index'])

                combined = pd.concat([product_data, forecast])
                combined['product'] = product
                combined['student_type'] = student_type
                combined['funnel_step'] = step

                historical_and_forecast = pd.concat([historical_and_forecast, combined])
                historical_and_forecast = historical_and_forecast.reset_index()
                historical_and_forecast = historical_and_forecast.set_index('index')



historical_and_forecast.to_csv('fb_prophet_forecast.csv')