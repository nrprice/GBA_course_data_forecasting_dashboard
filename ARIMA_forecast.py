import pandas as pd
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 400000)
pd.set_option("display.width", 1000)

from data_creation import course_data, student_type_options, product_options, funnel_step_options

from statsmodels.tsa.arima.model import ARIMA

from matplotlib import pyplot as plt

course_data = course_data
all_course_forecasts = pd.DataFrame()

for product in product_options:
    for student_type in student_type_options:
        for funnel_step in funnel_step_options:
            data = course_data[(course_data['product'] == product) &
            (course_data['student_type'] == student_type)][['date', funnel_step]]
            data = data.set_index('date')

            model = ARIMA(data, order=(1, 1, 1))

            model_fit = model.fit()
            output = model_fit.predict(start=2, end=5)

            output = pd.DataFrame(output)
            output['predicted_mean'] = output['predicted_mean'].astype(int)
            data = pd.DataFrame(data)
            data['type'] = 'Historical'
            output['type'] = 'Forecast'
            data = data.reset_index()
            data = data.rename(columns={funnel_step: 'y', 'date': 'ds'})


            output = output.reset_index()
            output = output[1:]
            # Gets last historical date and value
            last_historical_date = data['ds'].tail(1).values[0]
            last_historical_y_value = data['y'].tail(1).values[0]
            # Adds it to the start of the forecast dataframe
            output.loc[-1] = [last_historical_date, last_historical_y_value, 'Forecast']
            output.index = output.index + 1
            output.sort_index(inplace=True)
            output = output.rename(columns={'index': 'ds', 'predicted_mean': 'yhat'})

            # print (data)
            # print (output)
            forecast = pd.concat([data, output])
            forecast['student_type'] = student_type
            forecast['product'] = product
            forecast['funnel_step'] = funnel_step

            all_course_forecasts = all_course_forecasts.append(forecast)




all_course_forecasts = all_course_forecasts.reset_index()
all_course_forecasts = all_course_forecasts.drop(columns='index')
print (all_course_forecasts.head(5))
all_course_forecasts.to_csv('ARIMA_forecast.csv')