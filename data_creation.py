import pandas as pd
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 400000)
pd.set_option("display.width", 1000)

import seaborn as sns
sns.set_style("darkgrid")
sns.set_palette('muted')

# Import Data
course_data = pd.read_excel('manually_cleaned.xlsx')
course_data.columns = [x.lower().replace(" ", '_') for x in course_data.columns]

# Create year column values and category for actual / forecasted
course_data['actual_or_forecast'] = course_data['year'].apply(lambda x: x.split(" ")[-1])
course_data['year'] = course_data['year'].apply(lambda x: x.split(" ")[0])

# Dataset will only forecast on actual values
course_data = course_data[course_data['actual_or_forecast'] == 'Actual']
# Named variables for the possible permutations
product_options = course_data['product'].unique()
student_type_options = course_data['student_type'].unique()
funnel_step_options = ['web_traffic', 'enquiries', 'applications', 'enrolments']

course_data['date'] = course_data['year'].apply(lambda x: x[:4])
# Converts 'ds' to a datetime object
course_data['date'] = pd.to_datetime(course_data['date'])
course_data['date'] = course_data['date'] + pd.Timedelta((30 * 8) + 3, unit='days')