import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import openpyxl
import datetime

# Hampel identifier to handle outliers
def hampel_filter(data, window_size=5, n_sigma=3):
    median_window = data.rolling(window=window_size, center=True).median()
    deviation = np.median(np.abs(data - median_window))
    outlier_idx = (np.abs(data - median_window) > n_sigma * deviation)
    data[outlier_idx] = median_window[outlier_idx]
    return data

# Function to automatically find the best Holt-Winters parameters
def find_best_holtwinters_params(data):
    seasonal_periods = 12  # Monthly data
    min_aic = np.inf
    best_params = None

    # Define grid search parameters
    trend_options = ["add", "additive", "multiplicative"]
    seasonal_options = ["add", "additive", "multiplicative", None]

    for trend in trend_options:
        for seasonal in seasonal_options:
            model = ExponentialSmoothing(data, seasonal=seasonal, seasonal_periods=seasonal_periods, trend=trend)
            try:
                fit = model.fit()
                aic = fit.aic
                if aic < min_aic:
                    min_aic = aic
                    best_params = {"seasonal": seasonal, "trend": trend}
            except Exception:
                pass

    return best_params

# Function to fit a Holt-Winters model with specified parameters
def fit_holtwinters(data, seasonal_params):
    model = ExponentialSmoothing(data, seasonal=seasonal_params["seasonal"], seasonal_periods=12, trend=seasonal_params["trend"])
    fit = model.fit()
    return fit

# Excel Extraction
# Read data from the Excel file
sales_df = pd.read_excel('sales_data.xlsx', na_values='')

# Convert the "Date" column to datetime if not already in datetime format
sales_df['Date'] = pd.to_datetime(sales_df['Date'])

# Set the "Date" column as the DataFrame's index
sales_df.set_index('Date', inplace=True)

# Unique countries in the dataset
unique_countries = sales_df['Country'].unique()

# List all available countries
print("\nAvailable Countries:")
print(", ".join(unique_countries))
print("\nChoose wisely! The fate of the sales forecast rests in your hands.")

# Prompt the user to select the country for forecasting (case-insensitive)
selected_country = input("\nEnter the name of the country you wish to forecast: ").strip().title()

# Validate the user input until a valid country is provided
while selected_country not in unique_countries:
    print("\nHmm... that's not one of the available countries. Try again!")
    selected_country = input("\nEnter the name of the country you wish to forecast: ").strip().title()

print("\nExcellent choice! Let's proceed with the forecast for", selected_country, "...\n")

# Filter data for the selected country
country_data = sales_df[sales_df['Country'] == selected_country]

# Initialize forecast variables
this_month_forecast_holtwinters = np.nan
this_month_forecast_sarima = np.nan

# Handle anomalous data points
if not country_data.empty:
    sales_series = hampel_filter(country_data['Sum of ActualValueEuro'].copy(), window_size=5, n_sigma=3)

# Find the best Holt-Winters parameters
holtwinters_params = find_best_holtwinters_params(sales_series)

# Fit Holt-Winters model with the best parameters
holtwinters_fit = fit_holtwinters(sales_series, holtwinters_params)

# SARIMA Model (Automatic parameter selection)
sarima_model = auto_arima(sales_series, seasonal=True, m=12, stepwise=True, trace=True)
sarima_order = sarima_model.order
sarima_seasonal_order = sarima_model.seasonal_order

try:
    sarima_fit = SARIMAX(sales_series, order=sarima_order, seasonal_order=sarima_seasonal_order).fit(disp=False)
    this_month_forecast_sarima = sarima_fit.forecast(steps=1).iloc[0]
except ValueError as e:
    sarima_fit = None
    this_month_forecast_sarima = np.nan
    print(f"Warning: SARIMA fit failed for {selected_country}.")
    print(f"Error: {e}")

# Select the model with the lowest AIC (Akaike Information Criterion)
models = []

if sarima_fit is not None:
    models.append(('SARIMA', sarima_fit.aic))
if holtwinters_fit is not None:
    models.append(('Holt-Winters', holtwinters_fit.aic))

if models:
    lowest_aic_model = min(models, key=lambda x: x[1])[0]
else:
    lowest_aic_model = 'No model available'

# Print the selected country and the forecasted value for this month
print(f"Selected Country: {selected_country}")
print(f"Forecast for this month (using {lowest_aic_model}):")
print(f"SARIMA Forecast: {this_month_forecast_sarima:.2f}")
print(f"Holt-Winters Forecast: {this_month_forecast_holtwinters:.2f}")

# Concatenate the forecasts to the original sales data
forecast_df = pd.DataFrame({
    'SARIMA Forecast': sarima_fit.fittedvalues if sarima_fit else np.nan,
    'Holt-Winters Forecast': holtwinters_fit.fittedvalues if holtwinters_fit else np.nan
}, index=sales_series.index)

# Adding the forecast for this month
this_month_date = pd.to_datetime('now').replace(day=1)  # Set the day to 1st of the current month
forecast_df = forecast_df.append(pd.DataFrame({
    'SARIMA Forecast': this_month_forecast_sarima if lowest_aic_model == 'SARIMA' else np.nan,
    'Holt-Winters Forecast': this_month_forecast_holtwinters if lowest_aic_model == 'Holt-Winters' else np.nan
}, index=[this_month_date]))

# Plotting the sales data and forecasts with seasonality
plt.figure(figsize=(12, 6))
plt.plot(forecast_df.index, forecast_df['SARIMA Forecast'], label='SARIMA Forecast', linestyle='dashed', color='orange')
plt.plot(forecast_df.index, forecast_df['Holt-Winters Forecast'], label='Holt-Winters Forecast', linestyle='dashed', color='green')
plt.scatter(sales_series.index, sales_series, label='Actual Sales', marker='o', color='red')
plt.scatter(this_month_date, this_month_forecast_sarima if lowest_aic_model == 'SARIMA' else np.nan, label='SARIMA Forecast', marker='x', color='blue')
plt.scatter(this_month_date, this_month_forecast_holtwinters if lowest_aic_model == 'Holt-Winters' else np.nan, label='Holt-Winters Forecast', marker='x', color='blue')
plt.title('Monthly Sales Data and Forecasts (with Seasonality)')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Convert the index to datetime (if not already)
sales_df.index = pd.to_datetime(sales_df.index)

# Plotting the sales data for each country with dynamic start date
plt.figure(figsize=(12, 6))
for country_name in unique_countries:
    # Filter data for the current country
    country_data = sales_df[sales_df['Country'] == country_name]

    # Find the index of the first non-missing value
    start_index = country_data['Sum of ActualValueEuro'].first_valid_index()

    # Plot sales data for the current country starting from the first non-missing value
    plt.plot(country_data.loc[start_index:, 'Sum of ActualValueEuro'], marker='o', label=country_name)

# Set plot labels and title
plt.xlabel('Date')
plt.ylabel('Sales Value (Euro)')
plt.title('Monthly Sales Data for Each Country')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# Create a DataFrame for the monthly sales data and forecasts
monthly_data = pd.concat([sales_series, forecast_df], axis=1)
monthly_data.columns = ['Actual Sales', 'SARIMA Forecast', 'Holt-Winters Forecast']

# Check for missing months and fill the NaN values with 0
monthly_data.fillna(0, inplace=True)

# Calculate percentage error for SARIMA and Holt-Winters forecasts
monthly_data['SARIMA Error (%)'] = ((monthly_data['SARIMA Forecast'] - monthly_data['Actual Sales']) / monthly_data['Actual Sales']) * 100
monthly_data['Holt-Winters Error (%)'] = ((monthly_data['Holt-Winters Forecast'] - monthly_data['Actual Sales']) / monthly_data['Actual Sales']) * 100

# Create an Excel writer object
excel_writer = pd.ExcelWriter(f'forecasted_sales_{selected_country}.xlsx', engine='xlsxwriter')

# Write the data to the Excel sheet
monthly_data.to_excel(excel_writer, sheet_name='Monthly Sales and Forecasts')

# Get the xlsxwriter workbook and worksheet objects
workbook = excel_writer.book
worksheet = excel_writer.sheets['Monthly Sales and Forecasts']

# Get the dimensions of the dataframe
num_rows, num_cols = monthly_data.shape

# Create a chart object
chart = workbook.add_chart({'type': 'line'})

# Configure the series of the chart from the data
for i in range(1, num_cols):
    chart.add_series({
        'name': [worksheet.name, 0, i],
        'categories': [worksheet.name, 1, 0, num_rows, 0],
        'values': [worksheet.name, 1, i, num_rows, i],
    })

# Insert the chart into the worksheet
worksheet.insert_chart('E2', chart)

# Close the Excel writer
excel_writer.save()

print(f"Monthly SARIMA and Holt-Winters forecasts and percentage errors for {selected_country} have been saved to 'forecasted_sales_{selected_country}.xlsx'.")
