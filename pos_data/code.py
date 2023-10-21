import random
import pandas as pd
from datetime import datetime, timedelta

# Store Information
stores = ["Shinjuku", "Machida", "Yokohama"]
store_hours = {"Shinjuku": (7, 20), "Machida": (10, 21), "Yokohama": (7, 20)}
peak_hours = {"Shinjuku": ["Morning", "Noon", "Evening"],
              "Machida": ["Noon", "Evening"],
              "Yokohama": ["Morning", "Noon", "Evening"]}
weekday_stores = ["Shinjuku", "Machida"]
closed_days = {"Shinjuku": ["Saturday", "Sunday"], "Machida": ["Thursday"]}

# Customer Information
age_groups = ["Children", "Youth", "Middle-aged", "Elderly"]
gender = ["Male", "Female"]
payment_methods = ["Cash", "Credit", "Debit"]

# Item Information
items = {"Tuna Mayo": 200, "Kelp": 180, "Salmon Roe": 420, "Chashu": 300, "Takana": 250}
item_popularity = {
    "Tuna Mayo": ["All"],
    "Kelp": ["Elderly"],
    "Salmon Roe": ["Female"],
    "Chashu": ["Male (20s-40s)"],
    "Takana": ["Elderly"]
}

# Initialize variables
transactions = []
transaction_id = 1

# Simulate one day's sales data
date = datetime(2023, 7, 24)
for _ in range(24):
    store = random.choice(stores)
    start_hour, end_hour = store_hours[store]
    hour = random.randint(start_hour, end_hour)
    date_time = date.replace(hour=hour, minute=0, second=0)

    if store in weekday_stores:
        day = date_time.strftime("%A")
        if day in closed_days[store]:
            continue

    if store == "Machida" and date_time.hour == 11:
        # Double customers due to kindergarten pick-up
        num_customers = random.randint(60, 120)
    else:
        num_customers = random.randint(20, 40)

    for _ in range(num_customers):
        age_group = random.choice(age_groups)
        if age_group == "Children":
            age = random.randint(1, 18)
        elif age_group == "Youth":
            age = random.randint(18, 35)
        elif age_group == "Middle-aged":
            age = random.randint(36, 55)
        else:
            age = random.randint(56, 100)

        gender_choice = random.choice(gender)
        payment_method = random.choice(payment_methods)
        purchased_items = random.sample(list(items.keys()), random.randint(1, 3))

        for item in purchased_items:
            unit_price = items[item]
            quantity = random.uniform(1, 3)
            subtotal = unit_price * quantity
            discount = 0

            if date_time.hour == end_hour - 1:
                # Apply one-hour-before-closing discount
                discount = 100
                subtotal -= discount

            transactions.append([transaction_id, date_time, day, store, gender_choice, age_group, item, unit_price, quantity, subtotal, payment_method])
            transaction_id += 1

# Create a DataFrame
df = pd.DataFrame(transactions, columns=["Transaction ID", "Date/Time", "Day", "Store", "Gender", "Age Group", "Items Purchased", "Unit Price", "Quantity", "Subtotal", "Payment Method"])

# Export to Excel
# df.to_excel("sales_data.xlsx", index=False)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the sales data into a DataFrame (replace 'sales_data.xlsx' with the actual file path)
# df = pd.read_excel('sales_data.xlsx')

# Filter data for the Machida store
machida_data = df[df['Store'] == 'Machida']

# Convert the Date/Time column to hours
machida_data['Hour'] = machida_data['Date/Time'].dt.hour

# Group data by hour and calculate the total number of visitors
hourly_visitors = machida_data.groupby('Hour')['Transaction ID'].count().reset_index()

# Extract the hour and visitor count
X = hourly_visitors['Hour'].values.reshape(-1, 1)
y = hourly_visitors['Transaction ID'].values

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict visitors based on the model
y_pred = model.predict(X)

# Plot the original data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Actual Visitors', color='blue')
plt.plot(X, y_pred, label='Linear Regression', color='red')
plt.title('Trend in the Number of Visitors per Hour (Machida Store)')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Visitors')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

# Print the regression coefficients
print(f'Intercept (b0): {model.intercept_}')
print(f'Slope (b1): {model.coef_[0]}')

