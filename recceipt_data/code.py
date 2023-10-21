

import pandas as pd
import random
from datetime import datetime, timedelta

# Define constants
start_date = datetime(2022, 9, 1, 7, 0)
end_date = datetime(2023, 8, 31, 22, 0)
business_types = ["Supermarket", "Convenience Store", "Drugstore", "Discount Store", "Home Center"]
payment_methods = ["Cash", "Credit", "Debit"]
genders = ["Male", "Female", "Other"]
age_categories = ["Children", "Youth", "Middle-aged", "Elderly"]
product_names = [f"Product_{i}" for i in range(1, 1001)]

# Initialize data list
data = []

# Helper function to categorize age
def categorize_age(age):
    if age <= 20:
        return "Children"
    elif age <= 40:
        return "Youth"
    elif age <= 60:
        return "Middle-aged"
    else:
        return "Elderly"

# Helper function to generate random datetime within business hours
def random_business_hour_time():
    return start_date + timedelta(minutes=random.randint(0, 15*60))

# Initialize variables
transaction_id = 1
user_id = 1
receipt_id = 1
jan_code = 1

# Generate data
while start_date <= end_date:
    user_age = random.randint(1, 80)
    user_age_category = categorize_age(user_age)
    user_gender = random.choice(genders)
    business_type = random.choice(business_types)
    purchase_time = random_business_hour_time()
    product = random.choice(product_names)
    unit_price = round(random.uniform(1, 100), 2)
    quantity = random.randint(1, 3)
    subtotal = unit_price * quantity
    payment_method = random.choice(payment_methods)

    data.append([transaction_id
                 , receipt_id
                 , user_id
                 , purchase_time
                 , purchase_time.strftime("%A")
                 , business_type
                 , user_gender
                 , user_age_category
                 , product
                 , unit_price
                 , quantity
                 , subtotal
                 , payment_method
                 , jan_code]
                )

    transaction_id += 1
    receipt_id += 1
    jan_code += 1

    # Generate a new transaction for a new user
    if random.random() < 0.2:
        user_id += 1
        transaction_id = 1

    start_date += timedelta(minutes=random.randint(10, 120))

# Create a DataFrame
df = pd.DataFrame(data, columns=["Transaction ID", "Receipt ID", "User ID", "Date/Time", "Day of the Week", "Business Type", "Gender", "Age", "Product Name", "Unit Price", "Quantity Purchased", "Subtotal", "Payment Method", "Jan Code"])

# Save data to a .csv file
df.to_csv("sales_data.csv", index=False)
