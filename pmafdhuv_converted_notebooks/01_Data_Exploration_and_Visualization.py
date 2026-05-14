# Converted from PMAF_1/Assignment 1.ipynb
# Original notebook code cells exported as a normal Python script.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "sales_data.csv"
df = pd.read_csv(file_path)

# First 5 rows
df.head()

# Data info
df.info()

# Descriptive stats
df.describe()

# Convert date column
df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])

# Check missing values
print(df.isnull().sum())

df['Profit_per_Unit'] = df['Unit_Price'] - df['Unit_Cost']
df['Total_Profit'] = df['Profit_per_Unit'] * df['Quantity_Sold']
df['Revenue_per_Unit'] = df['Sales_Amount'] / df['Quantity_Sold']

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure()
    df[col].hist()
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

corr = df[numeric_cols].corr()

corr

# Heatmap
plt.figure()
plt.imshow(corr)
plt.title("Correlation Heatmap")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

pairs = [
    ('Sales_Amount', 'Quantity_Sold'),
    ('Sales_Amount', 'Unit_Price'),
    ('Sales_Amount', 'Unit_Cost'),
    ('Quantity_Sold', 'Unit_Price'),
]

for x, y in pairs:
    plt.figure()
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y}")
    plt.show()

df['Month'] = df['Sale_Date'].dt.to_period('M')

monthly_sales = df.groupby('Month')['Sales_Amount'].sum()

plt.figure()
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.show()

categorical_cols = [
    'Region', 'Product_Category', 'Sales_Channel',
    'Customer_Type', 'Payment_Method'
]

for col in categorical_cols:
    grouped = df.groupby(col)['Sales_Amount'].sum().sort_values()

    print(f"\n===== Sales by {col} =====")
    print(grouped)

    plt.figure()
    grouped.plot(kind='bar')
    plt.title(f"Sales by {col}")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.show()

# Top products
top_products = df.groupby('Product_ID')['Sales_Amount'].sum().sort_values(ascending=False).head(10)
print("\n===== TOP 10 PRODUCTS =====")
print(top_products)

# Top sales reps
top_reps = df.groupby('Sales_Rep')['Sales_Amount'].sum().sort_values(ascending=False).head(10)
print("\n===== TOP SALES REPS =====")
print(top_reps)

plt.figure()
plt.scatter(df['Discount'], df['Sales_Amount'])
plt.xlabel("Discount")
plt.ylabel("Sales Amount")
plt.title("Discount vs Sales")
plt.show()

plt.figure()
plt.scatter(df['Total_Profit'], df['Sales_Amount'])
plt.xlabel("Total Profit")
plt.ylabel("Sales Amount")
plt.title("Profit vs Sales")
plt.show()

print("1. Dataset is clean with no missing values.")
print("2. Weak correlation between sales and numerical features.")
print("3. Strong relationship between Unit Price and Unit Cost.")
print("4. Categorical features likely drive business performance.")
print("5. Profit and discount strategies need deeper analysis.")
