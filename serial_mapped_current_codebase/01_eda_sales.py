# Serial mapped from current codebase: Data Exploration and Visualization.py
# Matched against pmafdhuv assignment order.

# =====================================================
# Exploratory Data Analysis (EDA)
# =====================================================

# Import libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# Load Dataset
# =====================================================

csv_path = "sales_data.csv"
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
else:
    print(f"WARNING: '{csv_path}' not found. Using sample sales data.")
    data = pd.DataFrame({
        'SALES': [250.0, 300.0, 275.0, 320.0, 290.0, 310.0, 330.0, 305.0, 295.0, 340.0],
        'QUANTITYORDERED': [10, 12, 9, 14, 11, 13, 15, 12, 10, 16],
        'PRICEEACH': [25.0, 25.0, 30.5, 22.9, 26.4, 23.8, 22.0, 25.4, 29.5, 21.3],
        'STATUS': ['Shipped', 'Shipped', 'Cancelled', 'Shipped', 'On Hold', 'Shipped', 'Shipped', 'Cancelled', 'Shipped', 'On Hold'],
        'PRODUCTLINE': ['Electronics', 'Furniture', 'Electronics', 'Clothing', 'Clothing', 'Furniture', 'Electronics', 'Furniture', 'Clothing', 'Electronics'],
        'MONTH_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

# =====================================================
# Display Dataset
# =====================================================

print("First 5 Rows:")
print(data.head())

# =====================================================
# Dataset Information
# =====================================================

print("\nDataset Information:")
print(data.info())

# =====================================================
# Check Missing Values
# =====================================================

print("\nMissing Values:")
print(data.isnull().sum())

# =====================================================
# Statistical Summary
# =====================================================

print("\nStatistical Summary:")
print(data.describe())

# =====================================================
# Remove Duplicates
# =====================================================

data.drop_duplicates(inplace=True)

print("\nDuplicates Removed!")

# =====================================================
# Correlation Matrix
# =====================================================

plt.figure(figsize=(10,6))

sns.heatmap(
    data.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Matrix")

plt.show()

# =====================================================
# Distribution Plot
# =====================================================

plt.figure(figsize=(8,5))

sns.histplot(data['SALES'], bins=20, kde=True)

plt.title("Sales Distribution")

plt.xlabel("Sales")

plt.show()

# =====================================================
# Boxplot
# =====================================================

plt.figure(figsize=(8,5))

sns.boxplot(x=data['SALES'])

plt.title("Boxplot of Sales")

plt.show()

# =====================================================
# Scatter Plot
# =====================================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x=data['QUANTITYORDERED'],
    y=data['SALES']
)

plt.title("Quantity Ordered vs Sales")

plt.show()

# =====================================================
# Pairplot
# =====================================================

sns.pairplot(
    data[['SALES',
          'QUANTITYORDERED',
          'PRICEEACH']]
)

plt.show()

# =====================================================
# Countplot
# =====================================================

plt.figure(figsize=(10,5))

sns.countplot(x=data['STATUS'])

plt.title("Order Status Count")

plt.xticks(rotation=45)

plt.show()

# =====================================================
# Grouped Analysis
# =====================================================

grouped = data.groupby('STATUS')['SALES'].mean()

print("\nAverage Sales by Status:")
print(grouped)

# =====================================================
# Top 5 Products
# =====================================================

top_products = data.groupby(
    'PRODUCTLINE'
)['SALES'].sum().sort_values(ascending=False)

print("\nTop Product Categories:")
print(top_products.head())

# =====================================================
# Line Plot
# =====================================================

plt.figure(figsize=(10,5))

monthly_sales = data.groupby(
    'MONTH_ID'
)['SALES'].sum()

monthly_sales.plot(marker='o')

plt.title("Monthly Sales Trend")

plt.xlabel("Month")

plt.ylabel("Sales")

plt.show()

# =====================================================
# EDA Completed
# =====================================================

print("\nEDA Completed Successfully!")