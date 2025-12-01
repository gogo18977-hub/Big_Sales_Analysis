import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("C:/Users/gogo1/Downloads/big_sales.csv")

# Identify categorical and numerical columns
df_cat = df.select_dtypes(include=['object', 'category']).columns
df_num = df.select_dtypes(include=np.number).columns

# Check missing values
print("Missing values per column:\n", df.isna().sum())

# -----------------------------
# Customer Data Analysis
# -----------------------------
customer_df = df[['CustomerID', 'Gender', 'Age', 'Country', 'PaymentMethod',
                  'Product', 'DeliveryTimeDays', 'Quantity']]

# Quantity by Country
country_quantity = customer_df.groupby('Country')['Quantity'].sum()
plt.figure(figsize=(10,6))
sns.barplot(x=country_quantity.index, y=country_quantity.values)
plt.title("Total Quantity Sold by Country")
plt.ylabel("Quantity")
plt.show()

# Quantity by Gender
plt.figure(figsize=(6,5))
sns.boxplot(data=customer_df, x='Gender', y='Quantity')
plt.title("Quantity Distribution by Gender")
plt.show()

# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(customer_df['Age'], bins=20)
plt.title("Age Distribution of Customers")
plt.show()

# -----------------------------
# Sales Data Analysis
# -----------------------------
sale_df = df[['OrderID', 'Product', 'Category', 'Warehouse', 'Supplier', 
              'Views', 'AdSource', 'Clicks', 'ReturnStatus', 'Rating', 'Quantity']]

# Fill missing AdSource
sale_df['AdSource'] = sale_df['AdSource'].fillna('unknown')

# Top Products by Quantity
top_products = sale_df.groupby('Product')['Quantity'].sum().sort_values(ascending=False)
print("Top products by quantity:\n", top_products.head())

# Top Categories by Quantity
top_categories = sale_df.groupby('Category')['Quantity'].sum().sort_values(ascending=False)
print("Top categories by quantity:\n", top_categories)

# Highest rated products
most_rating = sale_df[sale_df['Rating'] == sale_df['Rating'].max()]
print("Products with highest rating:\n", most_rating[['Product','Rating']])

# -----------------------------
# Return Analysis
# -----------------------------
returned = sale_df[sale_df['ReturnStatus'] == 'Returned']

# Returned Quantity per Product
returned_per_product = returned.groupby('Product')['Quantity'].sum().sort_values(ascending=False)
print("Returned quantity per product:\n", returned_per_product.head())

# Returned Quantity per Category
returned_per_category = returned.groupby('Category')['Quantity'].sum().sort_values(ascending=False)
print("Returned quantity per category:\n", returned_per_category)

# Return Rate per Supplier
total_supplier = sale_df.groupby('Supplier')['Quantity'].sum()
returned_supplier = returned.groupby('Supplier')['Quantity'].sum()
return_rate_supplier = (returned_supplier / total_supplier * 100).fillna(0)
print("Return rate per Supplier (%):\n", return_rate_supplier)

# Visualize Returns by Supplier
plt.figure(figsize=(10,6))
sns.countplot(data=sale_df, x='Supplier', hue='ReturnStatus')
plt.title("Return Status by Supplier")
plt.show()

# Visualize Returns by Warehouse
plt.figure(figsize=(10,6))
sns.countplot(data=sale_df, x='Warehouse', hue='ReturnStatus')
plt.title("Return Status by Warehouse")
plt.show()

# -----------------------------
# Clicks vs Quantity Analysis
# -----------------------------
plt.figure(figsize=(8,6))
sns.regplot(data=sale_df, x='Clicks', y='Quantity')  # sample for performance
plt.title("Clicks vs Quantity")
plt.show()

# Correlation Heatmap
corr = sale_df[['Clicks', 'Quantity', 'Views', 'Rating']].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation between numerical columns")
plt.show()

# -----------------------------
# AdSource Analysis
# -----------------------------
adsource_views = sale_df.groupby('AdSource')['Views'].sum()
print("Total Views per AdSource:\n", adsource_views)

