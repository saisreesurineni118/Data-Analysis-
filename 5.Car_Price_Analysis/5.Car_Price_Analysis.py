#!/usr/bin/env python
# coding: utf-8

# ##  Car Price Analysis
# The Car Price Dataset is designed for analyzing factors that influence car prices in both new and used car markets. Understanding these factors helps buyers, sellers, and dealers make informed decisions when pricing vehicles. The dataset is useful for predictive modeling, regression analysis, and market trend evaluation in the automobile industry.
# ## Description:
# --
# The dataset includes various features affecting a car's price, such as:
# 
# Brand – Premium brands like Audi, BMW, and Mercedes tend to have higher prices.
# 
# Year of Manufacture – Newer cars generally have higher prices, though depreciation affects value over time.
# 
# Fuel Type – Cars with different fuel types (petrol, diesel, electric) have varying price points due to efficiency and maintenance costs.
# 
# Transmission – Automatic cars are often priced higher than manual ones.
# 
# Mileage – Lower mileage typically increases a car's value, as it suggests less wear and tear.
# 
# Engine Size & Power – Higher engine displacement and horsepower often lead to increased prices.
# 
# Car Condition – Used cars with better maintenance records and lower accident history are valued higher.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv("car_price_dataset.csv")
df


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df['Brand'].value_counts()


# In[7]:


a=min(df['Price'])
a


# In[8]:


df['Year'].value_counts()


# In[9]:


a=df[df['Doors']==5]
a


# In[10]:


s=df[df['Year']==2009]
s


# In[11]:


df['Owner_Count'].value_counts()


# In[12]:


df[df['Year']==2002]['Owner_Count'].value_counts()


# In[13]:


plt.figure(figsize=(5,4))
sns.histplot(df['Price'], kde=True, bins=30,color='yellow')
plt.title('Distribution of Car Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# The histogram shows a right-skewed distribution with most car prices around 8000-10000, fewer expensive cars, and a KDE line 
# confirming the trend.

# In[14]:


plt.figure(figsize=(5,4))
sns.boxplot(x='Year', y='Price', data=df)
plt.title('Car Price vs Year of Manufacture')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# The box plot shows that car prices generally increase with newer manufacturing years, with higher median prices and wider price
# ranges in recent years, while older cars have lower prices with more variability.

# In[15]:


plt.figure(figsize=(5,4))
brand_price = df.groupby('Brand')['Price'].mean().sort_values(ascending=False)
sns.barplot(x=brand_price.index, y=brand_price.values)
plt.title('Average Car Price by Brand')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# The bar plot shows the average car price by brand, with Chevrolet, Mercedes, and Audi having the highest average prices, while
# all brands have relatively similar price ranges.

# In[16]:


plt.figure(figsize=(5,4))
fuel_price = df.groupby('Fuel_Type')['Price'].mean().sort_values(ascending=False)
sns.barplot(x=fuel_price.index, y=fuel_price.values)
plt.title('Average Car Price by Fuel Type')
plt.ylabel('Average Price')
plt.xlabel('Fuel Type')
plt.show()


# The bar plot shows that Electric cars have the highest average price, followed by Hybrid, while Diesel and Petrol cars have the 
# lowest average prices, indicating that alternative fuel vehicles tend to be more expensive.

# In[17]:


plt.figure(figsize=(6, 5))
sns.countplot(x='Transmission', data=df)
plt.title('Count of Cars by Transmission Type')
plt.xlabel('Transmission Type')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# The bar plot shows the distribution of cars by transmission type, with Manual, Automatic, and Semi-Automatic transmissions 
# having nearly equal counts.

# In[18]:


plt.figure(figsize=(5,4))
sns.boxplot(x='Fuel_Type', y='Price', data=df)
plt.title('Car Price Distribution by Fuel Type')
plt.show()


# This box plot displays the distribution of car prices by fuel type (Diesel, Hybrid, Electric, and Petrol). Key insights from 
# the plot:
# Electric cars generally have higher median prices.
# Diesel and Petrol cars show wider price variability.
# Outliers are present, especially for Hybrid and Petrol cars.

# In[19]:


plt.figure(figsize=(5,4))
sns.scatterplot(x='Mileage', y='Price', hue='Fuel_Type', data=df)
plt.title('Mileage vs Price by Fuel Type')
plt.show()


# This scatter plot visualizes the relationship between Mileage and Price, categorized by Fuel Type. Key observations:
# Negative correlation: Higher mileage generally corresponds to lower price.
# Fuel Type distribution: All fuel types follow a similar trend, but Electric and Hybrid cars tend to have higher prices at lower
# mileage.
# Diesel and Petrol cars are more evenly spread across the mileage range.

# In[20]:


plt.figure(figsize=(5,4))
avg_price_year = df.groupby('Year')['Price'].mean()
sns.lineplot(x=avg_price_year.index, y=avg_price_year.values, marker='o')
plt.title('Average Car Price Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.show()


# This line plot shows the trend of average car prices over the years. Key insights:
# Steady increase: The average car price has been rising consistently from 2000 to 2023.
# Sharp growth post-2010: Prices have increased at a faster rate after 2010, possibly due to inflation, technological advancement,
#  or market demand.

# In[21]:


top_cars = df.sort_values(by='Price', ascending=False).head(10)

plt.figure(figsize=(5,4))
sns.barplot(x=top_cars['Model'], y=top_cars['Price'])
plt.title('Top 10 Most Expensive Cars')
plt.xticks(rotation=45)
plt.xlabel('Car Model')
plt.ylabel('Price')
plt.show()


# In[22]:


plt.figure(figsize=(5,4))
brand_count = df['Brand'].value_counts().head(15)
sns.barplot(x=brand_count.index, y=brand_count.values)
plt.title('Top 15 Brands by Number of Listings')
plt.ylabel('Number of Cars')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[23]:


fuel_counts = df['Fuel_Type'].value_counts()

plt.figure(figsize=(5,4))
plt.pie(fuel_counts, labels=fuel_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Fuel Type Distribution')
plt.axis('equal')  # Equal aspect ratio ensures pie is a circle.
plt.show()


# In[24]:


plt.figure(figsize=(5,4))
sns.boxplot(x='Transmission', y='Mileage', data=df)
plt.title('Mileage Distribution by Transmission Type')
plt.show()


# In[25]:


plt.figure(figsize=(5,4))
sns.violinplot(x='Fuel_Type', y='Price', data=df)
plt.title('Violin Plot: Price by Fuel Type')
plt.show()


# In[26]:


plt.figure(figsize=(5,4))
sns.stripplot(x='Transmission', y='Price', data=df, jitter=True, palette='Set2')
plt.title('Strip Plot: Price by Transmission Type')
plt.show()


# ## Summary
# 
# This analysis can help:
# 
# - **Buyers** choose the best car for their budget by understanding what factors drive up prices.
# - **Sellers or Dealers** set competitive prices based on car features.
# - **ML Models** later use this analysis to build accurate car price prediction systems.

# In[ ]:




