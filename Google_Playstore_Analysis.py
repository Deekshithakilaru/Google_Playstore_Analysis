#!/usr/bin/env python
# coding: utf-8

# In[ ]:


project_name = "google-play-store-eda"


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
os.listdir('C:\Users\skkr1\OneDrive\Desktop\DC\googleplaystore')


# In[ ]:


apps_df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
apps_df.head(10)


# In[ ]:


apps_df.sample(10)


# In[ ]:


apps_df['Category'].unique()


# In[ ]:


apps_df['Type'].unique()


# In[ ]:


apps_df['Content Rating'].unique()


# In[ ]:


apps_df.info()


# In[ ]:


reviews = [i for i in apps_df['Reviews']]

def clean_reviews(reviews_list):
    """
    As 'M' has been found the in reviews data, this function
    replace it with million
    """
    cleaned_data = []
    for review in reviews_list:
        if 'M' in review:
            review = review.replace('M', '')
            review = float(review) * 1000000  # 1M = 1,000,000
        cleaned_data.append(review)
    return cleaned_data

apps_df['Reviews'] = clean_reviews(reviews)
apps_df['Reviews'] = apps_df['Reviews'].astype(float)


# In[ ]:


index = apps_df[apps_df['Size'] == '1,000+'].index
apps_df.drop(axis=0, inplace=True, index=index)

sizes = [i for i in apps_df['Size']]

def clean_sizes(sizes_list):
    """
    As sizes are represented in 'M' and 'k', we remove 'M'
    and convert 'k'/kilobytes into megabytes
    """
    cleaned_data = []
    for size in sizes_list:
        if 'M' in size:
            size = size.replace('M', '')
            size = float(size)
        elif 'k' in size:
            size = size.replace('k', '')
            size = float(size)
            size = size/1024  # 1 megabyte = 1024 kilobytes
        # representing 'Varies with device' with value 0
        elif 'Varies with device' in size:
            size = float(0)
        cleaned_data.append(size)
    return cleaned_data

apps_df['Size'] = clean_sizes(sizes)
apps_df['Size'] = apps_df['Size'].astype(float)


# In[ ]:


installs = [i for i in apps_df['Installs']]

def clean_installs(installs_list):
    cleaned_data = []
    for install in installs_list:
        if ',' in install:
            install = install.replace(',', '')
        if '+' in install:
            install = install.replace('+', '')
        install = int(install)
        cleaned_data.append(install)
    return cleaned_data
        
apps_df['Installs'] = clean_installs(installs)
apps_df['Installs'] = apps_df['Installs'].astype(float)


# In[ ]:


prices = [i for i in apps_df['Price']]

def clean_prices(prices_list):
    cleaned_data = []
    for price in prices_list:
        if '$' in price:
            price = price.replace('$', '')
        cleaned_data.append(price)
    return cleaned_data

apps_df['Price'] = clean_prices(prices)
apps_df['Price'] = apps_df['Price'].astype(float)


# In[ ]:


apps_df.sample(10)


# In[ ]:


apps_df.isna().sum()


# In[ ]:


def replace_with_median(series):
    """
    Given a series, replace the rows with null values 
    with median values
    """
    return series.fillna(series.median())

apps_df['Rating'] = apps_df['Rating'].transform(replace_with_median)
apps_df['Rating'] = apps_df['Rating'].astype(float)


# In[ ]:


index = apps_df[apps_df['Type'].isna()].index
apps_df.drop(axis=0, inplace=True, index=index)


# In[ ]:


apps_df.isna().sum()


# In[ ]:


apps_df = apps_df.groupby(['App', 'Reviews', 'Category', 'Rating', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 
                           'Last Updated', 'Current Ver', 'Android Ver'], as_index=False)
apps_df = apps_df['Installs'].mean()
apps_df.sort_values(by='Reviews', ascending=False, inplace=True)
apps_df.drop_duplicates(subset=['App'], inplace=True)
apps_df


# In[ ]:


apps_df.describe()


# In[ ]:


sns.set_style('darkgrid')
plt.figure(figsize=(10, 5))
sns.countplot(x='Category', data=apps_df)
plt.title('Number of Apps Per Category')
plt.xticks(rotation=90)
plt.ylabel('Number of Apps')
plt.show()


# In[ ]:


categories = apps_df.groupby('Category')
category_installs_sum_df = categories[['Installs']].sum()
category_installs_sum_df = category_installs_sum_df.reset_index()  # to convert groupby object into dataframe

plt.figure(figsize=(10, 5))
sns.barplot(x='Category', y='Installs', data=category_installs_sum_df)
plt.xticks(rotation=90)
plt.ylabel('Installs (e+10)')
plt.title('Number of Installs For Each Category')
plt.show()


# In[ ]:


plt.figure(figsize=(10, 5))
sns.countplot(x='Rating', data=apps_df)
plt.title('Rating Distribution')
plt.xticks(rotation=90)
plt.ylabel('Number of Apps')
plt.show()


# In[ ]:


rating_df = apps_df.groupby('Rating').sum().reset_index()

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

axes[0].plot(rating_df['Rating'], rating_df['Reviews'], 'r')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Reviews')
axes[0].set_title('Reviews Per Rating')

axes[1].plot(rating_df['Rating'], rating_df['Size'], 'g')
axes[1].set_xlabel('Rating')
axes[1].set_ylabel('Size')
axes[1].set_title('Size Per Rating')

axes[2].plot(rating_df['Rating'], rating_df['Installs'], 'g')
axes[2].set_xlabel('Rating')
axes[2].set_ylabel('Installs (e+10)')
axes[2].set_title('Installs Per Rating')

axes[3].plot(rating_df['Rating'], rating_df['Price'], 'k')
axes[3].set_xlabel('Rating')
axes[3].set_ylabel('Price')
axes[3].set_title('Price Per Rating')

plt.tight_layout(pad=2)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 5))
sns.countplot(apps_df['Type'])
plt.title('Type Distribution')
plt.ylabel('Number of Apps')
plt.show()


# In[ ]:


plt.figure(figsize=(12, 6))
sns.scatterplot(apps_df['Size'],
               apps_df['Rating'],
               hue=apps_df['Type'],
               s=50)


# In[ ]:


apps_df.corr()


# In[ ]:


fig, axes = plt.subplots(figsize=(8, 8))
sns.heatmap(apps_df.corr(), ax=axes, annot=True, linewidths=0.1, fmt='.2f', square=True)
plt.show()


# Asking and Answering Questions 

# In[ ]:


# 1. What is the top 5 apps on the basis of installs?
df = apps_df.sort_values(by=['Installs'], ascending=False)
df.head(5)


# In[ ]:


print(f'The 5 apps that have the most number of installs are: {", ".join(df["App"].head(5))}')


# In[ ]:


# 2. What is the top 5 reviewed apps?
df = apps_df.groupby(by=['App', 'Category', 'Rating'])[['Reviews']].sum().reset_index()
df = df.sort_values(by=['Reviews'], ascending=False)
df.head(5)


# In[ ]:


print(f'The 5 apps that have the most number of total reviews are: {", ".join(df["App"].head(5))}')


# In[ ]:


# 3. What is the top 5 expensive apps?
df = apps_df.sort_values(by=['Price'], ascending=False)
df.head(5)


# In[ ]:


print(f'The top 5 most expensive apps in the store are: {", ".join(df["App"].head(5))}')


# In[ ]:


# 4. What is the top 3 most installed apps in Game category?
df = apps_df[apps_df['Category'] == 'GAME']
df = df.sort_values(by=['Installs'], ascending=False)
df.head(3)


# In[ ]:


print(f'The top 3 most expensive apps in the GAME category are: {", ".join(df["App"].head(3))}')


# In[ ]:


# 5. Which 5 apps from the 'FAMILY' category are having the lowest rating?
df = apps_df[apps_df['Category'] == 'FAMILY']
df = df.sort_values(by=['Rating'], ascending=True)
df.head(5)


# In[ ]:


print(f'The 5 apps from the FAMILY category having the lowest rating are: {", ".join(df["App"].head(5))}'

