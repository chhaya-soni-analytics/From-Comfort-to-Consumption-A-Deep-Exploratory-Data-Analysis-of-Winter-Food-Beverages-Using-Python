import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/WIN10/Downloads/Winter_Food_and_Beverages.csv") 

df.head()

df.columns

df.shape

# Check info
df.info()


# Fill missing values
df['Calories'] = df['Calories'].fillna(df['Calories'].median())
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())

# Verify
df.isnull().sum()


df["Type"].value_counts()
df["Season"].value_counts()
df["Origin"].value_counts()


#eda


plt.figure(figsize=(6,4))
plt.hist(df['Price(USD)'], bins=10)

plt.title('Price Distribution')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(6,4))
plt.hist(df['Rating'], bins=10)

plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['Calories'], bins=10)

plt.title('Calories Distribution')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.show()


#Category-wise Analysis

#Average Price by Type

avg_price = df.groupby("Type")["Price(USD)"].mean().sort_values() 

avg_price.plot(kind="bar",figsize=(7,4))
plt.title('Average Price by Food Type')
plt.ylabel('Price (USD)')
plt.show()


#Average Rating by Type
avg_rating = df.groupby('Type')['Rating'].mean()

avg_rating.plot(kind='bar', figsize=(7,4))
plt.title('Average Rating by Food Type')
plt.ylabel('Rating')
plt.show() 

# Step 6: Seasonal Analysis

season_price = df.groupby("Season")["Price(USD)"].mean()

season_price.plot(kind="bar",figsize=(6,4))
plt.title('Average Price by Season')
plt.ylabel('Price')
plt.show()


popular_items = df.sort_values('Popularity_Score', ascending=False).head(10)

plt.figure(figsize=(8,4))
plt.bar(popular_items['Item'], popular_items['Popularity_Score'])
plt.xticks(rotation=45)
plt.title('Top 10 Popular Winter Items')
plt.ylabel('Popularity Score')
plt.show()

#Price vs Rating
plt.figure(figsize=(6,4))
plt.scatter(df['Price(USD)'], df['Rating'])
plt.xlabel('Price')
plt.ylabel('Rating')
plt.title('Price vs Rating')
plt.show()

num_df = df[['Price(USD)', 'Calories', 'Rating', 'Popularity_Score']]
num_df.corr()
corr = num_df.corr()



plt.figure(figsize=(6,5))
plt.imshow(corr)

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title('Correlation Heatmap of Numerical Features')
plt.show() 


