#!/usr/bin/env python
# coding: utf-8

# In[3]:


#SALES PROJECT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the advertising dataset
data = pd.read_csv("C:\\Users\\compaq\\Desktop\\advertising.csv")

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Define features and target
X = data[['TV', 'Radio', 'Newspaper']]  # Features: advertising expenditures on TV, Radio, Newspaper
y = data['Sales']  # Target: Sales

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training (using Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Plotting actual vs. predicted sales
plt.scatter(y_test, predictions)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.show()


# In[ ]:




