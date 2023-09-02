# import necessary libraries..


import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data  - number of bedrooms and their corresponding house prices..

bedrooms = np.array([1, 2, 3, 4, 5])
house_prices = np.array([100000, 150000, 200000, 250000, 300000])

# Reshape the data (needed for single feature)
bedrooms = bedrooms.reshape(-1, 1)


# create linear regressive model

model = LinearRegression()

# Fit the model to the data

model.fit(bedrooms, house_prices)

# predict the price of the house with 6 bedrooms

predicted_price = model.predict([[10]])

print("Predicted price for the 6 bedroom house: ", predicted_price[0])

