# Assignmnet2 of Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Lets Read the csv file
df = pd.read_csv('zameencom-property-data-By-Kaggle-Short.csv', index_col="property_id", sep=';')

print(df)
print('df - describe()', df.describe())
print('df - shape', df.shape)
print('df - info()', df.info())
print('df - dtypes', df.dtypes)

df.plot.scatter(x='bedrooms', y='price', title='scatter plot of bedrooms and price percentage')
plt.show()

X = df['price'].values.reshape(-1,1)
y = df['bedrooms'].values.reshape(-1,1)

print('X   ', X)
print('y   ',y)

SEED = 50

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=SEED)

print(X_train)
print(y_train)

# Training a Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train,y_train)
print(regressor)
print(regressor.intercept_)
print(regressor.coef_)

def calc(slope, intercept, price):
    return slope*price+intercept

score = calc(regressor.coef_, regressor.intercept_, [1,5,7])
print(score)

score = regressor.predict([[1]])
print(score)

score = regressor.predict([[5]])
print(score)

score = regressor.predict([[7]])
print(score)

y_pred = regressor.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

#We will also print the metrics results using the f string and the 2 digit precision after the comma with :.2f:

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')