# Assignment of Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Lets Read the csv file
df = pd.read_csv('Real_Estate_Sales_2001-2022_GL-Short.csv', index_col='Serial Number')

print(df)

print('df - Shape', df.shape)
print('df - info()', df.info())
print('df - describe', df.describe())
print('df - dtypes', df.dtypes)

df.plot.scatter(x='Assessed Value', y='Sale Amount', title='Scatter Plot of Assessed Value and Sale Amount percentages');
plt.show()

X = df[['Assessed Value']].values
y = df['Sale Amount'].values

print('X   ', X)
print('y   ', y)


SEED = 42

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.1, random_state=SEED)

print(X_train)
print(y_train)

# Training a Linear Regression Model

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train,y_train)
print(model.intercept_)
print(model.coef_)


def predict_sale_amount(slope, intercept, AssessedValue):
    return slope*AssessedValue+intercept


score = predict_sale_amount(model.coef_, model.intercept_, [110500.00, 219900.00, 43400.00])
print(score)

score = model.predict([[110500.00]])
print(score)

score = model.predict([[219900.00]])
print(score)

score = model.predict([[43400.00]])
print(score)

y_pred = model.predict(X_test)
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