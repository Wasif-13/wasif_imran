
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Lets read the csv file
df = pd.read_csv('number-of-registered-medical-and-dental-doctors-by-gender-in-pakistan (1).csv', index_col='Years', thousands=',')
print(df)

print('df - shape', df.shape)
print('df - describe()', df.describe())
print('df - info()', df.info())
print('df - dtypes', df.dtypes)

df.plot.scatter(x='Female Doctors', y='Female Dentists', title= 'Scatterplot between Female Doctors and Female Dentists percentage')
plt.show()

X = df[['Female Doctors']].values.reshape(-1,1)
y = df[['Female Dentists']].values.reshape(-1,1)

print('X  ', X)
print('y  ', y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training a Linear Regression Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)
print(model.intercept_)
print(model.coef_)

def predicted_doctors(slope,intercept, Female_Doctors):
    return slope*Female_Doctors+intercept

score = predicted_doctors(model.intercept_, model.coef_, [3146,5407,7180])
print(score)

score = model.predict([[3146]])
print(score)

score = model.predict([[5407]])
print(score)

score = model.predict([[7180]])
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