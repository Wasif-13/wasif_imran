
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lets read the csv file
df = pd.read_csv('50_Startups (1).csv')
print('df - dtypes',df.dtypes)
print('df - info()',df.info())
print('df - describe()',df.describe())
print('df - shape',df.shape)

import seaborn as sns

variables = ['R&D Spend', 'Administration', 'Marketing Spend']
for var in variables:
    plt.figure()
    
    sns.regplot(x=var, y='Profit', data=df).set(title=f'Regression plot of {var} and Profit');
    plt.show()

read = input("Wait here: \n")
plt.figure()

"""We can also calculate the correlation of the new variables, this time using Seaborn's heatmap() to help us spot the strongest and weaker correlations based on warmer (reds) and cooler (blues) tones:"""
correlations = df.corr(numeric_only=True)
print('Correlations  \n', correlations)
g = sns.heatmap(correlations, annot=True, cmap='coolwarm').set(title='Heatmap of Profit')
plt.show()
read = input('Wait for me  \n')

y = df['Profit']
X = df[['R&D Spend',
         'Administration',
           'Marketing Spend']]

SEED = 200
from sklearn.model_selection import train_test_split
#After setting our X and y sets, we can divide our data into train and test sets. We will be using the same seed and 10% of our data for training:
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     test_size=0.1,
                                                     random_state=SEED)

print("X.shape... \n", X.shape)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train,y_train)

print('regressor.intercept ', regressor.intercept_)
print('regressor.coef ', regressor.coef_)


feature_name = X.columns
model_coefficients = regressor.coef_

coefficients_df = pd.DataFrame(data=model_coefficients, 
                               index=feature_name,
                               columns=['Coefficient Value'])
print(coefficients_df)


new_input = np.array([[150000, 120000, 100000]])  
predicted_profit = regressor.predict(new_input)
print("Predicted Profit:", predicted_profit[0])


y_pred = regressor.predict(X_test)
results = pd.DataFrame({'Actual ': y_test, 'Predicted': y_pred})
print("Actual vs Predicted...\n", results)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')