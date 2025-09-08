import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv.csv')
print(df)

print('df.head -- \n', df.head())

print('df.tail -- \n', df.tail())

print('df.shape -- \n', df.shape)

print("df.describe().round(2).T:    \n",df.describe().round(2).T)

variables = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'median_income']
for var in variables:
    plt.figure()
    sns.regplot(x=var, y= 'median_house_value', data=df).set(title=f'Regression plot of {var} and median_house_value')

    plt.show()

read = input("wait...")

plt.figure()
correlations = df.corr(numeric_only=True)
print('correlations ...\n', correlations)

g = sns.heatmap(correlations, annot=True, cmap='coolwarm').set(title='Heat map of median house value - Pearson Correlations')
plt.show()

read = input('wait...')

y = df['median_house_value']
X = df[['total_rooms', 'total_bedrooms', 'median_income']]

SEED = 200
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
X_train,X_test,y_train,y_test = train_test_split(X, y,
                                                 test_size=0.2, 
                                                 random_state=SEED)

#After splitting the data, we can train our multiple regression model. Notice that now there is no need to reshape our X data, once it already has more than one dimension:
print("X.shape :     \n", X.shape ) 

# Handle missing values
imputer = SimpleImputer(strategy="mean")  # or "median"
X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

#After fitting the model and finding our optimal solution, we can also look at the intercept:
print("regressor.intercept_......\n", reg.intercept_)

#And at the coefficients of the features
print("regressor.coef_ " , reg.coef_)

feature_names = X.columns
model_coefficients = reg.coef_

coefficients_df = pd.DataFrame(data = model_coefficients, 
                              index = feature_names, 
                              columns = ['Coefficient value'])
print(coefficients_df)

#In the same way we had done for the simple regression model, let's predict with the test data:
y_pred = reg.predict(X_test)


results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Actual vs Predicted.....\n" , results)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
 
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

actual_minus_predicted = sum((y_test - y_pred)**2)
actual_minus_actual_mean = sum((y_test - y_test.mean())**2)
r2 = 1 - actual_minus_predicted/actual_minus_actual_mean
print('R²:', r2)

print(" R2 also comes implemented by default into the score method of Scikit-Learn's linear regressor class...\n", reg.score(X_test, y_test))