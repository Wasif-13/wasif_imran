# macro monthly using pandas

import pandas as pd

df = pd.read_csv('macro_monthly.csv',delimiter=',', parse_dates=[10], date_format={'date_added': '%d-%m-%Y'})

print(df)

print("df - data types ", df.dtypes)
print('macro monthly - describe -', df.describe())
print('macro monthly - shape -', df.shape)
print("df info", df.info())
print('daraframe to_string', df.to_string)

# display the last three rows 
print("last four rows:")
print(df.tail(4))

print("first Four rows: ")
print(df.head(4))

# Summary of statistics

print('Summary of Statistics', df.describe())
 #Counting the rows and columns in DataFrame using shape(). It returns the no. of rows and columns enclosed in a tuple.
print("number of rows and columns ", df.shape)

# access the name column
ind_prod = df['Industrial_Production']
print("access the name column : df : ") 
print(ind_prod)
print()

# access single columns
man_ord_durable_goods = df[['Manufacturers_New_Orders: Durable Goods']]
print("access single column: df : ")
print(man_ord_durable_goods)
print()

# Access multiple columns 
ind_prod_man_ord_durable_goods = df[['Industrial_Production', 'Manufacturers_New_Orders: Durable Goods']]
print('Access of multiple columns: ')
print(ind_prod_man_ord_durable_goods)
print()

# selecting a single row using .loc
third_row = df.loc[3]
print("# selecting a single row using .loc")
print(third_row)
print()

# selecting multiple rows using .loc
third_row = df.loc[[3, 5, 7]]
print("# selecting multiple rows using .loc")
print(third_row)
print()

#Selecting a slice of rows using .loc
third_row = df.loc[5:15]
print("#Selecting a slice of rows using .loc")
print(third_row)
print()

# selecting a single column using .loc
third_row = df.loc[:1,'Year']
print('# selecting a single column using .loc')
print(third_row)
print()

# selecting multiple columns using .loc
third_row = df.loc[:2,['Year','Month']]
print('# selecting multiple colums using .loc')
print(third_row)
print()
 
#Selecting a slice of columns using .loc
forth_row = df.loc[1:6]
print("#Selecting a slice of columns using .loc")
print(forth_row)
print()

#Combined row and column selection using .loc
fifth_row = df.loc[df['Year'] == 'Year','Month':'Consumer_Price Index']
print("#Combined row and column selection using .loc")
print(fifth_row)
print()

# Selecting single column with index 9
column = df.loc[:9,['Industrial_Production', 'Retail_Sales', 'Manufacturers_New_Orders: Durable Goods','Personal_Consumption_Expenditures']]
returnvalue = print(column)

# Selecting slice of columns 
slice = df.loc[df['Industrial_Production'] <=0.5, ['Industrial_Production']]
print(slice)


