# macro_monthly using numpy

import numpy as np

Year, Industrial_Production, Consumer_Price, Consumer_Confidence = np.genfromtxt('macro_monthly.csv',delimiter=',',unpack=True, dtype=None, skip_header=1,invalid_raise=False,usecols=(0,2,4,6))

print(Year)
print(Industrial_Production)
print(Consumer_Price)
print(Consumer_Confidence)

# macro monthly Consumer_Price - Statistics Operations
print('Consumer Price Mean',np.mean(Consumer_Price))
print('Consumer Price Average',np.average(Consumer_Price))
print('Consumer Price std',np.std(Consumer_Price))
print('Consumer Price mod',np.median(Consumer_Price))
print('Consumer Price percentile - 25',np.percentile(Consumer_Price,25))
print('Consumer Price percentile - 70',np.percentile(Consumer_Price,70))
print('Consumer Price percentile - 5',np.percentile(Consumer_Price,5))

# macro monthly maths operations 
print('Consumer Price Square',np.square(Consumer_Price))
print('Consumer Price Sqrt',np.sqrt(Consumer_Price))
print('Consumer Price abs',np.abs(Consumer_Price))

# macro monthly arithmetic operations
add = Consumer_Confidence + Consumer_Price
sub = Consumer_Confidence - Consumer_Price
mul = Consumer_Confidence * Consumer_Price
div = Consumer_Confidence / Consumer_Price

print("Addition",add)
print("subtraction",sub)
print("Multiply",mul)
print("Division",div)

# macro monthly Trignometric Functions

pricepie = (Consumer_Price/np.pi) + 1
# Calculate Sine Cosine And tangents
sine = np.sin(pricepie)
cosine = np.cos(pricepie)
tangent = np.tan(pricepie)

print("Sine values",sine)
print("Cosine values ",cosine)
print("tangent values ",tangent)


