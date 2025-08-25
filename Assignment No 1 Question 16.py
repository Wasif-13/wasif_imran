def Body_Mass_Index(height, weight):
    BMI = weight / (height**2)
    return BMI
Height = float(input("Enter your Height: "))
Weight = float(input("Enter your Weight: "))

BMI = Weight / (Height**2)
print(f"Body Mass Index is {BMI}.")