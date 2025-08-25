# Salary Calculator

Basic = int(input("Enter Basic Salary: "))

HRA = (Basic*20)/100
DA = (Basic*15)/100
Total_Salary = Basic + HRA + DA

print(Total_Salary)