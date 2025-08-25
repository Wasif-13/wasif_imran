# Total Marks and Percentage

Sub1 = float(input("Enter marks of 1st Subject:"))
Sub2 = float(input("Enter marks of 2nd Subject:"))
Sub3 = float(input("Enter marks of 3rd Subject:"))
Sub4 = float(input("Enter marks of 4th Subject:"))
Sub5 = float(input("Enter marks of 5th Subject:"))

Total_Marks = Sub1+Sub2+Sub3+Sub4+Sub5

max_marks = 250

Percentage = (Total_Marks/max_marks)* 100
Average = Total_Marks/ 5

print("Total Marks are ",Total_Marks)
print("Percentage ",Percentage)
print("Average ",Average)