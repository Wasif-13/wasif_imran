# list of a student
student = [13, 21, "Wasif", 984, 89.6, True]

print(student)

print(student[0])
print(student[1])
print(student[2])
print(student[3])
print(student[4])
print(student[5])


# checking data type of student list
print(type(student))
print(type(student[2]))
print(type(student[3]))
print(type(student[5]))
print(type(student[4]))


student.append(10)
print("After append10",student)

student.insert(0,5.5)
print("After insert",student)

student.remove(13)
print("After remove",student)

student.pop(1)
print("After index",student)

del student[4]
print("after del",student)