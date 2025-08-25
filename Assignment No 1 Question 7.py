def distribute_candies(n,k):
    candies_per_student = n // k
    candies_left = n % k

    return candies_per_student, candies_left
# Test the function
n = int(input("Enter the number of candies: "))
k = int(input("Enter the number of students: "))

candies_per_student, candies_left = distribute_candies(n,k)

print(f'Each student gets {candies_per_student} candies.')
print(f'{candies_left} candies are left.')
