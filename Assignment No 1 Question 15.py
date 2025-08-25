def speed(distance, time):
    speed = distance / time 
    return speed
Distance = float(input("Enter the Distance (in km/miles/etc.): "))
Time = float(input("Enter the time (in hours/minutes/etc.): "))

Speed = Distance / Time
print(f"The Speed is {Speed:.2f} km/h. ")