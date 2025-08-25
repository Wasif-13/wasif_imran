def Minutes_to_hours(Minutes, Hours):
    min_to_hours = Minutes // 60
    remaining_min = Minutes % 60
    return Minutes_to_hours

Min = int(input("Enter the number of minutes: "))
hours = Min // 60
rem_min = Min % 60

print(f"{Min} minutes is equal to {hours} hours and {rem_min} minutes.")