# Profit Or Loss

Cost_Price = int(input("Enter The Cost Price: "))
Selling_Price = int(input("Enter The selling Price: "))

Profit = Selling_Price - Cost_Price
Loss = Cost_Price - Selling_Price
Amount = Profit or Loss
if Selling_Price > Cost_Price:
    print("Profit is ",Profit)
    print("Amount is ", Amount)
elif Selling_Price < Cost_Price:
    print("Loss is ",Loss)
    print("Amount is ",Amount)    
else:
    print(" No Profit No Loss")    
