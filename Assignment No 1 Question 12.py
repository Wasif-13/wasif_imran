def usd_to_pkr(usd_amount, excahange_rate):
    pkr_amount = usd_amount * excahange_rate
    return pkr_amount
USD_amount = float(input("Enter amount in USD: "))

exchange_rate = USD_amount * 280.59 # Current exchange rate

print(f"Amount in PKR is {exchange_rate}: ")
