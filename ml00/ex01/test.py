from TinyStatistician import TinyStatistician

a = [1, 42, 300, 10, 59]
print(TinyStatistician().mean(a))
# Output:
print(82.4)

print(TinyStatistician().median(a))
# Output:
print(42.0)

print(TinyStatistician().quartile(a))
# Output:
print([10.0, 59.0])

print(TinyStatistician().percentile(a, 10))
# Output:
print(4.6)

print(TinyStatistician().percentile(a, 15))
# Output:
print(6.4)

print(TinyStatistician().percentile(a, 20))
# Output:
print(8.2)

print(TinyStatistician().var(a))
# Output:
print(15349.3)

print(TinyStatistician().std(a))
# Output:
print(123.89229193133849)