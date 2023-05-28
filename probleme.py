import sys
import os

# n = 100
# sum_squares = 0
# square_sum = 0

# for i in range(0,n+1):
#     sum_squares += i**2
    
# print(sum_squares)

# for i in range (0, n+1):
#     square_sum += i

# square_sum = square_sum*square_sum
# print(square_sum)

# diff = square_sum - sum_squares

# print(diff)

def is_palindrome(s):
    # Convert the string to lowercase and remove non-alphabetic characters
    s = ''.join(filter(str.isalpha, s.lower()))
    
    # Check if the string is equal to its reverse
    return s == s[::-1]

print(is_palindrome('1234321'))