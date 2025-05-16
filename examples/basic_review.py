"""
Basic example of using the code review model
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.code_correction import review_code

def main():
    """
    Main function to demonstrate code review
    """
    # Example code with syntax errors
    code_with_errors = """
def calculate_sum(a, b)
    return a + b

for i in range(10)
    if i % 2 == 0
        print(f"{i} is even")
    else
        print(f"{i} is odd")

# Function with missing parenthesis
def greet(name
    print("Hello, " + name)

# Variable assignment with incorrect operator
x = 10
if x = 5:
    print("x is 5")
    """
    
    print("Original code:")
    print(code_with_errors)
    print("\n" + "="*50 + "\n")
    
    # Review and correct the code
    corrected_code = review_code(code_with_errors)
    
    print("Corrected code:")
    print(corrected_code)

if __name__ == "__main__":
    main()
