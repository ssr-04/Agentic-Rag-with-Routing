def example_function_part1():
    # Line 1
    print("This is line 1.")
    # Line 2
    x = 10
    # Line 3
    y = 20
    return x, y

def example_function_part2():
    x, y = example_function_part1()
    # Line 4
    z = x + y
    # Line 5
    print(z)

def another_function():
    print("This function should be ignored as it's not 5 lines.")
    return True