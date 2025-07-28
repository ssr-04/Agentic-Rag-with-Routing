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

def five_line_function_part1():
    a = 1
    b = 2
    c = 3
    return a, b, c

def five_line_function_part2():
    a, b, c = five_line_function_part1()
    d = a + b + c
    print(f"Result: {d}")

def another_five_line_function_part1():
    name = "Python"
    version = 3.9
    is_cool = True
    return name, version, is_cool

def another_five_line_function_part2():
    name, version, is_cool = another_five_line_function_part1()
    message = f"{name} {version} is cool: {is_cool}"
    return message

def combinable_function_part1():
    x = 100
    y = 200
    z = 300
    return x, y, z

def combinable_function_part2():
    x, y, z = combinable_function_part1()
    result = x + y + z
    print(f"Combined: {result}")

def cycle_function_part1():
    step1 = "Initialize"
    step2 = "Process"
    step3 = "Analyze"
    return step1, step2, step3

def cycle_function_part2():
    step1, step2, step3 = cycle_function_part1()
    step4 = "Report"
    step5 = "Conclude"
    return [step1, step2, step3, step4, step5]

def perfect_match_part1():
    first = 10
    second = 20
    third = 30
    return first, second, third

def perfect_match_part2():
    first, second, third = perfect_match_part1()
    total = first + second + third
    print(total)

# Split using Rule 1
def new_five_liner_part1():
    day = "Monday"
    month = "January"
    year = 2023
    return day, month, year

def new_five_liner_part2():
    day, month, year = new_five_liner_part1()
    date_str = f"{day}, {month} {year}"
    return date_str