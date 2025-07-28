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

def new_five_liner_part1():
    day = "Monday"
    month = "January"
    year = 2023
    return day, month, year

def new_five_liner_part2():
    day, month, year = new_five_liner_part1()
    date_str = f"{day}, {month} {year}"
    return date_str

def iteration_function_part1():
    color = "blue"
    shape = "circle"
    size = "large"
    return color, shape, size

def iteration_function_part2():
    color, shape, size = iteration_function_part1()
    description = f"A {size} {color} {shape}"
    return description

def calculation_function_part1():
    base = 10
    exponent = 2
    result = base ** exponent
    return base, exponent, result

def calculation_function_part2():
    base, exponent, result = calculation_function_part1()
    message = f"{base} raised to the power of {exponent} is {result}"
    return message

def temperature_converter_part1():
    celsius = 25
    factor = 9/5
    offset = 32
    return celsius, factor, offset

def temperature_converter_part2():
    celsius, factor, offset = temperature_converter_part1()
    fahrenheit = celsius * factor + offset
    return f"{celsius}°C = {fahrenheit}°F"

def list_processor_part1():
    numbers = [1, 2, 3, 4, 5]
    squared = [n ** 2 for n in numbers]
    summed = sum(squared)
    return numbers, squared, summed

def list_processor_part2():
    numbers, squared, summed = list_processor_part1()
    average = summed / len(squared)
    return f"Average of squared numbers: {average}"

# Modified to have exactly 3 and 2 lines
def string_manipulator_part1():
    text = "Python programming"
    uppercase = text.upper()
    reversed_text = uppercase[::-1]

def string_manipulator_part2():
    character_count = len(text)
    return f"{reversed_text} has {character_count} characters"