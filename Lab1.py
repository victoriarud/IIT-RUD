import numpy as np
import time


def task_1(n):
    matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        matrix[i, i] = i + 1

    return matrix

# Measure execution time
start_time = time.time()

n = 5
result = task_1(n)
for row in result:
    print(row)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds", end='\n\n\n\n')




def task_2(n):
    # Create an n x n matrix filled with zeros using numpy
    matrix = np.zeros((n, n), dtype=int)
    
    # Fill the matrix with 1 and 0 in a checkerboard pattern
    for i in range(n):
        for j in range(n):
            # If the sum of indices is even, set the element to 1
            if (i + j) % 2 == 0:
                matrix[i, j] = 1

    return matrix

# Measure execution time
start_time = time.time()

n = 5
result = task_2(n)
for row in result:
    print(row)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds", end='\n\n\n\n')



def task_3(n, m, r, c):
    # Create an n x m matrix filled with ones using numpy
    matrix = np.ones((n, m), dtype=int)
    
    # Fill the row with zeros at index r
    matrix[r, :] = 0

    # Fill the column with zeros at index c
    matrix[:, c] = 0

    return matrix

# Measure execution time
start_time = time.time()

n, m, r, c = 3, 7, 2, 3
result = task_3(n, m, r, c)
for row in result:
    print(row)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds", end='\n\n\n\n')



def task_4(n, m):
    # Create an n x m matrix filled with zeros using numpy
    matrix = np.zeros((n, m), dtype=int)

    # Fill the first row with numbers from 0 to m-1
    matrix[0, :] = np.arange(m)

    return matrix

# Measure execution time
start_time = time.time()

n, m = 4, 3
result = task_4(n, m)
for row in result:
    print(row)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds", end='\n\n\n\n')




def task_5(n):
    # Create an n x n matrix filled with zeros using numpy
    matrix = np.zeros((n, n), dtype=int)

    # Fill the rows with ones based on the condition
    for i in range(n):
        if i % 2 == 0:
            matrix[i, :] = 1

    return matrix

# Measure execution time
start_time = time.time()

n = 5
result = task_5(n)
for row in result:
    print(row)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds", end='\n\n\n\n')




def task_6():
    # Read the array from input
    arr = np.array(list(map(int, input().split())))

    # Replace all non-zero elements with -1
    result = np.where(arr != 0, -1, 0)

    return result.tolist()  # Convert back to list for consistent output format

# Measure execution time
start_time = time.time()

input_array = "3 4 0 9 7 0 6 0 4 0 3"
result = task_6()
print(result)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds", end='\n\n\n\n')




def task_7():
    # Read the array from input
    arr = np.array(list(map(int, input().split())))

    # Replace all zero elements with -1
    result = np.where(arr == 0, -1, arr)

    return result.tolist()  # Convert back to list for consistent output format

# Measure execution time
start_time = time.time()

input_array = "3 4 0 6 5 0 3 0 4"
result = task_7()
print(result)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds", end='\n\n\n\n')



def task_8():
    # Read the array from input
    arr = np.array(list(map(int, input().split())))

    # Count the number of zero and non-zero elements
    zeros_count = np.count_nonzero(arr == 0)
    non_zeros_count = len(arr) - zeros_count

    print(f"Нулів: {zeros_count}")
    print(f"Не нулів: {non_zeros_count}")

# Measure execution time
start_time = time.time()

input_array = "3 4 0 9 8 2 4 0 8 4 0"
task_8()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds", end='\n\n\n\n')



def task_9():
    n = int(input("Введіть число n: "))
    
    # Create an array of values from n down to 0 using numpy
    arr = np.arange(n, -1, -1)
    
    return arr.tolist()  # Convert back to list for consistent output format

# Measure execution time
start_time = time.time()

result = task_9()
print(result)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds", end='\n\n\n\n')




def task_12(n):
    # Create an n x n matrix filled with ones using numpy
    matrix = np.ones((n, n), dtype=int)

    # Replace the "frame" with zeros using numpy operations
    matrix[0, :] = 0
    matrix[-1, :] = 0
    matrix[:, 0] = 0
    matrix[:, -1] = 0

    return matrix.tolist()  # Convert back to list for consistent output format

# Measure execution time
start_time = time.time()

n = 4
result = task_12(n)
for row in result:
    print(row)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds", end='\n\n\n\n')




def task_13():
    # Create an 8x8 matrix filled with zeros using numpy
    matrix = np.zeros((8, 8))

    # Fill the matrix in a checkerboard pattern using numpy operations
    matrix[1::2, ::2] = 1
    matrix[::2, 1::2] = 1

    return matrix

def task_10(matrix):
    # Calculate the minimum, maximum, mean, and variance using numpy operations
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    mean_val = np.mean(matrix)
    variance_val = np.var(matrix)

    # Round the values to 3 decimal places
    min_val = round(min_val, 3)
    max_val = round(max_val, 3)
    mean_val = round(mean_val, 3)
    variance_val = round(variance_val, 3)

    return min_val, max_val, mean_val, variance_val

# Measure execution time for task_13
start_time = time.time()
matrix = task_13()
for row in matrix:
    print(row)
end_time = time.time()
execution_time_13 = end_time - start_time

# Measure execution time for task_10
start_time = time.time()
min_val, max_val, mean_val, variance_val = task_10(matrix)
print("\nмінімум:", min_val)
print("максимум:", max_val)
print("середнє:", mean_val)
print("дисперсія:", variance_val)
end_time = time.time()
execution_time_10 = end_time - start_time

print(f"\nExecution time for task_13: {execution_time_13} seconds")
print(f"Execution time for task_10: {execution_time_10} seconds", end='\n\n\n\n')




def task_13():
    # Create an 8x8 matrix filled with zeros using numpy
    matrix = np.zeros((8, 8))

    # Fill the matrix in a checkerboard pattern using numpy operations
    matrix[1::2, ::2] = 1
    matrix[::2, 1::2] = 1

    return matrix

def tile_matrix(matrix):
    # Determine the dimensions of the input matrix
    n, m = matrix.shape

    # Use the `tile` function to tile the matrix to a 32x8 matrix
    tiled_matrix = np.tile(matrix, (4, 1))

    return tiled_matrix

# Measure execution time for create_checkerboard_matrix
start_time = time.time()
checkerboard_matrix = task_13()
end_time = time.time()
execution_time_13 = end_time - start_time

print("Checkerboard Matrix:")
for row in checkerboard_matrix:
    print(row)

print(f"\nExecution time for create_checkerboard_matrix: {execution_time_13} seconds\n")

# Measure execution time for tile_matrix
start_time = time.time()
tiled_matrix = tile_matrix(checkerboard_matrix)
end_time = time.time()
execution_time_11 = end_time - start_time

print("Tiled Matrix:")
for row in tiled_matrix:
    print(row)

print(f"\nExecution time for tile_matrix: {execution_time_11} seconds", end='\n\n\n\n')




def task_20(n):
    # Generate a vector with n elements in the interval (0, 1)
    vector = np.linspace(0, 1, n, endpoint=False)[1:]  # Exclude the first element which is 0
    
    # Round the values to 3 decimal places
    rounded_vector = np.round(vector, 3)
    
    return rounded_vector

# Measure execution time for generate_rounded_vector
start_time = time.time()

n = 10
result = task_20(n)

end_time = time.time()
execution_time_20 = end_time - start_time

print(result)
print(f"\nExecution time for generate_rounded_vector: {execution_time_20} seconds", end='\n\n\n\n')





def task_17(n):
    # Create a vector from 0 to n
    vector = np.arange(n+1)
    
    # Replace values that satisfy the condition with 0
    mask = (vector > n/4) & (vector < 3*n/4)
    vector[mask] = 0
    
    return vector

# Measure execution time for modify_vector
start_time = time.time()

n = 10
result = task_17(n)

end_time = time.time()
execution_time_17 = end_time - start_time

print(result)
print(f"\nExecution time for modify_vector: {execution_time_17} seconds", end='\n\n\n\n')





def task_19(n):
    # Generate a vector with n random numbers in the range from 0 to 99
    vector = np.random.randint(0, 100, n)
    
    # Find unique values and their counts
    unique_values, counts = np.unique(vector, return_counts=True)

    return unique_values, counts

# Measure execution time for generate_unique_values_and_counts
start_time = time.time()

n = 30
unique_values, counts = task_19(n)

end_time = time.time()
execution_time_19 = end_time - start_time

print("Унікальні значення:", unique_values)
print("Кількість кожного унікального значення:", counts)
print(f"\nExecution time for generate_unique_values_and_counts: {execution_time_19} seconds")
