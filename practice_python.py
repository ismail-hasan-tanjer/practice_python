
print("Hello, Python!")


if 10 > 5:
    print("10 is greater than 5")  
    
    
# variable 

name = "Tanjer"
age = 28
height = 5.4
is_student = True 
print (name, age, height, is_student)

#data type 
# build in type data int, float, str, bool 

#integer int 

x = 100 
y = 50 
z = 0 
print (type(x)) 

#Floating Point (float) 
#any decimal number save by float 

pi = 3.1416
g = 9.8 
big_number = 1.5e4
print(type(pi))

#string (str)

name = "python"
message = "Hello, World!"
multi_line = """This is 
a multi_line
string."""
print(type(name))

#Boolean (bool)
#Boolean accept true or false two type values 
#used for logical operations and conditions 

is_python_easy = True
is_java_hard = False 
print(type(is_python_easy))

#Type conversion (type casting) 

#Integer to  float 
x = 10
y = float(x)

#float to integer 

z = 5.9 
w = int (z)

#string to integer 
s = "100"
num = int(s)

# integer to string

num_str = str (num)
print(y, w, num, num_str)

#Python control flow 
#conditional statements (if, elif, else)
#Loops (for, while)
#List Comprehension

#conditional statements (if, elif, else)

age = 18

if age >= 18:
    print("You are adult.")
elif age >=13:
    print("You are a teenager")
else: 
    print("You are a  child.")  #when we take conditional decision
    
#Loops (for, while) 
#Continuous working In python used loop 

#for loop 

for i in age (5): 
    print("Iteration:", 1)
    
#while loop 

x = 0
while x < 5: 
    print("Number:", x)
    x+=1
    
#list Comprehension 

numbers = [x**2 for x in range(5)]
print(numbers) 

#python data structure 

# => Lists (List) - Mutable Collection 
# => Tuples (Tuple) - Immutable Collection  
# => Sets (Set)  - Unique Items Collection 
# => Dictionaries (Dict) - Key-Value Pair Collection 

#List (Mutable, Ordered)

fruits = ["Apple", "Banana", "Mango"]
fruits.append("Orange") #add new element 
print(fruits[1])

#Tuple (Immutable, Ordered)

coordinate = (10, 20)
print(coordinate[0])

#Set [Unique, Unordered] 

unique_numbers = {1, 2, 3, 4, 4, 5}
print(unique_numbers)

#Dictionary (Key-Value Pair, Mutable)

person = {"name": "Tanjer", "age":28, "city": "Dhaka"}
print(person["name"])


#Functions and Modules 

#Defining Functions 
#Lambda Functions 
#Importing Modules 

#Function Example 

def greet(name):
    return "Hello, {}!".format(name)
print(greet("Tanjer"))

#Lambda Function 
square = lambda x: x**2
print(square(5))

#Importing Modules 

import math
print(math.sqrt(16))


#Object-Oriented Programming (OOP) in Python 

#classes & Objects 
#Inheritance & Polymorphism 

#class and objects 

class Person: 
    def _init_(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return "My name is {}".format(self.name) and "I am {} years old.".format(self.age)
    
    #file handling 
    
    #file write 
    with open("example.txt", "w") as file: file.write("Hellow, Python!")
    
    #file read 
    
    with open ("example.txt", "r") as file: 
        content = file.read()
        print(content)
    
    # Exception Handling
    #Error Management 
    
    try: 
        num = int(input("Enter a number:"))
        print (10/num)          
    except ZeroDivisionError: 
        print("You cannot divide by zero!")
    except ValueError: 
        print("Invalid input! Please enter a number.")
        
#Regular Expressions (Regex)

#Find digit 

import re 

text = "My phone number is 01938568752"
pattern = r"\d{3}-\d{3}-\d{4}"

match = re.search(pattern, text)
if match: 
    print("Phone number found:", match.group())
else: 
    print("No match found.")

#Found Image Address: 

import re

text = "Contact us at support@example.com or info@domain.com."
pattern = r"[\w\.-]+@[\w\.-]+"  

matches = re.findall(pattern, text)
print("Email addresses found:", matches)


#Text Replace 

import re

text = "The rain in Bangladesh  falls mainly in the plain."
pattern = r"rain"
replacement = "snow"

new_text = re.sub(pattern, replacement, text)
print("Modified text:", new_text)

#String frist to last matching 
import re

text = "Hello, World!"
pattern = r"^Hello"  

if re.match(pattern, text):
    print("Match found at the beginning.")
else:
    print("No match found.")

#Regex Grouping 

import re

text = "Date: 2023-10-05"
pattern = r"(\d{4})-(\d{2})-(\d{2})"  # divide day, month, year

match = re.search(pattern, text)
if match:
    print("Year:", match.group(1))
    print("Month:", match.group(2))
    print("Day:", match.group(3))
    
#Email Validation 

import re

def validate_email(email): 
    
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

print(validate_email("tanjerinfo@gmail.com"))  # True
print(validate_email("tanjerinfo@gmail"))      # False

#Check Bangladeshi Number
import re

def is_valid_bd_phone(number):
    pattern = r"^(01[3-9]\d{8})$"
    return bool(re.match(pattern, number))

print(is_valid_bd_phone("01711112222"))  # True
print(is_valid_bd_phone("0199999"))      # False

#Create Thread 
import threading

def print_numbers():
    for i in range(5):
        print(i)

t1 = threading.Thread(target=print_numbers)  # create thread 
t1.join()

#thread run  
t1.start() 

#Main Thread 
print("Main thread is running.....")


#Multiple Threads 

import threading 

def task (name): 
    for i in range(3):
        print(f"{name} is running iteration {i}")
#create two thread 
t1 = threading.Thread(target=task, args=("Thread-1",))
t2 = threading.Thread(target=task, args=("Thread-2",))

#run thread 
t1.start()
t2.start() 

#wait untill finished thread 

t1.join()
t2.join()

print("All threads completed")

#Efficient Web Scraping by Multithreading 

import threading
import requests 
urls = [
    "https://www.amardesh.com",
    "https://www.python.org",
    "https://www.github.com",
]

def fetch_data(url):
    response = requests.get(url)
    print(f"Fetched {len(response.text)} characters from {url}")

#Thread 
threads = []
for url in urls:
    t = threading.Thread(target=fetch_data, args=(url,))
    threads.append(t)
    t.start()

# wait for all great 
for t in threads:
    t.join()

print("All requests completed!")

#Multiprocessing 

import multiprocessing
import time 

def worker (name): 
    for i in range(3):
        print(f"{name} is running iteration {i}")
        time.sleep(1) # wait for 1 second 
if __name__ == "__main__":
    #create two process 
    p1 = multiprocessing.Process(target=worker, args=("Process-1",))
    p2 = multiprocessing.Process(target=worker, args=("Process-1",))

#process run 
p1.start()
p2.start()

#wait for untill finished 
p1.join()
p2.join()

print("All processes completed!") 

#python machine learning 

#Mean Median Mode 
import numpy

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = numpy.mean(speed)

print(x)


#Numerical Computing Library

#NumPy Install 

#  => pip install numpy

#Create Array by Numpy

import numpy as np

# Ekmatrik (1D) Array
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)

# Dimatrik(2D) Array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)

# Treematrik (3D) Array
arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr3)

#Array Shape, Size, Type check 

print(arr2.shape)  # (2, 3) -example: 2 row, 3 column
print(arr2.size)   # Have a  total element
print(arr2.dtype)  # Datatype (int, float etc.)


#Zeros, Ones, Identity Matrix Create 
print(np.zeros((3, 2))) #[3x3 matrix, all 0]
print(np.ones((2, 2))) #2x2 matrix, all 1 
print (np.eye(3)) #3x3 Identity Matrix (Diagonal 1)

#Number Create by Random 

print(np.random.rand(3, 3))  # 3x3 Martix (0-1 in random number)
print(np.random.randint(1, 10, (2, 2)))  # From 1 to 10, random 2x2 matrix 


#Array Indexing & Slicing

arr = np.array([10, 20, 30, 40, 50])
print(arr[0])      
print(arr[1:4])    
print(arr[-1])     # end element 

#Matrix Operations

#Element-wise

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(a + b)  # Sum
print(a - b)  # Subtraction
print(a * b)  # Multiplication (Element-wise)
print(a / b)  # Division 


#Dot Product (Matrix Multiplication)
print(np.dot(a, b)) 


#Statistical Operations in NumPy

arr = np.array([1, 2, 3, 4, 5])
print(np.mean(arr))  # (Mean)
print(np.median(arr))  # (Median)
print(np.std(arr))  # (Standard Deviation)
print(np.var(arr))  # (Variance)


#NumPy Practice Task 
arr = np.array([[3, 6, 9], [2, 5, 8], [1, 4, 7]])
print(np.max(arr, axis=0))  # Accordingly high rate
print(np.min(arr, axis=1))  # Accordingly lowest rate 

#Pandas - Data Analysis Library 

#pandas install 

# pip install pandas

#Pandas Series 

#Single Column Data 

import pandas as pd

data = [10, 20, 30, 40, 50]
series = pd.Series(data)

print(series)

# Pandas DataFrame
#Table circle data 
#A DataFrame is a 2D database with multiple columns. 

data = {
    "Name": ["Tanjer", "Rahim", "Karim"],
    "Age": [25, 30, 35],
    "Salary": [50000, 60000, 70000]
}

df = pd.DataFrame(data)
print(df) 

#data load from CSV File 

df = pd.read_csv("data.csv")
print(df.head())  # show first 5 row 
print(df.tail())  # show last 5 row 

#Selecting specific columns and rows from a DataFrame

print(df["Name"])  # show selected column 
print(df.iloc[0])  # show first row 
print(df.loc[0, "Salary"])  # show data from selected data column and row 


#Data Analysis 

print(df.describe())  # Summary statistics showed
print(df.info())  # type of data set and missing values showed
print(df["Age"].mean())  # Finding the average age 

#Data Filtering 
filtered_df = df[df["salary"]>5500]
print(filtered_df)

#Missing Values Handling 
df.fillna(0, implace=True) #Missing Values fullfil by 0
df.dropna(implace=True) #Missing Values row totally out 

#NumPy  - Numerical Computation 

# NumPy (Numerical Python) is the most popular Python library for high-performance numerical computation.
# Can work with multi-dimensional arrays & matrices.
# It is essential in Machine Learning, Data Science and AI.

#NumPy Install 

#pip install numpy

# Creating a NumPy Array
# 1D Array (one-dimensional array)

import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))  # numpy.ndarray 


#2D Array (two-dimensional matrix)

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix)

#Knowing the size of an array (Shape & Size)

print(matrix.shape)  # (2, 3) -> 2 rows, 3 columns
print(matrix.size)   # Total number of elements
print(matrix.ndim)   # How many (dimensions) array 

#Important functions of NumPy

#Creating an Array of Zeros and Ones

zeros = np.zeros((3, 3))  # 3x3 matrix, all values ​​0
ones = np.ones((2, 4))  # 2x4 matrix, all values ​​1
print(zeros)
print(ones)

#Generating random numbers

random_numbers = np.random.rand(3, 3) #3x3 matrix random number 
print(random_numbers)

#Changing the elements of an array 

arr[0] = 100  # First element change
matrix[1, 2] = 99  # Change the value of the 3rd column of the 2nd row.

#Arithmetic operations of Array

arr = np.array([10, 20, 30, 40])
print(arr+5) # Adding 5 to each element
print(arr * 2) # Doubling each element
print(np.sqrt(arr)) # Square root of each element

#Operations between two Arrays 
arr1 = np.array([1, 2, 3])
arr2 = np.array([4,  5, 6])

print(arr1 + arr2) # Adding by ingredient
print(arr1 * arr2) # Multiply by elements 

#Matrix Multiplication 

A = np.array([1, 2], [3, 4])
B = np.array([5, 6], [7, 8])

result = np.dot(A, B)
print(result)

#Lambda Functions

#Lambda functions are short, one-line functions that can be written without using def.

#General Function 
def square(x):
    return x * x

#Same work by Lambda Function 

square_lambda = lambda x: x * x

print (square(5))
print (square_lambda) 

#add by lambda 
square_lambda = lambda x: x + x 
print (square_lambda(5))

#Generators (Memory Efficient Iterators)

def my_generator():
    yield 1
    yield 2
    yield 3

gen = my_generator()
print(next(gen))  # Output: 1
print(next(gen))  # Output: 2
print(next(gen))  # Output: 3


























































    


