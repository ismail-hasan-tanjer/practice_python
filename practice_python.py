
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

#Decorates [Decorator is a technique used to modify or extend functions.]

def my_decorator(func):
    def wrapper():
        print("Executive before function call")
        func()
        print("Executive after function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello, world!")

say_hello()

#Scikit-Learn 

#Scikit-Learn is the most widely used library in Python, 
# used for machine learning model creation, training, and evaluation.

#Install 

# pip install scikit-learn

#Library Import  

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 

#Working with Datasets (Real-World Example)
#  Scikit-Learn has in-built datasets (e.g. iris, digits, wine, boston housing, etc.).
# We can read CSV files with pandas and analyze them in Scikit-Learn.

# Example of loading a dataset (Iris Dataset)

from sklearn.datasets import load_iris

#Iris dataset load 

iris = load_iris() 
X = iris.data
y = iris.target 

#data frame convert 

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

print(df.head()) #show 5 data 

#Splitting the dataset into training and test sets
# The data model is kept 70% for training and 30% for testing.
# Splitting the training and test data (Train-Test Split)

x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Train Data: {x_train.shape}, Test Data: {X_test.shape}")

#Machine Learning Model Training and Prediction
# (1) Linear Regression (Regression Prediction Model)
# Uses: House price prediction, sales prediction, etc.
# Linear Regression Code:

from sklearn.linear_model import LinearRegression

#Create Model 
model = LinearRegression () 

#Model Training 
model.fit(x_train, y_train)

#Prediction 
y_pred = model.predict(X_test)

#Model Evaluation 
mse = mean_squared_error (y_test, y_pred)
print(f"Mean Squared Error: {mse}") 

#Decision Tree Classifier (Classification Model)
#Uses: Credit Risk Analysis, Disease Diagnosis, etc.
#Decision Tree Code:

from sklearn.tree import DecisionTreeClassifier

#Model Create and Training 
dt_model = DecisionTreeClassifier() 
dt_model.fit(x_train, y_train) 

#Prediction and Accuracy Check 
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = dt_model.score(X_test, y_test)
print(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")

#Random Forest Classifier (Ensemble Model)
#Uses: Fraud Detection, Image Classification, etc.
#Random Forest Code: 

from sklearn.ensemble import RandomForestClassifier

# Model Create 
rf_model = RandomForestClassifier(n_estimators=100)

# Model Training 
rf_model.fit(x_train, y_train)

# Prediction 
y_pred_rf = rf_model.predict(X_test)

# Accuracy Check 
accuracy_rf = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%") 


#Support Vector Machine (SVM)
#Uses: Image processing, text classification, etc.

# SVM code:
from sklearn.svm import SVC

# Model Create 
svm_model = SVC()

# Model Training 
svm_model.fit(x_train, y_train)

# Accuracy Check 
accuracy_svm = svm_model.score(X_test, y_test)
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")  


#Hyperparameter Tuning & Model Optimization
# Models are optimized using Grid Search CV and Randomized Search CV.
# Grid Search CV Example:

from sklearn.model_selection import GridSearchCV

params = {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7]}
grid_search = GridSearchCV(RandomForestClassifier(), params, cv=5)
grid_search.fit(x_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

#Deep Learning (TensorFlow & Keras)
#Deep Learning models are created using Neural Networks.
#Image, text and speech analysis is done using TensorFlow/Keras.

# A simple Neural Network code:

import tensorflow as tf
from tensorflow import keras

# Model Create 
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Model Compile 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Model successfully created and compiled!")

Advanced Python Topics

1️⃣ Python Internals & Memory Management
It is important to understand the inner workings of how Python executes code.

✅ Garbage Collection & Memory Optimization
The Garbage Collector (GC) in Python automatically cleans up unused memory.


import gc

class MyClass:
    def __init__(self, value):
        self.value = value

obj = MyClass(10)
del obj  # Object deleted, but memory may still be in use

gc.collect()  # Force garbage collection


2️⃣ Metaclasses (Dynamic Class Creation)
In Python, classes can be created at runtime using Metaclasses.

#code 

class Meta(type):
    def __new__(cls, name, bases, dct):
        print(f"Creating class: {name}")
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=Meta):
    pass  # When this class is created, metaclass runs

obj = MyClass()

Decorators & Closures (Function Modification)

Function Decorators

def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello, World!")

say_hello()


Generators & Iterators


Lazy Evaluation with Generators

def my_generator():
    for i in range(5):
        yield i

gen = my_generator()
print(next(gen))  # 0
print(next(gen))  # 1

5. Multithreading & Multiprocessing (Concurrency & Parallelism)
Multithreading (For I/O Bound Tasks)

import threading

def print_numbers():
    for i in range(5):
        print(i)

t1 = threading.Thread(target=print_numbers)
t1.start()
t1.join()

**Multiprocessing (For CPU Bound Tasks)

import multiprocessing

def square(x):
    return x * x

pool = multiprocessing.Pool(processes=4)
result = pool.map(square, [1, 2, 3, 4])
print(result)


6️⃣ Regular Expressions (Regex)
✅ Find Email Using Regex

code 

import re

text = "My email is example@email.com"
match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
if match:
    print(match.group())

7️⃣ Web Scraping (BeautifulSoup & Selenium)
✅ Scraping HTML Content

from bs4 import BeautifulSoup
import requests

url = "https://example.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

print(soup.title.text)


8️⃣ Database Handling (SQLite, MySQL, MongoDB)
✅ Using SQLite in Python

import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER, name TEXT)")
cursor.execute("INSERT INTO users VALUES (1, 'John Doe')")

conn.commit()
conn.close()


9️⃣ API Development (FastAPI, Flask, Django Rest Framework)

=> Simple FastAPI Example

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

🔟 Docker & Cloud Deployment (AWS, GCP, Azure)
✅ Dockerizing a Python App

FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]

Step 1: Python Internals & Memory Management
Understanding how Python manages memory and how the backend processes of code execution work is crucial for an advanced Python programmer.

1️⃣ Python Execution Model (How Python Runs Code)
Python is an interpreted language, which means that the CPython interpreter executes the code line by line.

✅ Python code is first compiled to Bytecode (.pyc), then the Python Virtual Machine (PVM) runs it.

✅ Python uses its own Garbage Collector (GC) for memory management.








































    


