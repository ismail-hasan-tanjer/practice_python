
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




















    


