
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



