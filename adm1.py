# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:41:14 2020

@author: Alessio Sentinelli
"""


#Birthday Cake Candles
import math
import os
import random
import re
import sys
# Complete the 'birthdayCakeCandles' function below.
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
def birthdayCakeCandles(candles):
    max_c = max(candles)
    return candles.count(max_c)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for k in range(N):
    command = input().split()
    if command[0] == "remove":
        s.remove(int(command[1]))
    elif command[0] == "discard":
        s.discard(int(command[1]))
    else:
        s.pop()
print(sum(s))


#Mean, Var, and Std
import numpy 
numpy.set_printoptions(legacy='1.13')

N,M = map(int,input().split())
mat1=[]
for k in range(N):
    lin1 = input().split()
    lin1 = list(map(int,lin1))
    mat1.append(lin1)
mat1 = numpy.array(mat1)
numpy.set_printoptions(sign=' ') 

print (numpy.mean(mat1, axis = 1))
print (numpy.var(mat1, axis = 0))
print (numpy.std(mat1, axis = None))


#Linear Algebra
import numpy
N = int(input())
mat1=[]
for k in range(N):
    lin1 = input().split()
    lin1 = list(map(float,lin1))
    mat1.append(lin1)
mat1 = numpy.array(mat1)
numpy.set_printoptions(legacy='1.13')

print(numpy.linalg.det(mat1))

#Inner and Outer
import numpy
arr1 = input().split()
arr1 = list(map(int,arr1))
arr1 = numpy.array(arr1)
arr2 = input().split()
arr2 = list(map(int,arr2))
arr2 = numpy.array(arr2)
print (numpy.inner(arr1, arr2))
print (numpy.outer(arr1, arr2))

#Dot and Cross
import numpy
N = int(input())
mat1=[]
for k in range(N):
    lin1 = input().split()
    lin1 = list(map(int,lin1))
    mat1.append(lin1)
mat1 = numpy.array(mat1)
mat2=[]
for k in range(N):
    lin1 = input().split()
    lin1 = list(map(int,lin1))
    mat2.append(lin1)
mat2 = numpy.array(mat2)

mat3 = numpy.dot(mat1,mat2)
print(mat3)



#Floor, Ceil and Rint
import numpy as np
vec = input().split()
vec = list(map(float,vec))
vec = np.array(vec)
np.set_printoptions(sign=' ')
print (np.floor(vec) )
print (np.ceil(vec) )
print (np.rint(vec) )


#Array Mathematics
import numpy as np
N,M = map(int,input().split())
mat1=[]
for k in range(N):
    lin1 = input().split()
    lin1 = list(map(int,lin1))
    mat1.append(lin1)
mat1 = np.array(mat1)
mat2=[]
for k in range(N):
    lin2 = input().split()
    lin2 = list(map(int,lin2))
    mat2.append(lin2)
mat2 = np.array(mat2)
print(mat1 + mat2)
print(mat1 - mat2)
print(mat1 * mat2)
print(mat1 // mat2)
print(mat1 % mat2)
print(mat1 ** mat2)



#Zeros and Ones
import numpy
tuple1 = tuple(map(int, input().split()))     
print (numpy.zeros(tuple1, dtype = numpy.int))
print (numpy.ones(tuple1, dtype = numpy.int))


#Polynomials
import numpy
coe = input().split()
coe = list(map(float,coe))
x = float(input())
print (numpy.polyval(coe, x))


#Eye and Identity
import numpy 
n,m = map(int,input().split())
numpy.set_printoptions(sign=' ') 
print(numpy.eye(n,m,k=0))


#Arrays
def arrays(arr):
    list_1 = list(map(float, arr))[::-1]
    return numpy.array(list_1, float)


#Min and Max
import numpy as np
N,M = map(int,input().split())
list_1 = []
for k in range (N):
    line = input().split()
    line = list(map(int, line))
    list_1.append(line)
a = np.array(list_1)
b = np.min(a, axis = 1) 
print (np.max(b))


#Sum and Prod
import numpy as np
N,M = map(int,input().split())
c_v = []
for k in range (N):
    v = input().split()
    v = list(map(int, v))
    c_v.append(v)
c_v = np.array(c_v)
c_v = np.sum(c_v, axis = 0)
print(np.prod(c_v))


#collections.Counter()
from collections import Counter  
X = input()
sho_siz = Counter(map(int,input().split()))
mon = 0
for k in range (int(input())):
    num, pri = map(int, input().split())
    if num in sho_siz and sho_siz[num]>0:
        mon = mon+pri
        sho_siz[num] = sho_siz[num]-1        
print(mon)    



#Check Subset
for k in range(int(input())):
    n = input()
    A = set(input().split())
    m = input()
    B = set(input().split())
    print(B.intersection(A) == A)



#Set .symmetric_difference() Operation
n=int(input())
n_stu = set(input().split())
b = int(input())
b_stu = set(input().split())
u_set = n_stu.symmetric_difference(b_stu)
print(len(u_set))


#Set .difference() Operation
n=int(input())
n_stu = set(input().split())
b = int(input())
b_stu = set(input().split())
u_set = n_stu.difference(b_stu)
print(len(u_set))



#Set .intersection() Operation
n=int(input())
n_stu = set(input().split())
b = int(input())
b_stu = set(input().split())
u_set = n_stu.intersection(b_stu)
print(len(u_set))


#Set .union() Operation
n=int(input())
n_stu = set(map(int, input().split()))

b =int(input())
b_stu = set(map(int, input().split()))

u_set = n_stu.union(b_stu)

print(len(u_set))


#Set .intersection() Operation
n=int(input())
n_stu = set(map(int, input().split()))
b = int(input())
b_stu = set(map(int, input().split()))
u_set = n_stu.intersection(b_stu)
print(len(u_set))


#Set .difference() Operation
n=int(input())
n_stu = set(map(int, input().split()))
b = int(input())
b_stu = set(map(int, input().split()))
u_set = n_stu.difference(b_stu)
print(len(u_set))


#Set .symmetric_difference() Operation
n=int(input())
n_stu = set(map(int, input().split()))
b = int(input())
b_stu = set(map(int, input().split()))
u_set = n_stu.symmetric_difference(b_stu)
print(len(u_set))


#Set Mutations
A=int(input())
n_stu = set(map(int, input().split()))
N = int(input())
for k in range(N):    
     command,pos = input().split()
     p_set = set(map(int, input().split()))
     if command == "intersection_update":
         n_stu.intersection_update(p_set)
     elif command == "update":
         n_stu.update(p_set)
     elif command == "symmetric_difference_update":
         n_stu.symmetric_difference_update(p_set)
     elif command == "difference_update":
         n_stu.difference_update(p_set)
print(sum(n_stu))   



#Symmetric Difference
m = int(input())
m_val = input()
m_lis = m_val.split()
m_lis2 = list(map(int, m_lis))
m_set = set(m_lis2)

n = int(input())
n_val = input()
n_lis = n_val.split()
n_lis2 = list(map(int, n_lis))
n_set = set(n_lis2)

u_set = n_set.union(m_set)
i_set = n_set.intersection(m_set)

r_set = u_set.difference(i_set)

for i in sorted(r_set):
    print(i)


#Set .add()
country = set()
for k in range(int(input())):
    country.add(input().strip())
    
print (len(country))



#Exceptions
for i in range(int(input())):
    try:
        a,b = list(map(int, input().split()))
        print(a//b)
    except ZeroDivisionError as e:
        print('Error Code:',e)
    except ValueError as e:
        print('Error Code:',e)


#Text Wrap
import textwrap
def wrap(string, max_width):
    wrapper = textwrap.TextWrapper(max_width) 
    word = wrapper.fill(text=string)    
    return word
    

#String Split and Join
def split_and_join(line):
    line = line.split(" ")
    line = "-".join(line)
    return(line)


#Mutations
def mutate_string(string, position, character):
    l = list(string)
    l[position] = character
    new_str = ''.join(l)
    return new_str


#What's Your Name?
def print_full_name(a, b):
    print("Hello "+a+" "+b+"! You just delved into python.")


#Text Alignment
#Replace all ______ with rjust, ljust or center.
thickness = int(input()) #This must be an odd number
c = 'H'
#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))  
#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# print 'HackerRank'.center(width,'-')
#print 'HackerRank'.ljust(width,'-')
#print 'HackerRank'.rjust(width,'-')


#Introduction to Sets
def average(array):
    array = set(array)
    return (sum(array)/len(array))


#sWAP cASE
def swap_case(s):
    return s.swapcase()


#Lists
lista = []
lista = []
n = int(input())
for k in range (n):
    com = input().split()
    for i in range(1,len(com)):
        com[i] = int(com[i])
    if com[0]== "insert":
        lista.insert(com[1],com[2])
    elif com[0]== "remove":
        lista.remove(com[1])
    elif com[0]== "append":
        lista.append(com[1])
    elif com[0]== "sort":
        lista.sort()
    elif com[0]== "pop":
        lista.pop()
    elif com[0]== "reverse":
        lista.reverse()
    elif com[0]== "print":
        print(lista)


#Exceptions
for i in range(int(input())):
    try:
        a,b = list(map(int, input().split()))
        print(a//b)
    except ZeroDivisionError as e:
        print('Error Code:',e)
    except ValueError as e:
        print('Error Code:',e)
        
        
        
        
        
#Tuples
n = int(input())
t=tuple(map(int, input().split()))
print(hash(t))


#Transpose and Flatten
import numpy
n,m = map(int,input().split())
arr = numpy.array([input().split() for k in range(n)], int)
print (arr.transpose())
print (arr.flatten())



#Concatenate
import numpy as np 
n,m,p = map(int,input().split())
arr1 = np.array( [input().split() for i in range(n)])
arr1 = arr1.astype(np.int)
arr2 = np.array( [input().split()for i in range(m)])
arr2 = arr2.astype(np.int)
print(np.concatenate((arr1, arr2)))



#Shape and Reshape
import numpy
cha_arr = input()
arr = cha_arr.split()
arr = numpy.array(arr)
arr = arr.astype(numpy.int)

arr.shape = (3, 3)
print (arr)     


#List Comprehensions
x = int(input())
y = int(input())
z = int(input())
n = int(input())

lista = []

for a in range(x+1):
    for b in range (y+1):
        for c in range (z+1):
            if a+b+c != n:
                lista.append([a,b,c])
                
print(lista)


#Print Function
n = int(input())
n_str = ""
for i in range (1,n+1):
    i_str = str(i)
    n_str = n_str + i_str
print(n_str)



#Write a function
def is_leap(year):
    if year%400==0:
        return(True)
    if year%100==0:
        return(False)
    if year%4==0:
        return(True)
    else:
        return(False)


#Python: Division
a = float(input())
b = float(input())
print(int(a//b))
print(a/b)


#Python If-Else
n = int(input())
if n in range(1,101):
    if n%2 ==1 or 6<=n<=20:
        print("Weird")
    else:
        print("Not Weird")


#Loops
for i in range(int(input())):
    print (i**2)



#Arithmetic Operators
a = int(raw_input())
b = int(raw_input())
print(a+b)
print(a-b)
print(a*b)


#Say "Hello, World!" With Python
name = 'Hello, World!'
print (name)