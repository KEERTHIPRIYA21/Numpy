#!/usr/bin/env python
# coding: utf-8

# In[60]:


#IMPORT STATEMENT
import numpy as np


# ## ARRAY CREATION IN NUMPY

# In[165]:


#ONE DIMENSIONAL ARRAY
OneD=np.array([1,2,3,4,5,6,7,8,9,10])
print(OneD,end="\n \n")

#Two Dimensional array: COLLECTION OF 1D Arrays
TwoD=np.array([[1,2],[3,4]])
print(TwoD,end="\n \n")

#Three Dimensional array: COLLECTION OF 2D Arrays
ThreeD=np.array([[[1,2,3],[4,5,6]],[[10,20,30],[40,50,60]]])
print(ThreeD,end="\n \n")

#More than three
MoreD=np.array([[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]])
print(MoreD)


# # DATATYPES

# In[248]:


#ARRAY DATATYPES
print(a1.dtype)
print(a2.dtype)


# In[78]:


##INTEGER TYPE ARRAY
a1=np.array([[1,2,3],[4,5,6]])
print(a1)
print(a1.dtype,end="\n \n")

#FLOAT TYPE ARRAY
a2=np.array([[1.1,2.2],[2,3]])
print(a2)
print(a2.dtype,end="\n \n")

#ARRAY WITH COMPLEX NUMBERS
c=np.array([1,2],complex)
print(c)
print(c.dtype,end="\n \n")


# In[264]:


#another method 
a1=np.array([1.1,20.20],dtype=np.int32)
print(a1.dtype)
print(a1)
a3=np.array([1,20],dtype=np.float64)
print(a3.dtype)
a2=np.array([10,20],dtype=complex)
print(a2.dtype)


# ## FILLER ARRAYS

# In[119]:


#array with Garbage values 
a=np.empty((3,3),int) 
print(a)
#array with Zeroes
b=np.zeros((3,4),int) 
print(b)
#array with Zeroes values 
c=np.ones((3,4),int)
print(c) 
#array with Zeroes values 
d=np.full((3,4),6)
print(d)


# #### ARRAYS WITH RANDOM VALUES or with evenly space values

# In[118]:


a=np.random.random((4))
print(a)
print()
b=np.random.random((4,2))
print(b)
print()
print()
c=np.random.random((4,2,2))
print(c)
#etc


# In[126]:


a=np.arange(10)
print(a)


# In[127]:


a=np.arange(0,10,2)
print(a)


# In[146]:


a=np.linspace(0,10,11)
print(a)


# In[162]:


"""IDENTITY MATRIX CREATION
Using 
np.eye(size) or
np.identity(size)
"""
s=np.eye(4)
print(s)
s=np.identity(4,int)
print(s)
s=np.identity(4,float)
print(s)
s=np.identity(4,complex)
print(s)


# # Array Attributes 
# #### -> dtype, itemsize, nbytes, data, strides <br />-> ndim, shape, size
# #### dtype -> Datatype of elemenets contained in the array <br /> itemsize -> Number of bytes occuied by individual array element <br /> nbytes -> Number of bytes occupied by entire array <br /> data -> base address of array <br /> strides -> Number of bytes that showuld be added to base address to reach the next array element <br /> <br /> ndim -> number of dimensions in the array Ex: 1D, 2D, 3D --> 1 2 3 4 .... <br /> shape -> Shape of array Ex: (2,3) <br/> size -> Number of elements in array

# In[280]:


#Integer array
arr=np.array([10,20,30,40,50,60])
print(arr.itemsize)
print(arr.nbytes)
print(arr.data)
print(arr.strides)
print(arr.ndim)
print(arr.shape)
print(arr.size)

#Float array
print()
arr=np.array([10,20,30,40,50,60],float)
print(arr.itemsize)
print(arr.nbytes)
print(arr.data)
print(arr.strides)
print(arr.ndim)
print(arr.shape)
print(arr.size)

#Complex numbers array
print()
arr=np.array([10,20,30,40,50,60],complex)
print(arr.itemsize)
print(arr.nbytes)
print(arr.data)
print(arr.strides)
print(arr.ndim)
print(arr.shape)
print(arr.size)


# # ARRAY OPERATORS
# #### Arithmetic Operators <br/> Statistical Operators <br/> Linear Algebra Operators <br/> Bitwise Operators <br/> Copying and Sorting Operators <br/> Comparision Operators

# ### Arithmetic operators

# In[286]:


#Arithmetic operators
a1=np.array([1,2,3])
a2=np.array([4,5,6])
print(a1)
print(a2)
#ADDITION
print(a1+a2)
#SUBTRACTION
print(a1-a2)
#MULTIPLICATION
print(a1*a2)
#DIVISION
print(a1/a2)
#FLOOR DIVISION
print(a1//a2)
print(a2//a1)
#MODULUS
print(a1%a2)
#POWER
print(a1**a2)
#ADDING CERTAIN VALUE TO EACH ELEMENT OF ARRAY
print(a1+3)
print(a2+3)
#SHORTHAND NOTATION
a1+=2
a2-=1
print(a1,a2)
a1*=2
print(a1)
a1=a1/2
print(a1)


# In[220]:


#OTHER OPERATIONS
#exp, sqrt, sin, cos, log

#Exponential
s=np.array([1,2,3,4])
print(np.exp(s))
#Squareroot
s=np.array([1,2,3,4])
print(np.sqrt(s))
#Sine
s=np.array([1,2,3,4])
print(np.sin(s))
#cosine
s=np.array([1,2,3,4])
print(np.cos(s))
#log
s=np.array([10,100,1000,10000])
print(np.log(s))


# ### Statistical operators

# In[314]:


#sum, min, max 
#sum, min, max via axes 0,1
#cumsum
a=np.array([[1,2,3],[4,5,6]])
print(a.sum())
print(a.sum(axis=0))
print(a.sum(axis=1))
print(a.min(axis=1))
print(a.min(axis=0))
print(a.min(axis=1))
print(a.max(axis=0))
print(a.max(axis=1))

#Cumulative sum
print(a.cumsum(axis=1)) #sum is done at end row or column based on axis value

print()
ar=np.array([1,2,3,4,5,6,7,8,9])
#Mean
print(np.mean(ar))
#Median
print(np.median(ar))
#Pearson Correlation coefficient
print(np.corrcoef(ar))
#Standard deveiation
print(np.std(a))


# ### Linear Algebra Operators

# #### *, @, dot(), transpose(), trace()  <br/> linalg -> <br/> linalg.inv() <br/> linalg.solve() 
# 

# In[315]:


#Multiplying corresponding elements of a and b
a=np.array([1,2,3])
b=np.array([4,5,6])
c=a*b
print(c)


# In[316]:


#Matrix multiplication
a=np.array([1,2,3])
b=np.array([4,5,6])
c=a@b
print(c)


# In[317]:


#Matrix multiplication
a=np.array([1,2,3])
b=np.array([4,5,6])
c=a.dot(b)
print(c)


# In[ ]:




