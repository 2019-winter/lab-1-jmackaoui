---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
by Justis Mackaoui


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


**YOUR ANSWER HERE**


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1

```python
import numpy as np

a = np.full(shape = (6,4), fill_value = 2)
a
```

## Exercise 2

```python
b = np.ones(shape = (6,4), dtype = int)
np.fill_diagonal(b, val = 3)
b
```

## Exercise 3

```python
ab = a*b
ab
```

## Exercise 4

```python
aTb = np.dot(a.transpose(), b)
print(aTb)

abT = np.dot(a, b.transpose())
print('\n' + str(abT))
```

## Exercise 5

```python
def print_something():
    print('something')

print_something()
```

## Exercise 6

```python
def random_array():
    nrows = np.random.randint(low = 1, high = 10,dtype = int)
    ncols = np.random.randint(low = 1, high = 10,dtype = int)
    arr = np.random.rand(nrows, ncols)
    print(arr)
    print('\nsum: %f' % arr.sum())
    print('mean: %f' % arr.mean())
    print('stdev: %f' % arr.std())

random_array()
```

## Exercise 7

```python
sample_array = np.array([[0,2,1], [1,1,1], [2,4,5]])
```

```python
def count_ones(array):
    count = 0
    for elem in np.nditer(array):
        if elem == 1:
            count += 1
    return count

count_ones(sample_array)
```

```python
def count_ones_where(array):
    return np.count_nonzero(np.where(array == 1, True, False))

count_ones_where(sample_array)
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
import pandas as pd

a = pd.DataFrame(np.full(shape = (6,4), fill_value = 2))
a
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
b = np.ones(shape = (6,4), dtype = int)
np.fill_diagonal(b, val = 3)
a = pd.DataFrame(b)
a
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
a*b
```

## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
sample_array = np.array([[0,2,1], [1,1,1], [2,4,5]])
df = pd.DataFrame(sample_array)
sum(df[df == 1].sum())
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
titanic_df.name
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
titanic_df.set_index('sex',inplace=True)
```

```python
titanic_df[titanic_df.index == 'female']
```

## Exercise 14
How do you reset the index?

```python
titanic_df.reset_index(inplace = True)
titanic_df.head()
```
