#==========================Importing needed packages and libraries======================
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
#%matplotlib inline  #this is necessary in jupyter notebooks not Visual Studio Code, PC terminal or IDE's.

# The target field, called custcat, has four possible values that correspond to the four customer groups, as follows:
#  1- Basic Service 
#  2- E-Service 
#  3- Plus Service 
#  4- Total Service

#============================Reading the data into the project================================p
df = pd.read_csv("./teleCust1000t.csv")
datashow = df.head()
print(datashow)
#=============================Data visualization and analysis====================================
#we count how many different outcome classes are available in the custcat field
valuecounts = df['custcat'].value_counts()
print(valuecounts)
#we also plot an histogram of the data at the region column at a 50 interval
df.hist(column='region', bins=1)
#we also plot an histogram of the data at the tenure column at a 50 interval
df.hist(column='tenure', bins=10)
#we also plot an histogram of the data at the age column at a 50 interval
df.hist(column='age', bins=5)
#we also plot an histogram of the data at the marital column at a 50 interval
df.hist(column='marital', bins=1)
#we also plot an histogram of the data at the address column at a 50 interval
df.hist(column='address', bins=3)
#we also plot an histogram of the data at the income column at a 50 interval
df.hist(column='income', bins=50)
#we also plot an histogram of the data at the ed column at a 50 interval
df.hist(column='ed', bins=1)
#we also plot an histogram of the data at the employ column at a 50 interval
df.hist(column='employ', bins=5)
#we also plot an histogram of the data at the retire column at a 50 interval
df.hist(column='retire', bins=1)
#we also plot an histogram of the data at the gender column at a 50 interval
df.hist(column='gender', bins=1)
#we also plot an histogram of the data at the reside column at a 50 interval
df.hist(column='reside', bins=1)
#we also plot an histogram of the data at the custcat column at a 50 interval
df.hist(column='custcat', bins=1)

