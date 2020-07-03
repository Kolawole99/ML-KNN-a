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
datashow = df.head(10)
print(datashow)
#=============================Data visualization and analysis====================================
#we count how many different outcome classes are available in the custcat field
df['custcat'].value_counts()
#we also plot an histogram of the data at the income column at a 50 interval
df.hist(column='income', bins=50)




