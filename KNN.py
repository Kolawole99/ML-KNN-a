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

#========================================== TARGET DECLARATION===============================
# The target field, called custcat, has four possible values that correspond to the four customer groups, as follows:
#  1- Basic Service 
#  2- E-Service 
#  3- Plus Service 
#  4- Total Service

#============================Reading the data into the project================================
df = pd.read_csv("./teleCust1000t.csv")
datashow = df.head()
print(datashow)
#=============================Data visualization and analysis====================================
#we count how many different outcome classes are available in the custcat field
valuecounts = df['custcat'].value_counts()
print(valuecounts)
#we also plot an histogram of the data at the region; there are just three regions in the dataset
region = df['region']
plt.hist(region, bins=3)
#we also plot an histogram of the data at the tenure column
tenure = df['tenure']
plt.hist(tenure, bins=72)
#we also plot an histogram of the data at the age column
age = df['age']
plt.hist(age, bins=10)
#we also plot an histogram of the data at the income column
income = df['income']
plt.hist(income, bins=50)
#plt.show()

#===========================Defining the feature set==========================
df.columns
