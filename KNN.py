#=====================================PACKAGES AND DATA FORMATTING=================================

#==========================Importing needed packages and libraries======================
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
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

#=================================Defining the feature set====================================
df.columns
#To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values.astype(float)
X[0:5]
#print(X)
#Defining the labels or the value we will predict.
y = df['custcat'].values
y[0:5]

#====================================Normalizing the Data =====================================
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]
#print(X)

#========================================Train/Test Split=====================================
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


#====================================CLASSIFICATION WITH KNN===================================

#=====================================Training the algorithm==========================
#lets initialize at 4
k = 9
#Train the Model  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

#===============================Predicting with the trained algorithm=============================
yhat = neigh.predict(X_test)
yhat[0:5]
#print(yhat)


#======================================ACCURACY EVALUATION====================================

#=====using accuracy classification score(equals the jaccard_similarity_score) it computes subset accuracy=====
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

#------DO NOT FORGET: Higher the Jaccard score higher the accuracy of the classifier.------


#=========================LOOPING THROUGH MULTIPLE K's AND FINDING THE MOST ACCURATE ONE==================
#specifying the first value of the k's as 10
Ks = 10
#deducting 1 from the value of k
mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))
ConfustionMx = []
for n in range(1,Ks):
    
    #Train Model 
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    #use model to predict
    yhat = neigh.predict(X_test)
    #check the accuracy
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    print(mean_acc[n-1])
    #calculating the standard deviation of all the scores
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
#The best accuracy was with 0.34 with k= 9

#End of project. I tried with K = 20 and K = 10, with K = 20, the best value was from K = 16, and the difference between it and K = 10 with the best at K = 9 is just 2. I choose K = 10.


