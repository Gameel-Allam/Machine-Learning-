#import needed packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

#read the dataset
data=pd.read_csv("C:/Users/NOUR SOFT/Downloads/CarPrice_Assignment.csv")
print(data)
print(data.describe())

#spilt the data into train and test
dataset=data.drop(['CarName','fueltype','doornumber','carbody','drivewheel','enginelocation','cylindernumber','aspiration','fuelsystem','enginetype'],axis=1)
X=dataset.drop('price',axis=1)
Y=dataset['price']
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=100)

#visualize the train and the test data  
plt.scatter(X_train['horsepower'],Y_train,label='Train Data',color='r')
plt.scatter(X_test['horsepower'],Y_test,label='Test Data',color='g')
plt.title('Data Set Spilit')
plt.show()

#create linear regression object to use 
lr=LinearRegression()
lr.fit(X_train,Y_train)
prediction=lr.predict(X_test)
print(r2_score(Y_test,prediction))

#visualize the results 
plt.scatter(X_test['horsepower'],prediction,label='Predicate price',color='y')
plt.scatter(X_test['horsepower'],Y_test,label='Actual price',color='b')
plt.show()
