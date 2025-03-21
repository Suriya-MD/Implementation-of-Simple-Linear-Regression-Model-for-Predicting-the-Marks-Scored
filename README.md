# Exp-02 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Suriya M
RegisterNumber:  212223110055
*/

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

dataset.info()

x=dataset.iloc[:,:-1].values
print(x)

y=dataset.iloc[:,1].values
print(y)

```

## Output:
![image](https://github.com/user-attachments/assets/5edcadd1-8314-4da3-975b-2a74f884b0c0)

```
print(x.shape)
print(y.shape)
```


## Output :
![image](https://github.com/user-attachments/assets/52d43a2c-30c6-4c79-bedc-85883dfd13b3)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
```

## Output :
![image](https://github.com/user-attachments/assets/6eb6980c-cad6-48db-880a-698e94799169)

```
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output :
![image](https://github.com/user-attachments/assets/454719e9-17db-49df-ade8-f8fc438a67aa)

```
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title("Training Set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="black")
plt.plot(x_test,reg.predict(x_test),color="green")
plt.title("Test Set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output :
![image](https://github.com/user-attachments/assets/c357aa7e-8778-4391-ab61-3f60cd470e8b)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
