# Exp-02 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1. **Load and Explore Dataset**  
   Read the CSV file (`student_scores.csv`) using Pandas. Display the head, tail, and structure (info) of the dataset.

2. **Prepare Input and Output Variables**  
   Extract the independent variable (`Hours`) as `x` and the dependent variable (`Scores`) as `y`. Display their shapes to confirm correct extraction.

3. **Split the Dataset**  
   Use `train_test_split()` to divide the dataset into training and testing sets (2:1 ratio).

4. **Train the Regression Model**  
   Create a `LinearRegression` object and train it on the training data using `fit()`.

5. **Predict and Evaluate the Model**  
   Predict the test set results using `predict()`. Evaluate using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

6. **Visualize the Results**  
   Plot the regression line along with both the training set and test set using `matplotlib.pyplot`.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SURIYA M
RegisterNumber: 212223110055
*/
print("Name: SURIYA M\nReg no: 212223110055\n")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head(10)
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
##plotting for training data
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
##plotting for test data
plt.scatter(x_test,y_test,color="grey")
plt.plot(x_test,y_pred,color="purple")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("Name: SURIYA M\nReg no: 212223110055\n")

```

## Output:

### Head
![image](https://github.com/user-attachments/assets/7eb0cae6-dfbc-4087-b289-2b8463354245)
![Screenshot 2025-03-05 225409](https://github.com/user-attachments/assets/eb01a188-a7c0-401f-a38e-de6422d7e9e9)
### Tail
![Screenshot 2025-03-05 225429](https://github.com/user-attachments/assets/9d3a05fd-bfb4-43c1-9885-35206f5e73f6)
### X Values
![Screenshot 2025-03-05 225448](https://github.com/user-attachments/assets/5e8daf0a-6563-425d-85dd-03b64bd841ba)
### Y Values
![Screenshot 2025-03-05 225505](https://github.com/user-attachments/assets/1b216402-94fa-47dd-90f2-38f79ed4301c)
### Y_Predicted Values
![Screenshot 2025-03-05 225708](https://github.com/user-attachments/assets/5792a960-7bc6-46b7-b915-f0bb02dab370)
### Y_Test Values
![Screenshot 2025-03-05 225752](https://github.com/user-attachments/assets/bafb8ffb-52c1-4894-9638-e4a8d62b7157)
### MSE,MAE AND RMSE
![Screenshot 2025-03-05 225816](https://github.com/user-attachments/assets/aa347465-140c-4fa2-8a95-fd83c25645ae)
### Training Set
![Screenshot 2025-03-05 225841](https://github.com/user-attachments/assets/9f8b488c-5d47-4d80-b68a-449834bbd6b6)
### Testing Set
![Screenshot 2025-03-05 225905](https://github.com/user-attachments/assets/f319fa47-0d6e-40cc-8794-7d1ccd61ef0c)
![image](https://github.com/user-attachments/assets/f123130b-4646-4d42-a093-69b3f908c387)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
