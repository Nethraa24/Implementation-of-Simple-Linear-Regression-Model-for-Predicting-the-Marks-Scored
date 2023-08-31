# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.

2. Set variables for assigning dataset values.

3. Import linear regression from sklearn.

4. Assign the points for representing in the graph.

5. Predict the regression for marks by using the representation of the graph.

6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: J.Nethraa
RegisterNumber:  212222100031

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
df.head()
df.tail()

## segregating data to variables

X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y

## graph plotting for training data

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

## Displaying predicted values

Y_pred

## Displaying actual values

Y_test

## Graph plot for training data

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

## Graph plot for test data

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

*/
```

## Output:
![263522726-8005dd99-5a09-4708-aaa2-0869a39e51ad](https://github.com/Nethraa24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215786/c80e9416-47f2-46b7-859d-eb60a47259bc)

![263522732-64ecbe05-5cd2-4719-8d41-71933e06d3a6](https://github.com/Nethraa24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215786/282f86ef-07ae-484a-a5f6-f2340717ac39)

## Array values of X
![263522752-bd10a07b-9dfa-4b25-bbd3-0fe8e9c35e01](https://github.com/Nethraa24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215786/f37167f0-4bb6-47c1-a19a-a10f72d722b1)

## Array values of Y
![263522769-c04e68b0-f11e-472b-a165-a7a688ad4d3f](https://github.com/Nethraa24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215786/a0b7d5a0-348e-4e68-b083-8ed61e1624c8)

## Values of Y prediction
![263522779-ad3d2f27-96e5-4e5c-8646-4d0071d68b33](https://github.com/Nethraa24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215786/8452b485-a423-4090-8c58-fd588dc2eacc)

## Values of Y test
![263522786-cb0eee3e-046b-4ddd-bdbd-c5e4f0322b4c](https://github.com/Nethraa24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215786/ee8bfdf3-6b0b-44bb-87b2-bb3df3641826)

## Training set graph
![263522801-d4bbb6ea-76d7-4827-aa5a-8196ac72ae83](https://github.com/Nethraa24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215786/013b0ec6-5c3b-4e73-a828-7a1d5184883f)

## Testing set graph
![263522812-8b3b3ebc-5020-4f2f-9e76-687a17f1c91c](https://github.com/Nethraa24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215786/d24b8d1f-48d9-4975-8d84-af280687d182)

## Value of MSE,MAE & RMSE
![263522827-eab2546b-aa47-4bc6-b359-b8bd6bf5b2ea](https://github.com/Nethraa24/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215786/e5c86c86-2ea5-4c17-ae23-fa749dddda3f)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
