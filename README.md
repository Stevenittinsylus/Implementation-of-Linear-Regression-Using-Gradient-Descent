# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: steve nittin sylus
RegisterNumber:212224040331

```
```

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")


```

## Output
## DATA INFORMATION:
<img width="369" height="868" alt="image" src="https://github.com/user-attachments/assets/02d57cfa-b400-494d-afbb-78b02955ad24" />
## VALUE OF X:
<img width="338" height="867" alt="image" src="https://github.com/user-attachments/assets/a299a57f-a172-4a89-aeb0-57b2a67fc61f" />
## VALUE OF X1:
<img width="445" height="874" alt="image" src="https://github.com/user-attachments/assets/9a6bd1a4-c0d4-4774-835f-d8bd1876de1e" />
## PREDICTED VALUE:
<img width="412" height="74" alt="image" src="https://github.com/user-attachments/assets/c024817d-7f33-4ed6-abdc-a31d6ba4948d" />






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
