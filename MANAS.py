import numpy as np
import pandas as pd

data = pd.read_csv("weatherHistory.csv")

X = data[['Humidity']].values   
y = data['Temperature (C)'].values 

X = (X - X.mean()) / X.std()


i = np.c_[np.ones((X.shape[0], 1)), X]

def gradient_descent(X, y, learn=0.01, itr=1000):
    m, n = X.shape
    theta = np.zeros(n)  
    
    for _ in range(itr):
        predictions = X.dot(theta)       
        errors = predictions - y          
        gradients = (2/m) * X.T.dot(errors)
        theta -= learn * gradients           
    
    return theta

theta = gradient_descent(i, y, learn=0.01, itr=1000)

print("Intercept:", theta[0])
print("Slope:", theta[1])


y_pred = i.dot(theta)


rmse = np.sqrt(np.mean((y - y_pred)**2))
print("RMSE:", rmse)
