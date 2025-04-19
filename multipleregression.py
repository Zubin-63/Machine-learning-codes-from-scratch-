import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
iterations=1000
#load the dataset make sure in the same folder or provide the absolute path
dataset=pd.read_csv(r"boston.csv")
mhp=dataset['median home price']
crime_rate=dataset['Crime Rate']
no2=dataset['NO2 concentration']
resid=dataset['Residential Proportion']
avg_rooms_dwell=dataset['Average Rooms/Dwelling.']
dist=dataset['Distance to Employment Centres']
taxx=dataset['ValueProperty/tax rate']
lower=dataset['Lower Status Percent']
n=8
m=len(no2)
alpha=0.01
thetas=[0]*(n)
X = np.column_stack((
    np.ones(len(mhp)),
    crime_rate,
    no2,
    resid,
    avg_rooms_dwell,
    dist,
    taxx,
    lower
))

thetas=np.array(thetas)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

# Handle division by zero: Replace 0 in standard deviation with 1 to avoid NaN
X_std = np.where(X_std == 0, 1, X_std)

X = (X - X_mean) / X_std

# Normalize target
mhp_mean = np.mean(mhp)
mhp_std = np.std(mhp)
mhp = (mhp - mhp_mean) / mhp_std
#functions for predicting,calculating cost or loss function and updating thetas
def predicted(X,thetas):
    predicted_y=np.dot(X,thetas)
    return predicted_y
def cost_func(predicted_y,mhp):
    return (np.mean((predicted_y-mhp)**2))/2
def update(predicted_y,thetas,m,mhp,X):
    error=mhp-predicted_y
    grd=(np.dot(X.T,error))/m
    thetas=thetas+alpha*grd
    return thetas
pcost=[]
best_thetas=thetas
minncost=99999999999999
# doing for 1000 iterations
while(iterations>0):
    predicted_y=predicted(X,thetas)
    cost=cost_func(predicted_y,mhp)
    pcost.append(cost)
    if(minncost>cost):
        minncost=cost
        best_thetas=thetas
    thetas=update(predicted_y,thetas,m,mhp,X)
    iterations-=1
print("Best thetas: ",sep=" ")
print(best_thetas)
print("Best cost: ",sep=" ")
print(minncost)
epochs=range(1,len(pcost)+1)
plt.figure(figsize=(10,7))
plt.plot(epochs,pcost,color='red')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost vs Epochs')
plt.grid(True)
plt.show()








