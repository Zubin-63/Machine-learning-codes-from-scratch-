import numpy as np
import pandas as pd
from math import e
import matplotlib.pyplot as plt
#load the dataset 
dataset=pd.read_csv(r"iris.csv")
spl=dataset['SepalLengthCm']
spw=dataset['SepalWidthCm']
ptl=dataset['PetalLengthCm']
ptw=dataset['PetalWidthCm']
species=dataset['Species']
cspecies=[]
m=len(ptl)
for i in range(m):
    if(species[i]=="Iris-setosa"):
        cspecies.append(0)
    elif(species[i]=="Iris-versicolor"):
        cspecies.append(1)
    else:
        cspecies.append(2)
#initalize parameters here only training data is taken further division into training and testing data can be done.
n=5
itr=1000
alpha=0.01
k=3
thetas=np.zeros((n,k))
true_y=np.zeros(((m,k)))
for i in range(m):
   true_y[i][cspecies[i]]=1
X=np.column_stack((
    np.ones(m),
    spl,
    spw,
    ptl,
    ptw
))
# prediction cost_func and update functions are written
def predict(X,thetas):
    hypo=np.dot(X,thetas)
    ehypo=np.exp(hypo)
    denom=[]
    for x in ehypo:
        denom.append(np.sum(x))
    mm=ehypo.shape[0]
    nn=ehypo.shape[1]
    for i in range(mm):
        for j in range(nn):
            ehypo[i][j]=ehypo[i][j]/denom[i]
    return ehypo
def cost_func(predicted_y,m,true_y):
    cs=-np.sum(true_y*np.log2(predicted_y))/m
    return cs
def update(true_y,thetas,X,m,alpha,predicted_y):
    error=predicted_y-true_y
    grd=np.dot(X.T,error)/m
    thetas=thetas-alpha*grd
    return thetas
pcost=[]
best_thetas=[]
mincost=999999999999
# doing for 1000 iterations
while(itr>0):
    predicted_y=predict(X,thetas)
    cost=cost_func(predicted_y,m,true_y)
    pcost.append(cost)
    if(cost<mincost):
        mincost=cost
        best_thetas=thetas
    thetas=update(true_y,thetas,X,m,alpha,predicted_y)
    itr-=1
#print(mincost)
#print(best_thetas)
predicted_y=predict(X,best_thetas)
training_output=[]
#calculating accuracy
for i in range(m):
    maxx=0
    idx=0
    for j in range(k):
        if(predicted_y[i][j]>maxx):
            maxx=predicted_y[i][j]
            idx=j
    training_output.append(idx)
correct=0
for i in range(m):
    if(training_output[i]==cspecies[i]):
        correct+=1
print("Accuracy of the training data is :"+str((correct/m)*100)+"%")
#plot
epochs=range(1,len(pcost)+1)
plt.figure(figsize=(10,7))
plt.plot(epochs,pcost,color='red')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost vs Epochs')
plt.grid(True)
plt.show()




