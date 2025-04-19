import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#load the dataset make sure in the same folder or provide the absolute path
dataset=pd.read_csv(r"iris.csv")
spl=dataset['SepalLengthCm']
spw=dataset['SepalWidthCm']
ptl=dataset['PetalLengthCm']
ptw=dataset['PetalWidthCm']
# normalizing each feature
spl=(spl-np.mean(spl))/np.std(spl)
spw=(spw-np.mean(spw))/np.std(spw)
ptl=(ptl-np.mean(ptl))/np.std(ptl)
ptw=(ptw-np.mean(ptw))/np.std(ptw)
X=np.column_stack((
    spl,
    spw,
    ptl,
    ptw
))
mapp={0:spl,1:spw,2:ptl,3:ptw}
m=X.shape[0]
n=X.shape[1]
# calculating covariance matrix
covmat=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        cv=0
        for k in range(m):
            cv+=(X[k,i]-np.mean(mapp[i]))*(X[k,j]-np.mean(mapp[j]))
        covmat[i,j]=cv/(m-1)
#print(covmat)
eigen_val,eigen_vect=np.linalg.eig(covmat)
mappp={}
i=0
# print(eigen_val)
# print(eigen_vect)
for ele in eigen_val:
    mappp[ele]=i
    i+=1
#print(mapp)
eigen_val.sort()
nev=eigen_vect.copy()
for i in range(eigen_vect.shape[0]):
    nev[:,i]=eigen_vect[:,mappp[eigen_val[i]]]
# print(eigen_val)
# print(nev)
top_2=nev[:,-2:]
X_red=X.dot(top_2)
print("Principal components(2): "+str(X_red))
plt.scatter(X_red[:,1],X_red[:,0])
plt.show()