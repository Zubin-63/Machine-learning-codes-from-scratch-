import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#load the dataset make sure in the same folder or provide the absolute path
dataset=pd.read_csv(r"mall.csv")
k=7
ids=dataset.index
#randomly select k cluster centers here k=7
centers=np.random.choice(ids,k,replace=False)
centers.sort()
cl=list(range(7))
clusters={}
ck=0
# give each cluster center as some id
for i in range(dataset.shape[0]):
    if i in centers:
        clusters[i]=ck
        ck+=1
    else:
        clusters[i] = -1
# print(centers)
# print(clusters,sep=" ")
m=dataset.shape[0]
# function for clustering (k-mediod) distance metric is manhattan distance
def clustering(centers,clusters,m,dataset):
    for i in range(m):
        if i in centers:
            continue
        x=dataset.loc[i,'Annual Income (k$)']
        y=dataset.loc[i,'Spending Score (1-100)']
        mindist=99999999
        mincl=-1
        for ele in centers:
            xc=dataset.loc[ele,'Annual Income (k$)']
            yc=dataset.loc[ele,'Spending Score (1-100)']
            if(abs(x-xc)+abs(y-yc)<mindist):
                mindist=abs(x-xc)+abs(y-yc)
                mincl=clusters[ele]
        clusters[i]=mincl
    return clusters
#function for updating centers
def update_center(centers,clusters,m,dataset):
    ncs=centers.copy()
    for i in range(len(centers)):
        points=[]
        nc=centers[i]
        for j in range(len(clusters)):
            if(clusters[j]==i):
                points.append(j)
        mindist=9999999
        for j in range(len(points)):
            summ=0
            x=dataset.loc[points[j],'Annual Income (k$)']
            y=dataset.loc[points[j],'Spending Score (1-100)']
            for k in range(len(points)):
                xc=dataset.loc[points[k],'Annual Income (k$)']
                yc=dataset.loc[points[k],'Spending Score (1-100)']
                summ+=abs(x-xc)+abs(y-yc)
            if(summ<mindist):
                mindist=summ
                nc=points[j]
        ncs[i]=nc
    return ncs
# for convergence i have taken a fixed no of iterations you can take some epsilon 
for _ in range(10):
    clusters=clustering(centers, clusters, m, dataset)
    oldcenters=centers.copy()
    centers=update_center(centers, clusters, m, dataset)
    if np.array_equal(oldcenters, centers):
        print(_)
        print("Converged!")
        break
print(centers)
# Assigning different colors to clusters
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']

plt.figure(figsize=(8, 6))

# Plot each cluster with a unique color
for i in range(k):
    points = [j for j in dataset.index if clusters[j] == i]
    plt.scatter(dataset.loc[points, 'Annual Income (k$)'],
                dataset.loc[points, 'Spending Score (1-100)'],
                s=50, color=colors[i], label=f'Cluster {i}')

# Plot cluster centers
for i in range(k):
    plt.scatter(dataset.loc[centers[i], 'Annual Income (k$)'],
                dataset.loc[centers[i], 'Spending Score (1-100)'],
                s=200, marker='*', color='black', edgecolors='k', label=f'Center {i}')

# Labels and title
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Clustering')
plt.legend()
plt.show()






