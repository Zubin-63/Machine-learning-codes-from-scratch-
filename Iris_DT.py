import numpy as np
import pandas as pd
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
total_class=3
species_map={0:"Iris-setosa",1:"Iris-versicolor",2:"Iris-virginia"}
feature_map={0:"SepalLengthCm",1:"SepalWidthCm",2:"PetalLengthCm",3:"PetalWidthCm"}
#split data into training and testing
train_ratio=0.8
training_size=round(train_ratio*m)
test_size=m-training_size
indices=np.random.permutation(m)
train_indices=indices[:training_size]
test_indices=indices[training_size:]

training_spl = [spl[i] for i in train_indices]
training_spw = [spw[i] for i in train_indices]
training_ptl = [ptl[i] for i in train_indices]
training_ptw = [ptw[i] for i in train_indices]
training_species = [cspecies[i] for i in train_indices]

test_spl = [spl[i] for i in test_indices]
test_spw = [spw[i] for i in test_indices]
test_ptl = [ptl[i] for i in test_indices]
test_ptw = [ptw[i] for i in test_indices]
test_species = [cspecies[i] for i in test_indices]
data=np.column_stack((training_spl,training_spw,training_ptl,training_ptw))


#class initialized for decision tree
class TreeNode:
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature = feature          
        self.threshold = threshold      
        self.left = left            
        self.right = right              
        self.value = value
    def isleaf(self):
        return self.value is not None
    def __repr__(self):
        if self.is_leaf():
            return f"Leaf(value={self.value})"
        return f"Node(feature={self.feature}, threshold={self.threshold})"
    def print_tree(self, depth=0):
        if self.isleaf():
            print(f"{'  ' * depth}Leaf: {species_map[self.value]}")
        else:
            print(f"{'  ' * depth}Node: [Feature {feature_map[self.feature]}, Threshold {self.threshold}]")
            if self.left:
                print(f"{'  ' * (depth + 1)}Left -> ", end="")
                self.left.print_tree(depth + 1)
            if self.right:
                print(f"{'  ' * (depth + 1)}Right -> ", end="")
                self.right.print_tree(depth + 1)
                

#functions to calculate entropy and information gain
def entropy(tt_species):
    if len(tt_species) == 0:
        return 0
    _, counts = np.unique(tt_species, return_counts=True)
    probs = counts / len(tt_species)
    return -np.sum(probs * np.log2(probs))

#selecting best candidate of a feature(continous value)
def candidate_gain(tt_species,test_attr,split_val):
    leftone=[]
    rightone=[]
    for i in range(len(test_attr)):
        if(test_attr[i]<=split_val):
            leftone.append(tt_species[i])
        else:
            rightone.append(tt_species[i])
    leftentropy=entropy(leftone)
    rightentropy=entropy(rightone)
    return ((len(leftone)/len(test_attr))*leftentropy)+((len(rightone)/len(test_attr))*rightentropy)
#selecting best feature
def attr_gain(test_attr,tt_species):
    sta=np.sort(test_attr)
    best_split=0
    best_entr=0
    entr=entropy(tt_species)
    for i in range(len(sta)-1):
        split_val=(sta[i]+sta[i+1])/2
        wsplit=candidate_gain(tt_species,test_attr,split_val)
        if entr-wsplit>best_entr:
            best_entr=entr-wsplit
            best_split=split_val
    return [best_split,best_entr]

#building tree

def build_Tree(data,tt_species):
    tt_species=np.array(tt_species)
    unq=np.unique(tt_species)
    if(len(data)==0):
        if(len(unq)==0):
            return None
        freq_class=tt_species.mode()[0]
        return TreeNode(value=freq_class)
    if(len(unq)==1):
        return TreeNode(value=unq[0])
    
    best_attr=None
    best_threshold=None
    best_entr=-99999
    for attr in range(data.shape[1]):
        atinfo=attr_gain(data[:,attr],tt_species)
        if(atinfo[1]>best_entr):
            best_attr=attr
            best_threshold=atinfo[0]
    if best_attr is None:
        freq_class = pd.Series(tt_species).mode()[0]
        return TreeNode(value=freq_class)
    left_indices = data[:, best_attr] <= best_threshold
    right_indices = data[:, best_attr] > best_threshold
    new_data=np.delete(data,best_attr,axis=1)
    left_child=build_Tree(new_data[left_indices,:],tt_species[left_indices])
    right_child=build_Tree(new_data[right_indices,:],tt_species[right_indices])
    return TreeNode(feature=best_attr,threshold=best_threshold,left=left_child,right=right_child)
#predict function
def predict(tree,sample):
    if(tree.isleaf()):
        return tree.value
    if(sample[tree.feature]<=tree.threshold):
        return predict(tree.left,sample)
    else:
        return predict(tree.right,sample)

#printing tree and calculating accuracy

Tree=build_Tree(data,training_species)
Tree.print_tree()
testing_data=np.column_stack((test_spl,test_spw,test_ptl,test_ptw))
cnt=0
correct=0
for samples in testing_data:
    predicted_class=predict(Tree,samples)
    if(predicted_class==test_species[cnt]):
        correct+=1
    cnt+=1
print("Accuracy :"+str((correct/cnt)*100)+"%")





