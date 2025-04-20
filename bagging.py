import numpy as np
import pandas as pd
from math import e
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"iris.csv")
feature_map={0:"SepalLengthCm",1:"SepalWidthCm",2:"PetalLengthCm",3:"PetalWidthCm"}
#generating bootstrap samples
def generate_bootstrap_samples(dataset, sample_size=100, num_subsets=4):
    bootstrap_samples = []
    for _ in range(num_subsets):
        sample = dataset.sample(n=sample_size, replace=True, random_state=np.random.randint(10000))
        bootstrap_samples.append(sample)

    return bootstrap_samples
bootstrap_samples = generate_bootstrap_samples(dataset, sample_size=100, num_subsets=4)


# Decision Tree Implementation
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def isleaf(self):
        return self.value is not None

    def print_tree(self, depth=0):
        if self.isleaf():
            print(f"{'  ' * depth}Leaf: {self.value}")
        else:
            print(f"{'  ' * depth}Node: [Feature {feature_map[self.feature]}, Threshold {self.threshold}]")
            if self.left:
                print(f"{'  ' * (depth + 1)}Left -> ", end="")
                self.left.print_tree(depth + 1)
            if self.right:
                print(f"{'  ' * (depth + 1)}Right -> ", end="")
                self.right.print_tree(depth + 1)


def entropy(species):
    values, counts = np.unique(species, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

#getting the best candidate for an attribute
def candidate_gain(species, attr, split_val):
    left = species[attr <= split_val]
    right = species[attr > split_val]
    return ((len(left) / len(attr)) * entropy(left)) + ((len(right) / len(attr)) * entropy(right))

#get the best attribute by information gain
def attr_gain(attr_col, species):
    best_split, best_gain = None, -np.inf
    sorted_vals = np.sort(attr_col)
    for i in range(len(sorted_vals) - 1):
        split_val = (sorted_vals[i] + sorted_vals[i + 1]) / 2
        gain = entropy(species) - candidate_gain(species, attr_col, split_val)
        if gain > best_gain:
            best_gain, best_split = gain, split_val
    return best_split, best_gain



def build_tree(data, species):
    if len(np.unique(species)) == 1:
        return TreeNode(value=species[0])
    best_attr, best_threshold = None, None
    best_gain = -np.inf
    for attr in range(data.shape[1]):
        split_val, gain = attr_gain(data[:, attr], species)

        if gain > best_gain:
            best_attr, best_threshold, best_gain = attr, split_val, gain
    if best_attr is None:
        return TreeNode(value=np.bincount(species).argmax())
    left_indices = data[:, best_attr] <= best_threshold
    right_indices = data[:, best_attr] > best_threshold
    left_child = build_tree(data[left_indices], species[left_indices])
    right_child = build_tree(data[right_indices], species[right_indices])
    return TreeNode(feature=best_attr, threshold=best_threshold, left=left_child, right=right_child)

def print_all_trees(trees):
    for i, tree in enumerate(trees):
        print(f"Tree {i+1}:")
        tree.print_tree()
        print("\n" + "-" * 40 + "\n")
def predict(tree, sample):
    if tree.isleaf():
        return tree.value
    return predict(tree.left, sample) if sample[tree.feature] <= tree.threshold else predict(tree.right, sample)


# Bagging Implementation
def bagging_predict(trees, sample):
    predictions = [predict(tree, sample) for tree in trees]
    return max(set(predictions), key=predictions.count)


def train_bagging_forest(dataset, num_trees=4, sample_size=100):
    bootstrap_samples = generate_bootstrap_samples(dataset, sample_size, num_trees)
    trees = []
    for sample in bootstrap_samples:
        data = sample.iloc[:, :-1].values  # Features
        species = sample.iloc[:, -1].values  # Labels
        tree = build_tree(data, species)
        trees.append(tree)
    return trees


# Training and Testing the Bagging Random Forest
dataset = dataset.drop(columns=["Id"]) 

train_ratio=0.8
m=dataset.shape[0]
training_size=round(train_ratio*m)
test_size=m-training_size
indices=np.random.permutation(m)
training_indices=indices[:training_size]
testing_indices=indices[training_size:]
training_dataset = dataset.iloc[training_indices].reset_index(drop=True)
testing_dataset = dataset.iloc[testing_indices].reset_index(drop=True)

forest = train_bagging_forest(training_dataset, num_trees=4, sample_size=100)
test_data = testing_dataset.iloc[:, :-1].values
test_species = testing_dataset.iloc[:, -1].values
#printing all trees and calculating accuracy
correct = sum(1 for i in range(len(test_data)) if bagging_predict(forest, test_data[i]) == test_species[i])
print_all_trees(forest)
print("Bagging Random Forest Accuracy: {:.2f}%".format((correct / len(test_data)) * 100))
